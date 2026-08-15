import Foundation

/// Bounded LFU cache of dequantized routed experts over a `.qpack` container
/// — the CPU twin of `ExpertCache`: same policy (LFU with recency tie-break,
/// batch protection via same-tick freshness), plain memory instead of Metal
/// buffers. This is what lets the CPU engine serve containers, whose routed
/// experts live in `packed_experts/` blobs rather than `model.safetensors`.
public final class CPUExpertCache {

    /// One expert's three matrices, row-major f32:
    /// gate (inter, hidden), up (inter, hidden), down (hidden, inter).
    public struct ExpertWeights {
        let gate: [Float]
        let up: [Float]
        let down: [Float]
    }

    let reader: QpackExpertReader
    private let gate: (w: Qpack.Section, s: Qpack.Section, b: Qpack.Section)
    private let up: (w: Qpack.Section, s: Qpack.Section, b: Qpack.Section)
    private let down: (w: Qpack.Section, s: Qpack.Section, b: Qpack.Section)
    private let spec: Checkpoint.QuantSpec
    private let inter: Int
    private let hidden: Int

    private var entries: [Int64: ExpertWeights] = [:]
    private var freq: [Int64: Int] = [:]
    private var lastUse: [Int64: Int] = [:]
    private var tick = 0
    private let maxEntries: Int

    public private(set) var hits = 0
    public private(set) var misses = 0

    /// `budgetBytes` bounds the f32 resident set (not the quantized blobs).
    init(containerDir: URL, spec: Checkpoint.QuantSpec,
         inter: Int, hidden: Int, budgetBytes: Int) throws {
        let reader = try QpackExpertReader(containerDir: containerDir)
        self.reader = reader
        guard spec.bits == 4 || spec.bits == 8 else {
            throw Checkpoint.Error.unsupportedBits(spec.bits)
        }
        func trio(_ proj: String) throws -> (Qpack.Section, Qpack.Section, Qpack.Section) {
            guard let w = reader.section(proj + ".weight"),
                  let s = reader.section(proj + ".scales"),
                  let b = reader.section(proj + ".biases")
            else { throw Checkpoint.Error.badShape("container expert sections missing for \(proj)") }
            return (w, s, b)
        }
        gate = try trio("gate_proj")
        up = try trio("up_proj")
        down = try trio("down_proj")
        self.spec = spec
        self.inter = inter
        self.hidden = hidden
        let bytesPerExpert = 3 * inter * hidden * 4
        let total = reader.layout.expertCount * reader.layout.layerCount
        maxEntries = min(max(budgetBytes / max(bytesPerExpert, 1), 16), total)
    }

    public var residentCount: Int { entries.count }
    public var residentBytes: Int { entries.count * 3 * inter * hidden * 4 }

    private func key(_ layer: Int, _ expert: Int) -> Int64 {
        Int64(layer) << 32 | Int64(expert)
    }

    /// Weights for one expert, dequantizing from its blob on a miss. A victim
    /// is always strictly less used (or equally used but older) than every
    /// member fetched earlier in the same batch, so a batch can never evict
    /// itself — the same guarantee ExpertCache gets from protected slots.
    func weights(layer: Int, expert: Int) throws -> ExpertWeights {
        tick += 1
        let k = key(layer, expert)
        freq[k, default: 0] += 1
        if let hit = entries[k] {
            hits += 1
            lastUse[k] = tick
            return hit
        }
        misses += 1
        let w = try dequantize(layer: layer, expert: expert)
        if entries.count >= maxEntries, let victim = victimKey() {
            entries.removeValue(forKey: victim)
            freq.removeValue(forKey: victim)
            lastUse.removeValue(forKey: victim)
        }
        entries[k] = w
        lastUse[k] = tick
        return w
    }

    /// LFU victim with recency tie-break, same scoring as ExpertCache.
    private func victimKey() -> Int64? {
        var victim: Int64?
        var victimScore = (Int.max, Int.max)
        for (k, f) in freq {
            let score = (f, lastUse[k] ?? 0)
            if score < victimScore {
                victimScore = score
                victim = k
            }
        }
        return victim
    }

    /// Reads the expert blob (one pread) and dequantizes its six sections
    /// with the same MLX affine math as `Checkpoint.dequantized`:
    /// w[i] = scale[g] * q[i] + bias[g], uint32-packed along the last axis.
    private func dequantize(layer: Int, expert: Int) throws -> ExpertWeights {
        var blob = [UInt8](repeating: 0, count: reader.layout.expertStride)
        try blob.withUnsafeMutableBytes { raw in
            try reader.readExpert(layer: layer, expert: expert, into: raw.baseAddress!)
        }
        return try blob.withUnsafeBytes { raw in
            ExpertWeights(
                gate: try Self.dequantMatrix(
                    raw, weight: gate.w, scales: gate.s, biases: gate.b,
                    rows: inter, inDim: hidden, spec: spec),
                up: try Self.dequantMatrix(
                    raw, weight: up.w, scales: up.s, biases: up.b,
                    rows: inter, inDim: hidden, spec: spec),
                down: try Self.dequantMatrix(
                    raw, weight: down.w, scales: down.s, biases: down.b,
                    rows: hidden, inDim: inter, spec: spec)
            )
        }
    }

    /// Dequantizes one (rows, inDim) matrix from its three blob sections.
    static func dequantMatrix(_ raw: UnsafeRawBufferPointer,
                              weight: Qpack.Section, scales: Qpack.Section, biases: Qpack.Section,
                              rows: Int, inDim: Int, spec: Checkpoint.QuantSpec) throws -> [Float] {
        let perWord = 32 / spec.bits
        let mask = UInt32((1 << spec.bits) - 1)
        let packedCols = inDim / perWord
        let groupsPerRow = inDim / spec.groupSize
        guard weight.dtype == "U32", weight.shape == [rows, packedCols],
              scales.shape == [rows, groupsPerRow], biases.shape.count == 2
        else { throw Checkpoint.Error.badShape(weight.name) }

        let s = try Self.floats(raw, scales, count: rows * groupsPerRow)
        let b = try Self.floats(raw, biases, count: rows * groupsPerRow)

        var out = [Float](repeating: 0, count: rows * inDim)
        out.withUnsafeMutableBufferPointer { o in
            for r in 0..<rows {
                for w in 0..<packedCols {
                    var word = raw.loadUnaligned(
                        fromByteOffset: weight.offset + (r * packedCols + w) * 4, as: UInt32.self)
                    let colBase = w * perWord
                    for j in 0..<perWord {
                        let col = colBase + j
                        let g = col / spec.groupSize
                        let q = Float(word & mask)
                        o[r * inDim + col] = s[r * groupsPerRow + g] * q + b[r * groupsPerRow + g]
                        word >>= UInt32(spec.bits)
                    }
                }
            }
        }
        return out
    }

    /// Float array from a blob section (scales/biases: F32, F16 or BF16).
    static func floats(_ raw: UnsafeRawBufferPointer, _ section: Qpack.Section, count: Int) throws -> [Float] {
        let byteOffset = section.offset
        switch section.dtype {
        case "F32":
            return (0..<count).map { raw.loadUnaligned(fromByteOffset: byteOffset + $0 * 4, as: Float.self) }
        case "F16":
            return (0..<count).map {
                SafetensorsFile.f16ToF32(raw.loadUnaligned(fromByteOffset: byteOffset + $0 * 2, as: UInt16.self))
            }
        case "BF16":
            return (0..<count).map {
                Float(bitPattern: UInt32(raw.loadUnaligned(fromByteOffset: byteOffset + $0 * 2, as: UInt16.self)) << 16)
            }
        default:
            throw SafetensorsFile.Error.unsupportedDtype(section.dtype, tensor: section.name)
        }
    }
}

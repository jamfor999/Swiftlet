import Foundation

/// Bounded cache of raw quantized expert blobs over a `.qpack` container —
/// the CPU twin of `ExpertCache`: fixed slot pool filled by single preads,
/// LFU eviction with recency tie-break, batch protection, memory use exactly
/// `slots * expertStride`. Unlike the f32 reference path, experts stay
/// packed and feed `QuantizedGEMV` directly, so RAM cost per expert is the
/// on-disk blob size and decode never pays a dequant pass.
public final class CPUExpertCache {
    /// Byte offsets of one matrix's three sections inside an expert blob.
    struct MatrixSections {
        let weight: Int
        let scales: Int
        let biases: Int
    }

    let reader: QpackExpertReader
    public let stride: Int
    let bits: Int
    let groupSize: Int
    let inter: Int
    let hidden: Int
    let scalesDtype: String
    let gate: MatrixSections
    let up: MatrixSections
    let down: MatrixSections

    private var slots: [UnsafeMutableRawPointer?] = []
    private var slotKey: [Int64] = []          // key occupying each slot, -1 free
    private var keyToSlot: [Int64: Int] = [:]
    private var freq: [Int64: Int] = [:]
    private var lastUse: [Int: Int] = [:]      // slot -> tick
    private var tick = 0

    public private(set) var hits = 0
    public private(set) var misses = 0
    private let maxSlots: Int

    init(containerDir: URL, spec: Checkpoint.QuantSpec,
         inter: Int, hidden: Int, budgetBytes: Int) throws {
        let reader = try QpackExpertReader(containerDir: containerDir)
        self.reader = reader
        guard spec.bits == 4 || spec.bits == 8 else {
            throw Checkpoint.Error.unsupportedBits(spec.bits)
        }
        guard reader.layout.expertStride > 0, !reader.layout.sections.isEmpty else {
            throw Checkpoint.Error.badShape("corrupt container: empty expert layout")
        }
        bits = spec.bits
        groupSize = spec.groupSize
        self.inter = inter
        self.hidden = hidden
        stride = reader.layout.expertStride

        func trio(_ proj: String) throws -> (Qpack.Section, Qpack.Section, Qpack.Section) {
            guard let w = reader.section(proj + ".weight"),
                  let s = reader.section(proj + ".scales"),
                  let b = reader.section(proj + ".biases")
            else { throw Checkpoint.Error.badShape("container expert sections missing for \(proj)") }
            return (w, s, b)
        }
        let g = try trio("gate_proj")
        let u = try trio("up_proj")
        let d = try trio("down_proj")
        gate = MatrixSections(weight: g.0.offset, scales: g.1.offset, biases: g.2.offset)
        up = MatrixSections(weight: u.0.offset, scales: u.1.offset, biases: u.2.offset)
        down = MatrixSections(weight: d.0.offset, scales: d.1.offset, biases: d.2.offset)
        scalesDtype = g.1.dtype

        let total = reader.layout.expertCount * reader.layout.layerCount
        maxSlots = min(max(budgetBytes / stride, 16), total)
    }

    deinit {
        for slot in slots {
            if let slot { free(slot) }
        }
    }

    public var slotCount: Int { maxSlots }
    public var allocatedSlots: Int { slots.count }
    public var residentBytes: Int { slots.count * stride }

    private func key(_ layer: Int, _ expert: Int) -> Int64 {
        Int64(layer) << 32 | Int64(expert)
    }

    /// Blob pointers for a batch of experts in one layer. The whole batch is
    /// resident simultaneously (a member is never evicted to make room for
    /// another member); pointers stay valid until the next `blobs` call.
    func blobs(layer: Int, experts: [Int]) throws -> [UnsafeRawPointer] {
        tick += 1
        var protectedSlots = Set<Int>()
        var result: [UnsafeRawPointer] = []
        result.reserveCapacity(experts.count)
        for e in experts {
            let k = key(layer, e)
            freq[k, default: 0] += 1
            if let s = keyToSlot[k] {
                hits += 1
                lastUse[s] = tick
                protectedSlots.insert(s)
                result.append(UnsafeRawPointer(slots[s]!))
                continue
            }
            misses += 1
            let s = try slotForFill(excluding: protectedSlots)
            if slotKey[s] >= 0 { keyToSlot.removeValue(forKey: slotKey[s]) }
            try reader.readExpert(layer: layer, expert: e, into: slots[s]!)
            slotKey[s] = k
            keyToSlot[k] = s
            lastUse[s] = tick
            protectedSlots.insert(s)
            result.append(UnsafeRawPointer(slots[s]!))
        }
        return result
    }

    /// LFU victim with recency tie-break; grow-on-demand, then free slots,
    /// then eviction — the same policy as ExpertCache.
    private func slotForFill(excluding: Set<Int>) throws -> Int {
        if slots.count < maxSlots {
            if let b = malloc(stride) {
                slots.append(b)
                slotKey.append(-1)
                return slots.count - 1
            }
        }
        if let free = slotKey.firstIndex(of: -1) { return free }
        var victim = -1
        var victimScore = (Int.max, Int.max)
        for s in 0..<slots.count where !excluding.contains(s) {
            let f = freq[slotKey[s]] ?? 0
            let score = (f, lastUse[s] ?? 0)
            if score < victimScore {
                victimScore = score
                victim = s
            }
        }
        guard victim >= 0 else {
            throw Checkpoint.Error.badShape("expert cache too small for batch")
        }
        return victim
    }
}

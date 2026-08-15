import Foundation
import Testing
@testable import SwiftletCore

/// The C packed GEMV against a Swift dequant-then-dot reference: both bit
/// widths, all three scale dtypes, and byte offsets that are not 4-aligned
/// (blob sections sit at arbitrary offsets inside a container blob).
@Suite struct CPUKernelTests {

    struct SeededRandom {
        var state: UInt64
        mutating func next() -> UInt64 {
            state ^= state >> 12
            state ^= state << 25
            state ^= state >> 27
            return state &* 2685821657736338717
        }
        mutating func float(_ range: ClosedRange<Float>) -> Float {
            Float(Int(next() % 10_000)) / 10_000 * (range.upperBound - range.lowerBound) + range.lowerBound
        }
        mutating func int(_ range: ClosedRange<Int>) -> Int {
            Int(next() % UInt64(range.upperBound - range.lowerBound + 1)) + range.lowerBound
        }
    }

    /// Packs flat q values into uint32 words, MLX layout: element j of each
    /// word holds q at bit offset j*bits.
    static func pack(_ q: [UInt32], bits: Int) -> [UInt32] {
        let perWord = 32 / bits
        let mask = UInt32((1 << bits) - 1)
        var words = [UInt32]()
        words.reserveCapacity(q.count / perWord + 1)
        var acc: UInt32 = 0
        var j = 0
        for v in q {
            acc |= (v & mask) << UInt32(j * bits)
            j += 1
            if j == perWord { words.append(acc); acc = 0; j = 0 }
        }
        if j > 0 { words.append(acc) }
        return words
    }

    @Test func qgemvMatchesReference() throws {
        // rows >= 64 so the multithreaded dispatch path is exercised too.
        let rows = 100, inDim = 320, groupSize = 64
        let groups = inDim / groupSize

        for bits in [4, 8] {
            for dtype in ["F32", "F16", "BF16"] {
                var rng = SeededRandom(state: 0xFEED5EED00000000 + UInt64(bits) * 100 + UInt64(dtype.utf8.first!))
                let mask = UInt32((1 << bits) - 1)

                // Scales/biases per dtype; the f32-domain values drive the
                // reference regardless of storage width.
                var scaleF = [Float](repeating: 0, count: rows * groups)
                var biasF = [Float](repeating: 0, count: rows * groups)
                var scalesBytes = [UInt8]()
                var biasesBytes = [UInt8]()
                scalesBytes.reserveCapacity(rows * groups * 4)
                for i in 0..<(rows * groups) {
                    var s: Float, b: Float
                    switch dtype {
                    case "F16":
                        // Random finite f16 patterns across the exponent range
                        // (heavily below 1.0, where the rebias arithmetic wraps
                        // if done unsigned), plus occasional subnormals;
                        // reference value via the Swift decoder.
                        var hs: UInt16
                        if rng.next() % 8 == 0 {
                            hs = UInt16(rng.int(1...0x03FF)) | (rng.next() % 2 == 0 ? 0 : 0x8000)
                        } else {
                            hs = UInt16(rng.int(0x1000...0x4400)) | (rng.next() % 2 == 0 ? 0 : 0x8000)
                        }
                        let hb = UInt16(rng.int(0x0800...0x3C00)) | (rng.next() % 2 == 0 ? 0 : 0x8000)
                        s = SafetensorsFile.f16ToF32(hs)
                        b = SafetensorsFile.f16ToF32(hb)
                        scalesBytes.append(UInt8(hs & 0xFF)); scalesBytes.append(UInt8(hs >> 8))
                        biasesBytes.append(UInt8(hb & 0xFF)); biasesBytes.append(UInt8(hb >> 8))
                    case "BF16":
                        // High half of an f32 pattern; both sides derive from
                        // the same bits independently.
                        let hs = UInt16(truncatingIfNeeded: rng.float(0.1...1.5).bitPattern >> 16)
                        let hb = UInt16(truncatingIfNeeded: rng.float(-0.5...0.5).bitPattern >> 16)
                        s = Float(bitPattern: UInt32(hs) << 16)
                        b = Float(bitPattern: UInt32(hb) << 16)
                        scalesBytes.append(UInt8(hs & 0xFF)); scalesBytes.append(UInt8(hs >> 8))
                        biasesBytes.append(UInt8(hb & 0xFF)); biasesBytes.append(UInt8(hb >> 8))
                    default:
                        s = rng.float(0.1...1.5)
                        b = rng.float(-0.5...0.5)
                        var sp = s.bitPattern, bp = b.bitPattern
                        withUnsafeBytes(of: &sp) { scalesBytes.append(contentsOf: $0) }
                        withUnsafeBytes(of: &bp) { biasesBytes.append(contentsOf: $0) }
                    }
                    scaleF[i] = s
                    biasF[i] = b
                }

                let q = (0..<(rows * inDim)).map { _ in UInt32(rng.int(0...Int(mask))) }
                var weightBytes = [UInt8]()
                weightBytes.reserveCapacity(q.count / (32 / bits) * 4)
                for w in Self.pack(q, bits: bits) {
                    var le = w.littleEndian
                    withUnsafeBytes(of: &le) { weightBytes.append(contentsOf: $0) }
                }
                let x = (0..<inDim).map { _ in rng.float(-1...1) }

                // Synthetic blob with deliberately non-4-aligned offsets.
                var blob: [UInt8] = [0xAA, 0xBB]
                let wOff = blob.count
                blob += weightBytes
                blob.append(0xCC)
                let sOff = blob.count
                blob += scalesBytes
                blob.append(0xDD)
                let bOff = blob.count
                blob += biasesBytes

                var y = [Float](repeating: .nan, count: rows)
                try blob.withUnsafeBytes { raw in
                    try x.withUnsafeBufferPointer { xp in
                        try y.withUnsafeMutableBufferPointer { yp in
                            QuantizedGEMV.gemv(
                                blob: raw.baseAddress!,
                                weightOffset: wOff, scalesOffset: sOff, biasesOffset: bOff,
                                scalesDtype: dtype,
                                x: xp.baseAddress!, out: yp.baseAddress!,
                                outDim: rows, inDim: inDim, groupSize: groupSize, bits: bits)
                        }
                    }
                }

                for r in 0..<rows {
                    var acc: Float = 0
                    for g in 0..<groups {
                        var qdot: Float = 0
                        var xsum: Float = 0
                        for i in 0..<groupSize {
                            let col = g * groupSize + i
                            qdot += Float(q[r * inDim + col]) * x[col]
                            xsum += x[col]
                        }
                        acc += scaleF[r * groups + g] * qdot + biasF[r * groups + g] * xsum
                    }
                    let diff = abs(y[r] - acc)
                    #expect(diff <= 2e-3 * (1 + abs(acc)),
                            "bits \(bits) \(dtype) row \(r): kernel \(y[r]) vs reference \(acc)")
                }
            }
        }
    }

    /// Known IEEE 754 half-precision values, including the sub-1.0 and
    /// subnormal cases that trip unsigned exponent arithmetic.
    @Test func f16KnownValues() {
        #expect(SafetensorsFile.f16ToF32(0x0000) == 0)
        #expect(SafetensorsFile.f16ToF32(0x8000) == 0 && SafetensorsFile.f16ToF32(0x8000).sign == .minus)
        #expect(SafetensorsFile.f16ToF32(0x0001) == Float(bitPattern: 0x3380_0000))   // 2^-24, smallest subnormal
        #expect(SafetensorsFile.f16ToF32(0x03FF) == Float(bitPattern: 0x387F_C000))   // largest subnormal
        #expect(SafetensorsFile.f16ToF32(0x0400) == Float(bitPattern: 0x3880_0000))   // min normal, 2^-14
        #expect(SafetensorsFile.f16ToF32(0x3000) == 0.125)
        #expect(SafetensorsFile.f16ToF32(0x3400) == 0.25)
        #expect(SafetensorsFile.f16ToF32(0x3800) == 0.5)
        #expect(SafetensorsFile.f16ToF32(0x3C00) == 1.0)
        #expect(SafetensorsFile.f16ToF32(0x4000) == 2.0)
        #expect(SafetensorsFile.f16ToF32(0xB800) == -0.5)
        #expect(SafetensorsFile.f16ToF32(0x7C00) == Float.infinity)
        #expect(SafetensorsFile.f16ToF32(0x7E00).isNaN)
    }

    /// Throughput on a realistic expert matrix (35B 4-bit: 512x2048, g64):
    /// reports effective GB/s over the quantized weight bytes. Informational —
    /// the bound is generous enough for the scalar fallback on slow CI Macs.
    @Test func qgemvThroughput() throws {
        let rows = 512, inDim = 2048, groupSize = 64, bits = 4
        let groups = inDim / groupSize
        var rng = SeededRandom(state: 0xB1057ED)
        let mask = UInt32((1 << bits) - 1)

        var blob = [UInt8]()
        let wOff = 0
        let q = (0..<(rows * inDim)).map { _ in UInt32(rng.int(0...Int(mask))) }
        for w in Self.pack(q, bits: bits) {
            var le = w.littleEndian
            withUnsafeBytes(of: &le) { blob.append(contentsOf: $0) }
        }
        let sOff = blob.count
        for _ in 0..<(rows * groups) {
            var v = Float(0.05).bitPattern
            withUnsafeBytes(of: &v) { blob.append(contentsOf: $0) }
        }
        let bOff = blob.count
        for _ in 0..<(rows * groups) {
            var v = Float(0.01).bitPattern
            withUnsafeBytes(of: &v) { blob.append(contentsOf: $0) }
        }
        let x = [Float](repeating: 0.1, count: inDim)
        var y = [Float](repeating: 0, count: rows)

        let weightBytes = rows * inDim * bits / 8
        try blob.withUnsafeBytes { raw in
            try x.withUnsafeBufferPointer { xp in
                try y.withUnsafeMutableBufferPointer { yp in
                    for _ in 0..<3 {   // warmup
                        QuantizedGEMV.gemv(blob: raw.baseAddress!, weightOffset: wOff,
                                           scalesOffset: sOff, biasesOffset: bOff, scalesDtype: "F32",
                                           x: xp.baseAddress!, out: yp.baseAddress!,
                                           outDim: rows, inDim: inDim, groupSize: groupSize, bits: bits)
                    }
                    let t0 = Date()
                    let iters = 200
                    for _ in 0..<iters {
                        QuantizedGEMV.gemv(blob: raw.baseAddress!, weightOffset: wOff,
                                           scalesOffset: sOff, biasesOffset: bOff, scalesDtype: "F32",
                                           x: xp.baseAddress!, out: yp.baseAddress!,
                                           outDim: rows, inDim: inDim, groupSize: groupSize, bits: bits)
                    }
                    let secs = -t0.timeIntervalSinceNow
                    let gbs = Double(weightBytes * iters) / secs / 1e9
                    print(String(format: "[CPUKernelTests] 4-bit GEMV 512x2048: %.1f GB/s (%.2f ms/matrix)",
                                 gbs, secs / Double(iters) * 1000))
                    // The floor only makes sense optimized; debug builds run
                    // the kernel at -O0 and are exempt.
                    #expect(gbs > 0.5 || _isDebugAssertConfiguration(),
                            "GEMV implausibly slow: \(gbs) GB/s")
                }
            }
        }
    }
}

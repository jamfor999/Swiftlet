import CCPUKernels
import Foundation

/// Swift entry to the C quantized GEMV: computes y = x @ W^T for an
/// MLX-affine-quantized matrix held in a container blob, unpacking nibbles
/// on the fly (AVX2 when available). Rows are split across the active cores;
/// below a small row count the call runs inline to avoid thread overhead.
enum QuantizedGEMV {
    /// Kernel scale dtype code for a safetensors dtype string.
    static func scalesType(_ dtype: String) -> Int32 {
        switch dtype {
        case "F16": return 1
        case "BF16": return 2
        default: return 0
        }
    }

    static func gemv(
        blob: UnsafeRawPointer,
        weightOffset: Int, scalesOffset: Int, biasesOffset: Int, scalesDtype: String,
        x: UnsafePointer<Float>,
        out: UnsafeMutablePointer<Float>,
        outDim: Int, inDim: Int, groupSize: Int, bits: Int
    ) {
        let w = (blob + weightOffset).assumingMemoryBound(to: UInt8.self)
        let s = (blob + scalesOffset).assumingMemoryBound(to: UInt8.self)
        let b = (blob + biasesOffset).assumingMemoryBound(to: UInt8.self)
        let stype = scalesType(scalesDtype)
        let cores = max(ProcessInfo.processInfo.activeProcessorCount, 1)
        if outDim < 64 || cores == 1 {
            swiftlet_qgemv(w, s, b, x, out, Int32(outDim), Int32(inDim),
                            Int32(groupSize), Int32(bits), stype, 0, Int32(outDim))
            return
        }
        DispatchQueue.concurrentPerform(iterations: cores) { i in
            let r0 = i * outDim / cores
            let r1 = (i + 1) * outDim / cores
            guard r1 > r0 else { return }
            swiftlet_qgemv(w, s, b, x, out, Int32(outDim), Int32(inDim),
                            Int32(groupSize), Int32(bits), stype, Int32(r0), Int32(r1))
        }
    }

    /// Array-slice entry matching `QwenCPUModel.matvec`'s shape: one matrix
    /// of an expert blob against a slice of `x`, into `out` from offset 0.
    static func gemv(
        blob: UnsafeRawPointer, sections: CPUExpertCache.MatrixSections, scalesDtype: String,
        x: [Float], xOffset: Int,
        into out: inout [Float],
        outDim: Int, inDim: Int, groupSize: Int, bits: Int
    ) {
        x.withUnsafeBufferPointer { xb in
            out.withUnsafeMutableBufferPointer { ob in
                gemv(blob: blob,
                     weightOffset: sections.weight, scalesOffset: sections.scales,
                     biasesOffset: sections.biases, scalesDtype: scalesDtype,
                     x: xb.baseAddress! + xOffset, out: ob.baseAddress!,
                     outDim: outDim, inDim: inDim, groupSize: groupSize, bits: bits)
            }
        }
    }
}

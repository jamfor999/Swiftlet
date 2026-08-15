import Foundation

/// Minimal safetensors reader: 8-byte little-endian header length, JSON header
/// mapping tensor name -> { dtype, shape, data_offsets }, then raw data.
public struct SafetensorsFile {
    public struct TensorInfo: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let byteRange: Range<Int>
    }

    private let data: Data
    private let dataStart: Int
    public private(set) var tensors: [String: TensorInfo] = [:]

    public enum Error: Swift.Error {
        case malformedHeader
        case missingTensor(String)
        case unsupportedDtype(String, tensor: String)
    }

    public init(url: URL) throws {
        // External drives occasionally interrupt the initial map (EINTR);
        // retry briefly before giving up.
        var mapped: Data? = nil
        var lastError: Swift.Error = Error.malformedHeader
        for attempt in 0..<4 {
            do {
                mapped = try Data(contentsOf: url, options: .mappedIfSafe)
                break
            } catch {
                lastError = error
                usleep(useconds_t(100_000 * (attempt + 1)))
            }
        }
        guard let mapped else { throw lastError }
        data = mapped
        guard data.count >= 8 else { throw Error.malformedHeader }
        let headerLen = data.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: 0, as: UInt64.self)) }
        dataStart = 8 + headerLen
        guard data.count >= dataStart else { throw Error.malformedHeader }
        let headerJSON = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<dataStart))
        guard let header = headerJSON as? [String: Any] else { throw Error.malformedHeader }
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { throw Error.malformedHeader }
            tensors[name] = TensorInfo(dtype: dtype, shape: shape, byteRange: offsets[0]..<offsets[1])
        }
    }

    public func info(_ name: String) throws -> TensorInfo {
        guard let t = tensors[name] else { throw Error.missingTensor(name) }
        return t
    }

    func rawBytes(_ t: TensorInfo) -> Data {
        data.subdata(in: (dataStart + t.byteRange.lowerBound)..<(dataStart + t.byteRange.upperBound))
    }

    /// Raw stored bytes of a tensor (no dtype conversion), with its info.
    public func raw(_ name: String) throws -> (info: TensorInfo, bytes: Data) {
        let t = try info(name)
        return (t, rawBytes(t))
    }

    /// Byte offset of a tensor's data from the start of the FILE (for mmap use).
    public func absoluteOffset(_ name: String) throws -> Int {
        dataStart + (try info(name)).byteRange.lowerBound
    }

    /// Zero-copy access to a tensor's raw bytes (view into the mapped file).
    public func withRawBytes<T>(_ name: String, _ body: (SafetensorsFile.TensorInfo, UnsafeRawBufferPointer) throws -> T) throws -> T {
        let t = try info(name)
        return try data.withUnsafeBytes { raw in
            let slice = UnsafeRawBufferPointer(
                rebasing: raw[(dataStart + t.byteRange.lowerBound)..<(dataStart + t.byteRange.upperBound)]
            )
            return try body(t, slice)
        }
    }

    public static func bytesPerElement(_ dtype: String) -> Int? {
        switch dtype {
        case "F32", "I32", "U32": return 4
        case "F16", "BF16", "I16", "U16": return 2
        case "F64", "I64", "U64": return 8
        case "I8", "U8": return 1
        default: return nil
        }
    }

    /// Serialize tensors into safetensors format (used by the repacker for the
    /// dense/resident weight file).
    public static func write(to url: URL, tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)]) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            header[t.name] = [
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, offset + t.bytes.count],
            ]
            offset += t.bytes.count
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        var len = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(headerData)
        for t in tensors { out.append(t.bytes) }
        try out.write(to: url)
    }

    /// IEEE 754 half -> float bit conversion. `Swift.Float16` is unavailable
    /// on Intel macOS, so decode by hand (load-time only, not a hot path).
    static func f16ToF32(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32(h & 0x7C00) >> 10
        let frac = UInt32(h & 0x03FF)
        if exp == 0x1F {   // Inf / NaN
            return Float(bitPattern: sign | 0x7F80_0000 | frac << 13 | (frac != 0 ? 0x0040_0000 : 0))
        }
        if exp == 0 {
            if frac == 0 { return Float(bitPattern: sign) }   // ±0
            // Subnormal: normalize so bit 10 is set, then rebias the exponent.
            var e: UInt32 = 0
            var n = frac
            while n & 0x0400 == 0 { n <<= 1; e += 1 }
            return Float(bitPattern: sign | (113 - e) << 23 | (n & 0x03FF) << 13)
        }
        return Float(bitPattern: sign | (exp - 15 + 127) << 23 | frac << 13)
    }

    /// Tensor contents converted to Float32. Supports F32, F16, BF16.
    public func floats(_ name: String) throws -> [Float] {
        let t = try info(name)
        let bytes = rawBytes(t)
        switch t.dtype {
        case "F32":
            return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        case "F16":
            return bytes.withUnsafeBytes { $0.bindMemory(to: UInt16.self).map(Self.f16ToF32) }
        case "BF16":
            return bytes.withUnsafeBytes {
                $0.bindMemory(to: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            }
        default:
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
    }

    /// Integer tensor contents. Supports I32, U32, I64.
    public func ints(_ name: String) throws -> [Int] {
        let t = try info(name)
        let bytes = rawBytes(t)
        switch t.dtype {
        case "I32": return bytes.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Int.init) }
        case "U32": return bytes.withUnsafeBytes { $0.bindMemory(to: UInt32.self).map(Int.init) }
        case "I64": return bytes.withUnsafeBytes { $0.bindMemory(to: Int64.self).map(Int.init) }
        default:
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
    }
}

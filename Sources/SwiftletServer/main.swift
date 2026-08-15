import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import SwiftletCore

// Loopback OpenAI-compatible Chat Completions server (modeled on
// TurboFieldfare's). One warm model, requests serialized. No auth/TLS: keep it
// on 127.0.0.1.
//
//   swiftlet-server --model <dir> [--port 8080] [--cache-gb 2]

/// Message content per the OpenAI Chat Completions spec: a plain string, an
/// array of parts ([{type:"text",text:"..."}]), or null on tool-call turns.
/// Text parts are joined; non-text parts are dropped; null decodes as "".
struct ChatContent: Decodable {
    struct Part: Decodable { let text: String? }
    let text: String
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if try c.decodeNil() {
            text = ""
        } else if let s = try? c.decode(String.self) {
            text = s
        } else {
            text = try c.decode([Part].self).compactMap(\.text).joined()
        }
    }
}

/// OpenAI-compatible Chat Completions request body.
struct ChatRequest: Decodable {
    struct Message: Decodable { let role: String; let content: ChatContent }
    let messages: [Message]
    let stream: Bool?
    let max_tokens: Int?
    let max_completion_tokens: Int?
    let temperature: Float?
    let top_p: Float?
}

let cliArgs = CommandLine.arguments
func flag(_ name: String) -> String? {
    guard let i = cliArgs.firstIndex(of: name), i + 1 < cliArgs.count else { return nil }
    return cliArgs[i + 1]
}
guard let modelPath = flag("--model") else {
    print("usage: swiftlet-server --model <dir> [--port 8080] [--cache-gb 2]")
    exit(2)
}
let port = Int(flag("--port") ?? "8080") ?? 8080
let cacheGB = Double(flag("--cache-gb") ?? "2") ?? 2
let modelURL = URL(fileURLWithPath: modelPath)

FileHandle.standardError.write(Data("loading model + tokenizer...\n".utf8))
// The same Metal streaming engine as `swiftlet chat`: routed experts stream
// from the .qpack blobs, so a 35B/80B container serves in a few GB of RAM.
// (The CPU model reader cannot serve containers: their experts are not in
// model.safetensors.)
let session = try await SwiftletSession(modelDir: modelURL, cacheBudgetGB: cacheGB)
let modelName: String = {
    let configURL = modelURL.appendingPathComponent("config.json")
    if let cfg = try? JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any],
       let t = cfg["model_type"] as? String { return t }
    return "swiftlet"
}()
// One request at a time: the session mutates shared conversation state.
let generationQueue = DispatchQueue(label: "swiftlet.generation")

@Sendable func jsonData(_ obj: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
}

@Sendable func completionPayload(
    id: String, text: String?, delta: String?, finish: String?,
    usage: (prompt: Int, completion: Int)? = nil
) -> [String: Any] {
    var choice: [String: Any] = ["index": 0]
    if let text { choice["message"] = ["role": "assistant", "content": text] }
    if let delta { choice["delta"] = ["content": delta] }
    if let finish { choice["finish_reason"] = finish }
    var payload: [String: Any] = [
        "id": id,
        "object": delta != nil ? "chat.completion.chunk" : "chat.completion",
        "created": Int(Date().timeIntervalSince1970),
        "model": modelName,
        "choices": [choice],
    ]
    if let usage {
        payload["usage"] = [
            "prompt_tokens": usage.prompt,
            "completion_tokens": usage.completion,
            "total_tokens": usage.prompt + usage.completion,
        ]
    }
    return payload
}

final class HTTPHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private var requestHead: HTTPRequestHead?
    private var body = ByteBuffer()

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            requestHead = head
            body = ByteBuffer()
        case .body(var chunk):
            body.writeBuffer(&chunk)
        case .end:
            guard let head = requestHead else { return }
            route(context: context, head: head, body: body)
            requestHead = nil
        }
    }

    private func route(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer) {
        switch (head.method, head.uri) {
        case (.GET, "/v1/models"):
            respondJSON(context, status: .ok, data: jsonData([
                "object": "list",
                "data": [["id": modelName, "object": "model", "owned_by": "swiftlet"]],
            ]))
        case (.POST, "/v1/chat/completions"):
            handleChat(context: context, body: body)
        default:
            respondJSON(context, status: .notFound, data: jsonData(["error": "not found"]))
        }
    }

    private func handleChat(context: ChannelHandlerContext, body: ByteBuffer) {
        let bytes: [UInt8] = body.getBytes(at: 0, length: body.readableBytes) ?? []
        let data = Data(bytes)
        guard let request = try? JSONDecoder().decode(ChatRequest.self, from: data) else {
            respondJSON(context, status: .badRequest, data: jsonData(["error": "malformed request"]))
            return
        }
        let messages = request.messages.map { ["role": $0.role, "content": $0.content.text] }
        let maxNew = request.max_tokens ?? request.max_completion_tokens ?? 512
        let streaming = request.stream ?? false
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let eventLoop = context.eventLoop
        let channel = context.channel

        let options: SwiftletSession.GenerationOptions = {
            var o = SwiftletSession.GenerationOptions()
            if let t = request.temperature {
                if t <= 0 { o = .greedy } else { o.temperature = t }
            }
            if let p = request.top_p { o.topP = p }
            return o
        }()

        // Agent clients (OpenCode etc.) send multi-thousand-token prompts, so
        // prefill can run minutes with no output. SSE comment heartbeats keep
        // the idle connection from being dropped; parsers ignore them.
        var heartbeat: RepeatedTask? = nil
        if streaming {
            var head = HTTPResponseHead(version: .http1_1, status: .ok)
            head.headers.add(name: "Content-Type", value: "text/event-stream")
            head.headers.add(name: "Cache-Control", value: "no-cache")
            head.headers.add(name: "Transfer-Encoding", value: "chunked")
            context.writeAndFlush(wrapOutboundOut(.head(head)), promise: nil)
            heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: .seconds(15), delay: .seconds(15)
            ) { _ in
                var buf = channel.allocator.buffer(capacity: 16)
                buf.writeString(": ping\n\n")
                _ = channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buf)))
            }
        }
        let stopHeartbeat = heartbeat

        @Sendable func writeSSE(_ obj: [String: Any]) {
            let payload = "data: " + (String(data: jsonData(obj), encoding: .utf8) ?? "{}") + "\n\n"
            eventLoop.execute {
                var buf = channel.allocator.buffer(capacity: payload.utf8.count)
                buf.writeString(payload)
                _ = channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buf)))
            }
        }

        generationQueue.async {
            let done = DispatchSemaphore(value: 0)
            Task {
                defer { done.signal() }
                do {
                    var fullText = ""
                    for try await delta in session.streamChat(
                        messages: messages, maxNew: maxNew, options: options
                    ) {
                        fullText += delta
                        if streaming {
                            writeSSE(completionPayload(id: id, text: nil, delta: delta, finish: nil))
                        }
                    }
                    let m = session.lastMetrics
                    FileHandle.standardError.write(Data(String(
                        format: "[%@] %d prompt + %d generated, ttft %.1fs, %.2f tok/s\n",
                        id as NSString, m.promptTokens, m.generatedTokens,
                        m.timeToFirstToken, m.tokensPerSecond
                    ).utf8))

                    eventLoop.execute {
                        stopHeartbeat?.cancel()
                        if streaming {
                            // The finish chunk, [DONE], and the HTTP end must
                            // land in this order on the wire. Write them here
                            // directly: routing the finish chunk through
                            // writeSSE would re-enqueue it on the event loop
                            // and it would land after .end and be dropped
                            // (strict clients then wait for it forever).
                            let finish = jsonData(completionPayload(
                                id: id, text: nil, delta: "", finish: "stop",
                                usage: (m.promptTokens, m.generatedTokens)))
                            let tail = "data: " + (String(data: finish, encoding: .utf8) ?? "{}")
                                + "\n\ndata: [DONE]\n\n"
                            var buf = channel.allocator.buffer(capacity: tail.utf8.count)
                            buf.writeString(tail)
                            _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(buf)))
                            _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                        } else {
                            let data = jsonData(completionPayload(
                                id: id, text: fullText, delta: nil, finish: "stop",
                                usage: (m.promptTokens, m.generatedTokens)))
                            var head = HTTPResponseHead(version: .http1_1, status: .ok)
                            head.headers.add(name: "Content-Type", value: "application/json")
                            head.headers.add(name: "Content-Length", value: String(data.count))
                            var buf = channel.allocator.buffer(capacity: data.count)
                            buf.writeBytes(data)
                            _ = channel.write(HTTPServerResponsePart.head(head))
                            _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(buf)))
                            _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                        }
                    }
                } catch {
                    eventLoop.execute {
                        stopHeartbeat?.cancel()
                        if streaming {
                            // The 200 + SSE head is already on the wire; a
                            // second head would be a protocol error. Report
                            // the failure as an SSE event and end the stream.
                            let payload = "data: " + (String(
                                data: jsonData(["error": "\(error)"]), encoding: .utf8
                            ) ?? "{}") + "\n\ndata: [DONE]\n\n"
                            var buf = channel.allocator.buffer(capacity: payload.utf8.count)
                            buf.writeString(payload)
                            _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(buf)))
                            _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                        } else {
                            let data = jsonData(["error": "\(error)"])
                            var head = HTTPResponseHead(version: .http1_1, status: .internalServerError)
                            head.headers.add(name: "Content-Length", value: String(data.count))
                            var buf = channel.allocator.buffer(capacity: data.count)
                            buf.writeBytes(data)
                            _ = channel.write(HTTPServerResponsePart.head(head))
                            _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(buf)))
                            _ = channel.writeAndFlush(HTTPServerResponsePart.end(nil))
                        }
                    }
                }
            }
            done.wait()
        }
    }

    private func respondJSON(_ context: ChannelHandlerContext, status: HTTPResponseStatus, data: Data) {
        var head = HTTPResponseHead(version: .http1_1, status: status)
        head.headers.add(name: "Content-Type", value: "application/json")
        head.headers.add(name: "Content-Length", value: String(data.count))
        var buf = context.channel.allocator.buffer(capacity: data.count)
        buf.writeBytes(data)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }
}

let group = MultiThreadedEventLoopGroup(numberOfThreads: 2)
let bootstrap = ServerBootstrap(group: group)
    .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
    .childChannelInitializer { channel in
        channel.pipeline.configureHTTPServerPipeline().flatMap {
            channel.pipeline.addHandler(HTTPHandler())
        }
    }

let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
print("swiftlet-server listening on http://127.0.0.1:\(port)/v1 (model: \(modelName))")
try await channel.closeFuture.get()

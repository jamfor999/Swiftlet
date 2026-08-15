import Foundation
import Testing
@testable import SwiftletServer

@Suite struct ChatRequestTests {
    private func decode(_ json: String) throws -> ChatRequest {
        try JSONDecoder().decode(ChatRequest.self, from: Data(json.utf8))
    }

    @Test func plainStringContent() throws {
        let r = try decode(
            #"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}],"stream":true,"max_tokens":512,"temperature":0.7}"#
        )
        #expect(r.messages.map(\.role) == ["system", "user"])
        #expect(r.messages.map(\.content.text) == ["sys", "hi"])
        #expect(r.stream == true)
        #expect(r.max_tokens == 512)
    }

    @Test func arrayOfPartsContent() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#
        )
        #expect(r.messages[0].content.text == "hi")
    }

    @Test func multipleTextPartsJoined() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#
        )
        #expect(r.messages[0].content.text == "ab")
    }

    @Test func imageOnlyPartsDropped() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,xxx"}}]}]}"#
        )
        #expect(r.messages[0].content.text == "")
    }

    @Test func nullContentWithToolCalls() throws {
        let r = try decode(
            #"{"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]}]}"#
        )
        #expect(r.messages[0].role == "assistant")
        #expect(r.messages[0].content.text == "")
    }

    @Test func unknownTopLevelKeysIgnored() throws {
        let r = try decode(
            #"{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"bash"}}],"stream_options":{},"store":false,"max_completion_tokens":256}"#
        )
        #expect(r.messages.count == 1)
        #expect(r.max_tokens == nil)
        #expect(r.max_completion_tokens == 256)
    }

    @Test func malformedInputThrows() {
        #expect(throws: DecodingError.self) {
            _ = try decode(#"{"messages":"nope"}"#)
        }
    }

    @Test func nonSpecContentTypeThrows() {
        #expect(throws: DecodingError.self) {
            _ = try decode(#"{"messages":[{"role":"user","content":42}]}"#)
        }
    }
}

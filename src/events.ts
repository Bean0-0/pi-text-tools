/**
 * Converts the event stream of a text-only model into pi's native tool-call events.
 *
 * Plain text streams through as it arrives. Text inside `<tool_call>` blocks is held until the
 * block closes, then emitted as `toolcall_start` / `toolcall_delta` / `toolcall_end`.
 */

import type {
	AssistantMessage,
	AssistantMessageEvent,
	TextContent,
	ThinkingContent,
	ToolCall,
} from "@earendil-works/pi-ai/compat";
import { type ParserEvent, parseToolCallBody, ToolCallParser } from "./parser.ts";

export interface ConvertOptions {
	/** Overrides applied to every emitted message, such as the outer provider and api. */
	identity: Pick<AssistantMessage, "api" | "provider" | "model">;
	/** Names of the tools offered to the model. Used to accept calls whose tags were stripped. */
	toolNames?: readonly string[];
	/** Creates tool-call ids. Injectable for tests. */
	newId?: () => string;
}

let counter = 0;
const defaultId = () => `call_tt_${Date.now().toString(36)}_${(counter++).toString(36)}`;

export async function* convertEvents(
	inner: AsyncIterable<AssistantMessageEvent>,
	options: ConvertOptions,
): AsyncGenerator<AssistantMessageEvent> {
	const newId = options.newId ?? defaultId;
	const toolNames = new Set(options.toolNames ?? []);
	const parser = new ToolCallParser({ isKnownTool: (name) => toolNames.has(name) });
	let out: AssistantMessage | undefined;
	let openText: { index: number; block: TextContent } | undefined;
	const thinkingMap = new Map<number, { index: number; block: ThinkingContent }>();
	const passthroughToolMap = new Map<number, number>();

	const message = (): AssistantMessage => {
		if (!out) throw new Error("text-tools: content event before start");
		return out;
	};

	function* closeText(): Generator<AssistantMessageEvent> {
		if (!openText) return;
		const { index, block } = openText;
		openText = undefined;
		yield { type: "text_end", contentIndex: index, content: block.text, partial: message() };
	}

	function* emitText(text: string): Generator<AssistantMessageEvent> {
		// Whitespace before a text block (for example between tool calls) should not create
		// empty blocks, so a new block only opens on real text, with leading whitespace dropped.
		const msg = message();
		if (!openText) {
			text = text.replace(/^\s+/, "");
			if (!text) return;
			const block: TextContent = { type: "text", text: "" };
			msg.content.push(block);
			openText = { index: msg.content.length - 1, block };
			yield { type: "text_start", contentIndex: openText.index, partial: msg };
		}
		openText.block.text += text;
		yield { type: "text_delta", contentIndex: openText.index, delta: text, partial: msg };
	}

	function* emitToolCall(body: string): Generator<AssistantMessageEvent> {
		yield* closeText();
		const msg = message();
		const parsed = parseToolCallBody(body);
		const block: ToolCall = { type: "toolCall", id: newId(), name: parsed.name, arguments: {} };
		msg.content.push(block);
		const index = msg.content.length - 1;
		yield { type: "toolcall_start", contentIndex: index, partial: msg };
		const json = JSON.stringify(parsed.arguments);
		block.arguments = parsed.arguments as ToolCall["arguments"];
		if (parsed.error) block.argumentsError = parsed.error;
		yield { type: "toolcall_delta", contentIndex: index, delta: json, partial: msg };
		yield { type: "toolcall_end", contentIndex: index, toolCall: block, partial: msg };
	}

	function* handle(events: ParserEvent[]): Generator<AssistantMessageEvent> {
		for (const event of events) {
			if (event.kind === "text") yield* emitText(event.text);
			else yield* emitToolCall(event.body);
		}
	}

	function finalize(source: AssistantMessage): AssistantMessage {
		const msg = message();
		const { content: _ignored, ...rest } = source;
		Object.assign(msg, rest, options.identity, { content: msg.content });
		return msg;
	}

	for await (const event of inner) {
		switch (event.type) {
			case "start": {
				out = { ...event.partial, ...options.identity, content: [] };
				yield { type: "start", partial: out };
				break;
			}
			case "text_start":
				break;
			case "text_delta":
				yield* handle(parser.push(event.delta));
				break;
			case "text_end":
				// Keep a tool call that spans text blocks open; otherwise release held-back text.
				yield* handle(parser.flushText());
				break;
			case "thinking_start": {
				yield* closeText();
				const msg = message();
				const block: ThinkingContent = { type: "thinking", thinking: "" };
				msg.content.push(block);
				const mapped = { index: msg.content.length - 1, block };
				thinkingMap.set(event.contentIndex, mapped);
				yield { type: "thinking_start", contentIndex: mapped.index, partial: msg };
				break;
			}
			case "thinking_delta": {
				const mapped = thinkingMap.get(event.contentIndex);
				if (!mapped) break;
				mapped.block.thinking += event.delta;
				yield { type: "thinking_delta", contentIndex: mapped.index, delta: event.delta, partial: message() };
				break;
			}
			case "thinking_end": {
				const mapped = thinkingMap.get(event.contentIndex);
				if (!mapped) break;
				const source = event.partial.content[event.contentIndex];
				if (source?.type === "thinking") Object.assign(mapped.block, source);
				else mapped.block.thinking = event.content;
				yield { type: "thinking_end", contentIndex: mapped.index, content: mapped.block.thinking, partial: message() };
				break;
			}
			case "toolcall_start": {
				// A native tool call from the endpoint: pass it through unchanged.
				yield* closeText();
				const msg = message();
				const source = event.partial.content[event.contentIndex];
				msg.content.push(source ? { ...(source as ToolCall) } : { type: "toolCall", id: newId(), name: "", arguments: {} });
				passthroughToolMap.set(event.contentIndex, msg.content.length - 1);
				yield { type: "toolcall_start", contentIndex: msg.content.length - 1, partial: msg };
				break;
			}
			case "toolcall_delta": {
				const index = passthroughToolMap.get(event.contentIndex);
				if (index === undefined) break;
				yield { type: "toolcall_delta", contentIndex: index, delta: event.delta, partial: message() };
				break;
			}
			case "toolcall_end": {
				const index = passthroughToolMap.get(event.contentIndex);
				if (index === undefined) break;
				const msg = message();
				msg.content[index] = event.toolCall;
				yield { type: "toolcall_end", contentIndex: index, toolCall: event.toolCall, partial: msg };
				break;
			}
			case "done": {
				yield* handle(parser.finish());
				yield* closeText();
				const msg = finalize(event.message);
				const hasToolCalls = msg.content.some((c) => c.type === "toolCall");
				const reason = hasToolCalls ? "toolUse" : event.reason === "toolUse" ? "stop" : event.reason;
				msg.stopReason = reason;
				yield { type: "done", reason, message: msg };
				return;
			}
			case "error": {
				if (!out) {
					yield { type: "error", reason: event.reason, error: { ...event.error, ...options.identity } };
					return;
				}
				yield* handle(parser.flushText());
				yield* closeText();
				const msg = finalize(event.error);
				yield { type: "error", reason: event.reason, error: msg };
				return;
			}
		}
	}
}

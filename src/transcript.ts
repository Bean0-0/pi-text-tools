/**
 * Rewrites a pi transcript for a model whose endpoint does not accept native tools.
 *
 * - Tool declarations move into the system prompt, in the Hermes function-calling format.
 * - Earlier assistant tool calls become `<tool_call>` text, so the model sees its own format.
 * - Tool results become user turns wrapped in `<tool_response>` tags.
 */

import type { AssistantMessage, ImageContent, Message, TextContent, Tool, UserMessage } from "@earendil-works/pi-ai/compat";
import { CLOSE_TAG, OPEN_TAG } from "./parser.ts";

export const RESPONSE_OPEN = "<tool_response>";
export const RESPONSE_CLOSE = "</tool_response>";

export function renderToolPrompt(tools: readonly Tool[]): string {
	if (tools.length === 0) return "";
	const declarations = tools
		.map((tool) =>
			JSON.stringify({
				type: "function",
				function: { name: tool.name, description: tool.description, parameters: tool.parameters },
			}),
		)
		.join("\n");

	return [
		"# Tools",
		"",
		"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to complete the user's request. Don't make assumptions about what values to plug into functions.",
		"",
		"<tools>",
		declarations,
		"</tools>",
		"",
		"To call a function, write a JSON object with the function name and arguments within <tool_call></tool_call> XML tags, exactly like this:",
		OPEN_TAG,
		'{"name": "<function-name>", "arguments": {<arguments-object>}}',
		CLOSE_TAG,
		"",
		"Rules:",
		"- The body of each <tool_call> must be a single valid JSON object. Escape newlines and quotes inside strings.",
		"- You may make several calls in one turn by writing several <tool_call> blocks.",
		"- After your tool calls, stop writing and wait. The results arrive in the next message inside <tool_response></tool_response> tags.",
		"- Never write <tool_response> yourself and never guess what a tool returned.",
		"- Use the tools to act (read files, edit files, run commands) instead of describing what you would do.",
		"- When the task is finished, reply in plain text without a <tool_call>.",
	].join("\n");
}

function toolCallText(name: string, args: unknown): string {
	return `${OPEN_TAG}\n${JSON.stringify({ name, arguments: args })}\n${CLOSE_TAG}`;
}

function assistantToText(message: AssistantMessage): AssistantMessage {
	const parts: string[] = [];
	for (const block of message.content) {
		if (block.type === "text" && block.text.trim()) parts.push(block.text.trim());
		else if (block.type === "toolCall") parts.push(toolCallText(block.name, block.arguments));
		// Thinking blocks are dropped: they belong to the earlier response and are not replayed.
	}
	return { ...message, content: parts.length ? [{ type: "text", text: parts.join("\n\n") }] : [] };
}

export interface RewriteOptions {
	/** Whether the model accepts image input. Otherwise images in tool results become placeholders. */
	supportsImages: boolean;
}

/**
 * Convert transcript messages (system messages already removed) into plain user/assistant turns.
 * Consecutive tool results are merged into one user message.
 */
export function rewriteMessages(messages: readonly Message[], options: RewriteOptions): Message[] {
	const out: Message[] = [];
	let pending: UserMessage | undefined;

	const flush = () => {
		if (pending) out.push(pending);
		pending = undefined;
	};

	for (const message of messages) {
		if (message.role === "system") continue;

		if (message.role === "toolResult") {
			const text = message.content
				.filter((c): c is TextContent => c.type === "text")
				.map((c) => c.text)
				.join("\n");
			const images = message.content.filter((c): c is ImageContent => c.type === "image");
			const payload: Record<string, unknown> = { name: message.toolName, content: text };
			if (message.isError) payload.error = true;
			if (images.length && !options.supportsImages) {
				payload.note = `${images.length} image(s) omitted: this model cannot read images.`;
			}

			pending ??= { role: "user", content: [], timestamp: message.timestamp };
			const content = pending.content as (TextContent | ImageContent)[];
			content.push({ type: "text", text: `${RESPONSE_OPEN}\n${JSON.stringify(payload)}\n${RESPONSE_CLOSE}` });
			if (options.supportsImages) content.push(...images);
			continue;
		}

		flush();
		if (message.role === "assistant") {
			const rewritten = assistantToText(message);
			// Skip empty assistant turns (for example a failed request with no content).
			if (rewritten.content.length) out.push(rewritten);
		} else {
			out.push(message);
		}
	}
	flush();
	return out;
}

/** Combine the current system prompt with the tool instructions. */
export function buildSystemPrompt(systemPrompt: string, tools: readonly Tool[]): string {
	const toolPrompt = renderToolPrompt(tools);
	if (!toolPrompt) return systemPrompt;
	return systemPrompt.trim() ? `${systemPrompt.trim()}\n\n${toolPrompt}` : toolPrompt;
}

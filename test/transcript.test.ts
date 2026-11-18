import assert from "node:assert/strict";
import { test } from "node:test";
import { buildSystemPrompt, rewriteMessages } from "../src/transcript.ts";

const base = { api: "x", provider: "p", model: "m", usage: {}, stopReason: "toolUse", timestamp: 1 } as const;

test("system prompt includes tool declarations and format", () => {
	const prompt = buildSystemPrompt("You are pi.", [
		{ name: "read", description: "Read a file", parameters: { type: "object", properties: { path: { type: "string" } } } },
	]);
	assert.ok(prompt.startsWith("You are pi."));
	assert.match(prompt, /<tools>\n\{"type":"function","function":\{"name":"read"/);
	assert.match(prompt, /<tool_call>/);
});

test("no tools leaves the prompt unchanged", () => {
	assert.equal(buildSystemPrompt("You are pi.", []), "You are pi.");
});

test("tool calls and results become text turns", () => {
	const messages = rewriteMessages(
		[
			{ role: "system", content: "sys", timestamp: 0 },
			{ role: "user", content: "list files", timestamp: 1 },
			{
				...base,
				role: "assistant",
				content: [
					{ type: "thinking", thinking: "hmm" },
					{ type: "text", text: "Looking." },
					{ type: "toolCall", id: "1", name: "ls", arguments: { path: "." } },
					{ type: "toolCall", id: "2", name: "read", arguments: { path: "a" } },
				],
			},
			{ role: "toolResult", toolCallId: "1", toolName: "ls", content: [{ type: "text", text: "a\nb" }], isError: false, timestamp: 2 },
			{
				role: "toolResult",
				toolCallId: "2",
				toolName: "read",
				content: [
					{ type: "text", text: "nope" },
					{ type: "image", data: "AAAA", mimeType: "image/png" },
				],
				isError: true,
				timestamp: 3,
			},
		] as never,
		{ supportsImages: false },
	);

	assert.equal(messages.length, 3);
	assert.equal(messages[0].role, "user");

	const assistant = messages[1] as { role: string; content: { type: string; text: string }[] };
	assert.equal(assistant.role, "assistant");
	assert.equal(assistant.content.length, 1);
	assert.equal(
		assistant.content[0].text,
		'Looking.\n\n<tool_call>\n{"name":"ls","arguments":{"path":"."}}\n</tool_call>\n\n<tool_call>\n{"name":"read","arguments":{"path":"a"}}\n</tool_call>',
	);

	const results = messages[2] as { role: string; content: { type: string; text?: string }[] };
	assert.equal(results.role, "user");
	assert.equal(results.content.length, 2);
	assert.equal(results.content[0].text, '<tool_response>\n{"name":"ls","content":"a\\nb"}\n</tool_response>');
	assert.match(results.content[1].text ?? "", /"error":true/);
	assert.match(results.content[1].text ?? "", /image\(s\) omitted/);
});

test("images in tool results are kept for image-capable models", () => {
	const messages = rewriteMessages(
		[
			{
				role: "toolResult",
				toolCallId: "1",
				toolName: "screenshot",
				content: [{ type: "image", data: "AAAA", mimeType: "image/png" }],
				isError: false,
				timestamp: 1,
			},
		] as never,
		{ supportsImages: true },
	);
	const content = (messages[0] as { content: { type: string }[] }).content;
	assert.deepEqual(
		content.map((c) => c.type),
		["text", "image"],
	);
});

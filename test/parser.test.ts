import assert from "node:assert/strict";
import { test } from "node:test";
import { parseToolCallBody, ToolCallParser } from "../src/parser.ts";

function run(chunks: string[]) {
	const parser = new ToolCallParser();
	const events = chunks.flatMap((chunk) => parser.push(chunk));
	events.push(...parser.finish());
	// Merge adjacent text events for easier assertions.
	const merged: typeof events = [];
	for (const event of events) {
		const last = merged.at(-1);
		if (event.kind === "text" && last?.kind === "text") last.text += event.text;
		else merged.push({ ...event });
	}
	return merged;
}

test("plain text passes through", () => {
	assert.deepEqual(run(["Hello ", "world"]), [{ kind: "text", text: "Hello world" }]);
});

test("tool call in one chunk", () => {
	assert.deepEqual(run(['Let me look.\n<tool_call>{"name":"ls","arguments":{}}</tool_call>']), [
		{ kind: "text", text: "Let me look.\n" },
		{ kind: "toolCall", body: '{"name":"ls","arguments":{}}', terminated: true },
	]);
});

test("tags split across every character boundary", () => {
	const input = 'a<tool_call>{"name":"x","arguments":{"p":1}}</tool_call>b';
	for (let size = 1; size <= 5; size++) {
		const chunks: string[] = [];
		for (let i = 0; i < input.length; i += size) chunks.push(input.slice(i, i + size));
		assert.deepEqual(
			run(chunks),
			[
				{ kind: "text", text: "a" },
				{ kind: "toolCall", body: '{"name":"x","arguments":{"p":1}}', terminated: true },
				{ kind: "text", text: "b" },
			],
			`chunk size ${size}`,
		);
	}
});

test("text that looks like a partial tag is released", () => {
	assert.deepEqual(run(["if a <tool", " is not a tag"]), [{ kind: "text", text: "if a <tool is not a tag" }]);
	assert.deepEqual(run(["ends with <tool_"]), [{ kind: "text", text: "ends with <tool_" }]);
});

test("multiple tool calls", () => {
	const events = run(['<tool_call>{"name":"a","arguments":{}}</tool_call>\n<tool_call>{"name":"b","arguments":{}}</tool_call>']);
	assert.deepEqual(
		events.filter((e) => e.kind === "toolCall").map((e) => parseToolCallBody((e as { body: string }).body).name),
		["a", "b"],
	);
});

test("unterminated tool call at end is still a tool call", () => {
	assert.deepEqual(run(['<tool_call>\n{"name":"a","arguments":{}}\n']), [
		{ kind: "toolCall", body: '\n{"name":"a","arguments":{}}\n', terminated: false },
	]);
});

test("parse: standard body", () => {
	assert.deepEqual(parseToolCallBody('\n{"name": "read", "arguments": {"path": "a.ts"}}\n'), {
		name: "read",
		arguments: { path: "a.ts" },
	});
});

test("parse: fenced body, parameters alias, string arguments", () => {
	assert.deepEqual(parseToolCallBody('```json\n{"name":"bash","parameters":{"command":"ls"}}\n```'), {
		name: "bash",
		arguments: { command: "ls" },
	});
	assert.deepEqual(parseToolCallBody('{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}'), {
		name: "bash",
		arguments: { command: "ls" },
	});
});

test("parse: stray text around the object", () => {
	assert.equal(parseToolCallBody('call: {"name":"ls","arguments":{}} thanks').name, "ls");
});

test("parse: invalid JSON keeps the name and reports an error", () => {
	const parsed = parseToolCallBody('{"name": "edit", "arguments": {"path": "a", oops}}');
	assert.equal(parsed.name, "edit");
	assert.deepEqual(parsed.arguments, {});
	assert.match(parsed.error ?? "", /not valid JSON/);
});

test("parse: missing name", () => {
	const parsed = parseToolCallBody('{"arguments": {}}');
	assert.equal(parsed.name, "invalid_tool_call");
	assert.ok(parsed.error);
});

test("implicit call: opening tag stripped by the endpoint", () => {
	// What Hermes on Nebius actually returns: the <tool_call> special token is removed.
	const input = '\n{"name": "read", "arguments": {"path": "calc.py"}}\n</tool_call>';
	for (let size = 1; size <= 4; size++) {
		const chunks: string[] = [];
		for (let i = 0; i < input.length; i += size) chunks.push(input.slice(i, i + size));
		const events = run(chunks).filter((e) => e.kind === "toolCall" || e.text.trim());
		assert.equal(events.length, 1, `chunk size ${size}`);
		assert.equal(events[0].kind, "toolCall");
		assert.equal(parseToolCallBody((events[0] as { body: string }).body).name, "read");
	}
});

test("implicit call after prose and between calls", () => {
	const events = run([
		'I will read it.\n{"name": "read", "arguments": {}}\n</tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>',
	]);
	assert.deepEqual(
		events.map((e) => e.kind),
		["text", "toolCall", "text", "toolCall"],
	);
});

test("implicit call without closing tag needs a known tool name", () => {
	const known = new ToolCallParser({ isKnownTool: (name) => name === "bash" });
	known.push('{"name": "bash", "arguments": {"command": "ls"}}');
	assert.equal(known.finish()[0].kind, "toolCall");

	const unknown = new ToolCallParser({ isKnownTool: (name) => name === "bash" });
	unknown.push('{"name": "Alice", "age": 3}');
	assert.deepEqual(unknown.finish(), [{ kind: "text", text: '{"name": "Alice", "age": 3}' }]);
});

test("JSON mid-line is not an implicit call", () => {
	assert.deepEqual(run(['Use {"name": "x"} as input']), [{ kind: "text", text: 'Use {"name": "x"} as input' }]);
});

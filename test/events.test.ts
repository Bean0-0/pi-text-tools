import assert from "node:assert/strict";
import { test } from "node:test";
import { convertEvents } from "../src/events.ts";

const identity = { api: "openrouter-text-tools", provider: "openrouter-tt", model: "m" };

function baseMessage() {
	return {
		role: "assistant",
		content: [],
		api: "openai-completions",
		provider: "openrouter-tt",
		model: "m",
		usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 15, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "pending",
		timestamp: 1,
	};
}

/** Build an inner stream that emits the given text deltas as one text block. */
async function* textStream(deltas: string[], opts: { thinking?: string; reason?: string } = {}) {
	const partial = baseMessage() as any;
	yield { type: "start", partial };
	let index = 0;
	if (opts.thinking) {
		partial.content.push({ type: "thinking", thinking: opts.thinking });
		yield { type: "thinking_start", contentIndex: 0, partial };
		yield { type: "thinking_delta", contentIndex: 0, delta: opts.thinking, partial };
		yield { type: "thinking_end", contentIndex: 0, content: opts.thinking, partial };
		index = 1;
	}
	partial.content.push({ type: "text", text: "" });
	yield { type: "text_start", contentIndex: index, partial };
	for (const delta of deltas) {
		partial.content[index].text += delta;
		yield { type: "text_delta", contentIndex: index, delta, partial };
	}
	yield { type: "text_end", contentIndex: index, content: partial.content[index].text, partial };
	yield { type: "done", reason: opts.reason ?? "stop", message: { ...partial, stopReason: opts.reason ?? "stop" } };
}

async function collect(deltas: string[], opts?: { thinking?: string; reason?: string }) {
	let n = 0;
	const events: any[] = [];
	for await (const event of convertEvents(textStream(deltas, opts) as any, { identity, newId: () => `id${n++}` })) {
		events.push(event);
	}
	return events;
}

test("plain text response streams through", async () => {
	const events = await collect(["Hel", "lo"]);
	assert.deepEqual(
		events.map((e) => e.type),
		["start", "text_start", "text_delta", "text_delta", "text_end", "done"],
	);
	const done = events.at(-1);
	assert.equal(done.reason, "stop");
	assert.deepEqual(done.message.content, [{ type: "text", text: "Hello" }]);
	assert.equal(done.message.provider, "openrouter-tt");
	assert.equal(done.message.api, "openrouter-text-tools");
	assert.equal(done.message.usage.totalTokens, 15);
});

test("text plus tool call becomes native tool call", async () => {
	const events = await collect(["I'll list files.\n<tool", '_call>\n{"name": "ls", "argu', 'ments": {"path": "."}}\n</tool_call>']);
	const done = events.at(-1);
	assert.equal(done.reason, "toolUse");
	assert.equal(done.message.stopReason, "toolUse");
	assert.deepEqual(done.message.content, [
		{ type: "text", text: "I'll list files.\n" },
		{ type: "toolCall", id: "id0", name: "ls", arguments: { path: "." } },
	]);
	const end = events.find((e) => e.type === "toolcall_end");
	assert.equal(end.contentIndex, 1);
	assert.deepEqual(end.toolCall.arguments, { path: "." });
});

test("whitespace between tool calls creates no text blocks", async () => {
	const events = await collect([
		'<tool_call>{"name":"a","arguments":{}}</tool_call>\n\n<tool_call>{"name":"b","arguments":{}}</tool_call>\n',
	]);
	const done = events.at(-1);
	assert.deepEqual(
		done.message.content.map((c: any) => c.type),
		["toolCall", "toolCall"],
	);
	assert.ok(!events.some((e) => e.type === "text_start"));
});

test("unterminated call (stop sequence) still runs", async () => {
	const events = await collect(['<tool_call>\n{"name":"bash","arguments":{"command":"ls"}}\n']);
	const done = events.at(-1);
	assert.equal(done.reason, "toolUse");
	assert.equal(done.message.content[0].name, "bash");
});

test("malformed call is reported through argumentsError", async () => {
	const events = await collect(['<tool_call>{"name":"edit","arguments":{oops}}</tool_call>']);
	const call = events.at(-1).message.content[0];
	assert.equal(call.name, "edit");
	assert.match(call.argumentsError, /not valid JSON/);
});

test("thinking blocks keep their position and indexes stay consistent", async () => {
	const events = await collect(['ok <tool_call>{"name":"ls","arguments":{}}</tool_call>'], { thinking: "plan" });
	const done = events.at(-1);
	assert.deepEqual(
		done.message.content.map((c: any) => c.type),
		["thinking", "text", "toolCall"],
	);
	for (const event of events) {
		if ("contentIndex" in event) assert.ok(event.contentIndex < event.partial.content.length);
	}
});

test("length stop without tool calls keeps its reason", async () => {
	const events = await collect(["truncated"], { reason: "length" });
	assert.equal(events.at(-1).reason, "length");
});

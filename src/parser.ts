/**
 * Streaming parser that splits model output into plain text and `<tool_call>` blocks.
 *
 * Text arrives in arbitrary chunks, so a tag can be split across deltas. The parser holds back
 * any trailing text that could start a tool call until the next chunk decides it.
 *
 * Some endpoints strip `<tool_call>` from the output because it is a special token in the model's
 * tokenizer (Hermes on Nebius does this), leaving only `{"name": ...}` and `</tool_call>`. So a
 * line that starts with `{"name":` also opens a tool call. Such an implicit call ends at
 * `</tool_call>`; if the response ends first, it only counts when its name is a known tool.
 */

export const OPEN_TAG = "<tool_call>";
export const CLOSE_TAG = "</tool_call>";
const IMPLICIT_PREFIX = '{"name":';
const IMPLICIT_START = /(^|\n)[ \t]*\{\s*"name"\s*:/g;

export type ParserEvent =
	| { kind: "text"; text: string }
	| { kind: "toolCall"; body: string; terminated: boolean };

export interface ParserOptions {
	/** Used to accept an implicit tool call that ends without `</tool_call>`. */
	isKnownTool?: (name: string) => boolean;
}

/** Length of the longest suffix of `text` that is a proper prefix of `tag`. */
function partialTagSuffix(text: string, tag: string): number {
	const max = Math.min(text.length, tag.length - 1);
	for (let len = max; len > 0; len--) {
		if (tag.startsWith(text.slice(text.length - len))) return len;
	}
	return 0;
}

/** Length of the trailing line if it could still grow into `{"name":`, otherwise 0. */
function partialImplicitSuffix(text: string, atLineStart: boolean): number {
	const newline = text.lastIndexOf("\n");
	if (newline === -1 && !atLineStart) return 0;
	const line = text.slice(newline + 1);
	const compact = line.replace(/\s+/g, "");
	if (!compact) return line.length;
	return IMPLICIT_PREFIX.startsWith(compact) ? line.length : 0;
}

export class ToolCallParser {
	private inside = false;
	private implicit = false;
	private buffer = "";
	/** Whether the start of `buffer` is the start of a line. */
	private atLineStart = true;
	private readonly options: ParserOptions;

	constructor(options: ParserOptions = {}) {
		this.options = options;
	}

	/** Feed a chunk of model text. Returns the text and completed tool calls it resolved. */
	push(chunk: string): ParserEvent[] {
		this.buffer += chunk;
		const events: ParserEvent[] = [];
		const emitText = (text: string) => {
			if (!text) return;
			events.push({ kind: "text", text });
			this.atLineStart = text.endsWith("\n");
		};

		while (true) {
			if (!this.inside) {
				const explicit = this.buffer.indexOf(OPEN_TAG);
				let implicit = -1;
				for (const match of this.buffer.matchAll(IMPLICIT_START)) {
					if (match[1] === "\n" || this.atLineStart) {
						implicit = match.index + match[1].length;
						break;
					}
				}

				if (explicit !== -1 && (implicit === -1 || explicit <= implicit)) {
					emitText(this.buffer.slice(0, explicit));
					this.buffer = this.buffer.slice(explicit + OPEN_TAG.length);
					this.inside = true;
					this.implicit = false;
				} else if (implicit !== -1) {
					emitText(this.buffer.slice(0, implicit));
					this.buffer = this.buffer.slice(implicit);
					this.inside = true;
					this.implicit = true;
				} else {
					const hold = Math.max(
						partialTagSuffix(this.buffer, OPEN_TAG),
						partialImplicitSuffix(this.buffer, this.atLineStart),
					);
					emitText(this.buffer.slice(0, this.buffer.length - hold));
					this.buffer = this.buffer.slice(this.buffer.length - hold);
					return events;
				}
			} else {
				const close = this.buffer.indexOf(CLOSE_TAG);
				if (close === -1) return events;
				events.push({ kind: "toolCall", body: this.buffer.slice(0, close), terminated: true });
				this.buffer = this.buffer.slice(close + CLOSE_TAG.length);
				this.inside = false;
				this.implicit = false;
				this.atLineStart = true;
			}
		}
	}

	/** Whether the parser is in the middle of a tool call. */
	get inToolCall(): boolean {
		return this.inside;
	}

	/**
	 * Flush everything at the end of the response. An unterminated `<tool_call>` is returned as a
	 * tool call: models often stop before writing the closing tag, especially at a stop sequence.
	 */
	finish(): ParserEvent[] {
		const rest = this.buffer;
		const wasImplicit = this.implicit;
		this.buffer = "";
		this.implicit = false;
		if (this.inside) {
			this.inside = false;
			if (!rest.trim()) return [];
			if (wasImplicit) {
				const parsed = parseToolCallBody(rest);
				const known = this.options.isKnownTool?.(parsed.name) ?? false;
				if (parsed.error || !known) return [{ kind: "text", text: rest }];
			}
			return [{ kind: "toolCall", body: rest, terminated: false }];
		}
		return rest ? [{ kind: "text", text: rest }] : [];
	}

	/** Flush held-back text without ending a tool call in progress. */
	flushText(): ParserEvent[] {
		if (this.inside || !this.buffer) return [];
		const text = this.buffer;
		this.buffer = "";
		this.atLineStart = text.endsWith("\n");
		return [{ kind: "text", text }];
	}
}

export interface ParsedToolCall {
	name: string;
	arguments: Record<string, unknown>;
	/** Set when the body could not be parsed into a name and an arguments object. */
	error?: string;
}

function stripFences(body: string): string {
	return body
		.trim()
		.replace(/^```(?:json)?\s*/i, "")
		.replace(/\s*```$/, "")
		.trim();
}

function tryJson(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch {
		return undefined;
	}
}

/** Parse the JSON body of a `<tool_call>` block, tolerating fences and stray text around the object. */
export function parseToolCallBody(body: string): ParsedToolCall {
	const cleaned = stripFences(body);
	let value = tryJson(cleaned);
	if (value === undefined) {
		const start = cleaned.indexOf("{");
		const end = cleaned.lastIndexOf("}");
		if (start !== -1 && end > start) value = tryJson(cleaned.slice(start, end + 1));
	}

	const nameMatch = /"name"\s*:\s*"([^"]+)"/.exec(cleaned);
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		return {
			name: nameMatch?.[1] ?? "invalid_tool_call",
			arguments: {},
			error: `The <tool_call> body is not valid JSON. Write exactly one JSON object like {"name": "...", "arguments": {...}}. Received: ${cleaned.slice(0, 500)}`,
		};
	}

	const obj = value as Record<string, unknown>;
	const name = typeof obj.name === "string" ? obj.name : nameMatch?.[1];
	let args: unknown = obj.arguments ?? obj.parameters ?? obj.args ?? {};
	if (typeof args === "string") args = tryJson(args) ?? args;

	if (!name) {
		return {
			name: "invalid_tool_call",
			arguments: {},
			error: `The <tool_call> JSON has no "name" field. Received: ${cleaned.slice(0, 500)}`,
		};
	}
	if (!args || typeof args !== "object" || Array.isArray(args)) {
		return {
			name,
			arguments: {},
			error: `The "arguments" field must be a JSON object. Received: ${JSON.stringify(args).slice(0, 500)}`,
		};
	}
	return { name, arguments: args as Record<string, unknown> };
}

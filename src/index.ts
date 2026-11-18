/**
 * pi-text-tools: tool calling for OpenRouter models whose endpoints do not support native tools.
 *
 * Registers the `openrouter-tt` provider. Its models receive pi's tools as text in the system
 * prompt (Hermes function-calling format), and `<tool_call>` blocks in their replies are turned
 * back into native pi tool calls. Requests go through pi's built-in OpenAI Chat Completions
 * implementation, so streaming, reasoning, usage and cost accounting work as usual.
 */

import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	getCurrentSystemPrompt,
	getCurrentTools,
	type Model,
	normalizeContext,
	openAICompletionsApi,
	type SimpleStreamOptions,
	type TranscriptContext,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { convertEvents } from "./events.ts";
import { buildSystemPrompt, RESPONSE_OPEN, rewriteMessages } from "./transcript.ts";

export const PROVIDER_ID = "openrouter-tt";
const API_ID = "openrouter-text-tools";
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";
/** Placeholder key: requests resolve the real key from pi's `openrouter` credentials. */
const SHARED_KEY = "use-openrouter-credentials";

interface ModelEntry {
	id: string;
	name?: string;
	reasoning?: boolean;
	input?: ("text" | "image")[];
	contextWindow?: number;
	maxTokens?: number;
	cost?: { input: number; output: number; cacheRead: number; cacheWrite: number };
}

interface Config {
	models?: ModelEntry[];
	/** Add the configured models to the defaults instead of replacing them. Default true. */
	extendDefaults?: boolean;
}

/** OpenRouter models whose only endpoints ignore the `tools` parameter. */
export const DEFAULT_MODELS: ModelEntry[] = [
	{
		id: "nousresearch/hermes-4-405b",
		name: "Hermes 4 405B (text tools)",
		// OpenRouter lists reasoning support, but with the `reasoning` parameter set this endpoint
		// returns no reasoning and repeats the previous tool call instead of using its result.
		reasoning: false,
		contextWindow: 131072,
		maxTokens: 32768,
		cost: { input: 1, output: 3, cacheRead: 0.2, cacheWrite: 0 },
	},
	{
		id: "cognitivecomputations/dolphin-mistral-24b-venice-edition",
		name: "Venice Uncensored (text tools)",
		contextWindow: 128000,
		maxTokens: 8192,
		cost: { input: 0.2, output: 0.9, cacheRead: 0, cacheWrite: 0 },
	},
	{
		id: "thedrummer/cydonia-24b-v4.1",
		name: "Cydonia 24B v4.1 (text tools)",
		contextWindow: 131072,
		maxTokens: 32768,
		cost: { input: 0.3, output: 0.5, cacheRead: 0.15, cacheWrite: 0 },
	},
];

function agentDir(): string {
	return process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
}

function loadConfig(): Config {
	const path = join(agentDir(), "text-tools.json");
	if (!existsSync(path)) return {};
	try {
		return JSON.parse(readFileSync(path, "utf8")) as Config;
	} catch (error) {
		console.warn(`pi-text-tools: ignoring ${path}: ${(error as Error).message}`);
		return {};
	}
}

function resolveModels(config: Config): ModelEntry[] {
	const configured = config.models ?? [];
	const base = config.extendDefaults === false ? [] : DEFAULT_MODELS;
	const byId = new Map<string, ModelEntry>();
	for (const entry of [...base, ...configured]) byId.set(entry.id, { ...byId.get(entry.id), ...entry });
	return [...byId.values()];
}

/** Read the OpenRouter key pi stored with /login, without going through the registry. */
function readStoredOpenRouterKey(): string | undefined {
	try {
		const auth = JSON.parse(readFileSync(join(agentDir(), "auth.json"), "utf8"));
		const entry = auth?.openrouter;
		return typeof entry?.key === "string" ? entry.key : undefined;
	} catch {
		return undefined;
	}
}

type KeyResolver = () => Promise<string | undefined>;

/** Set PI_TEXT_TOOLS_DEBUG=/path/to/file to append the raw model text and thinking of each response. */
async function* debugTap(events: AsyncIterable<AssistantMessageEvent>): AsyncGenerator<AssistantMessageEvent> {
	const path = process.env.PI_TEXT_TOOLS_DEBUG;
	if (!path) return yield* events;
	let text = "";
	let thinking = "";
	for await (const event of events) {
		if (event.type === "text_delta") text += event.delta;
		else if (event.type === "thinking_delta") thinking += event.delta;
		else if (event.type === "done" || event.type === "error") {
			const record = {
				time: new Date().toISOString(),
				type: event.type,
				reason: event.reason,
				thinking,
				text,
				error: event.type === "error" ? event.error.errorMessage : undefined,
			};
			appendFileSync(path, `${JSON.stringify(record)}\n`);
		}
		yield event;
	}
}

export function createStream(resolveKey: KeyResolver) {
	const inner = openAICompletionsApi();

	return function streamTextTools(
		model: Model<Api>,
		context: TranscriptContext,
		options?: SimpleStreamOptions,
	): AssistantMessageEventStream {
		const stream = createAssistantMessageEventStream();
		const identity = { api: model.api, provider: model.provider, model: model.id };

		(async () => {
			let apiKey = options?.apiKey;
			if (!apiKey || apiKey === SHARED_KEY) apiKey = await resolveKey();
			if (!apiKey) throw new Error("No OpenRouter API key. Run /login openrouter or set OPENROUTER_API_KEY.");

			const tools = getCurrentTools(context.messages);
			const systemPrompt = buildSystemPrompt(getCurrentSystemPrompt(context.messages), tools);
			const messages = rewriteMessages(context.messages, { supportsImages: model.input.includes("image") });
			const rewritten = normalizeContext({ systemPrompt, messages, tools: [] });

			const userOnPayload = options?.onPayload;
			const innerOptions: SimpleStreamOptions = {
				...options,
				apiKey,
				toolChoice: undefined,
				onPayload: async (payload, payloadModel) => {
					const body = { ...(payload as Record<string, unknown>) };
					delete body.tools;
					delete body.tool_choice;
					delete body.parallel_tool_calls;
					if (!model.reasoning) delete body.reasoning;
					// Stop before the model invents a tool result.
					const stop = Array.isArray(body.stop) ? (body.stop as string[]) : [];
					body.stop = [...new Set([...stop, RESPONSE_OPEN])];
					if (process.env.PI_TEXT_TOOLS_DEBUG) {
						appendFileSync(process.env.PI_TEXT_TOOLS_DEBUG, `${JSON.stringify({ time: new Date().toISOString(), type: "request", body })}\n`);
					}
					const replaced = await userOnPayload?.(body, payloadModel);
					return replaced ?? body;
				},
			};

			const innerModel = { ...model, api: "openai-completions" } as Model<"openai-completions">;
			const events = debugTap(inner.streamSimple(innerModel, rewritten, innerOptions));
			for await (const event of convertEvents(events, { identity, toolNames: tools.map((tool) => tool.name) })) stream.push(event);
			stream.end();
		})().catch((error: unknown) => {
			stream.push({
				type: "error",
				reason: options?.signal?.aborted ? "aborted" : "error",
				error: {
					role: "assistant",
					content: [],
					...identity,
					usage: {
						input: 0,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 0,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason: options?.signal?.aborted ? "aborted" : "error",
					errorMessage: error instanceof Error ? error.message : String(error),
					timestamp: Date.now(),
				},
			});
			stream.end();
		});

		return stream;
	};
}

export default function (pi: ExtensionAPI) {
	let registryKey: KeyResolver | undefined;
	pi.on("session_start", (_event, ctx) => {
		registryKey = () => ctx.modelRegistry.getApiKeyForProvider("openrouter");
	});

	const resolveKey: KeyResolver = async () =>
		(await registryKey?.().catch(() => undefined)) ?? process.env.OPENROUTER_API_KEY ?? readStoredOpenRouterKey();

	const models = resolveModels(loadConfig());
	pi.registerProvider(PROVIDER_ID, {
		name: "OpenRouter (text tools)",
		baseUrl: OPENROUTER_BASE_URL,
		apiKey: SHARED_KEY,
		api: API_ID,
		models: models.map((entry) => ({
			id: entry.id,
			name: entry.name ?? `${entry.id} (text tools)`,
			reasoning: entry.reasoning ?? false,
			input: entry.input ?? ["text"],
			contextWindow: entry.contextWindow ?? 128000,
			maxTokens: entry.maxTokens ?? 16384,
			cost: entry.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		})),
		streamSimple: createStream(resolveKey),
	});
}

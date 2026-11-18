# pi-text-tools

Tool calling for [pi](https://pi.dev) with OpenRouter models whose endpoints don't support tools.

Some models on OpenRouter are only served by providers that ignore the `tools` parameter. Examples are Nous Hermes 4 405B, Venice Uncensored (Dolphin Mistral 24B) and Cydonia. In pi they can chat, but they can't read files, edit code or run commands.

This extension adds an `openrouter-tt` provider ("text tools") that emulates tool calling in the prompt:

1. pi's tool definitions are written into the system prompt in the Hermes function-calling format (`<tools>…</tools>`).
2. The model replies with `<tool_call>{"name": …, "arguments": {…}}</tool_call>` blocks.
3. The extension parses those blocks while the reply streams and hands them to pi as native tool calls. pi runs the tools as usual.
4. Tool results go back to the model as `<tool_response>` blocks, and earlier calls are replayed as `<tool_call>` text.

Requests go through pi's built-in OpenAI Chat Completions implementation, so streaming, usage, cost accounting and cancellation behave like any other OpenRouter model. Text streams live, and only the `<tool_call>` blocks are held until they're complete.

## Install

```bash
pi install git:github.com/Bean0-0/pi-text-tools
```

Or try it without installing:

```bash
pi -e git:github.com/Bean0-0/pi-text-tools
```

The extension uses your existing OpenRouter key. That's the one stored with `/login openrouter`, or `OPENROUTER_API_KEY` if you set it.

## Use

Pick one of the `openrouter-tt/…` models with `/model`, or start pi with one:

```bash
pi --model openrouter-tt/nousresearch/hermes-4-405b
```

| Model | Notes |
|---|---|
| `openrouter-tt/nousresearch/hermes-4-405b` | Trained on this exact format, so it's the most reliable. |
| `openrouter-tt/cognitivecomputations/dolphin-mistral-24b-venice-edition` | Works for simple multi-step tasks. |
| `openrouter-tt/thedrummer/cydonia-24b-v4.1` | A roleplay-tuned model, so expect more format mistakes. |

To get native tool use in pi tools such as T3 Code, add the model to `enabledModels` in `~/.pi/agent/settings.json`.

## Add models

Create `~/.pi/agent/text-tools.json`:

```json
{
  "models": [
    {
      "id": "some-vendor/some-chat-only-model",
      "name": "Some Model (text tools)",
      "contextWindow": 32768,
      "maxTokens": 8192,
      "cost": { "input": 0.5, "output": 1.5, "cacheRead": 0, "cacheWrite": 0 }
    }
  ]
}
```

Entries are merged with the defaults by `id`. Set `"extendDefaults": false` to use only your list. Any field other than `id` is optional. To check whether a model needs this extension, look at its endpoints on OpenRouter: if none list `tools` under supported parameters, it does.

## Things that behave differently from native tools

- **Reasoning is off for Hermes 4.** With OpenRouter's `reasoning` parameter set, Hermes 4 on its only endpoint (Nebius) returns no reasoning and keeps repeating its previous tool call. The extension removes the parameter for any model whose entry has `"reasoning": false`.
- **Stripped tags.** `<tool_call>` is a special token in the Hermes tokenizer, and some endpoints strip it from the output. A line that starts with `{"name":` is therefore also treated as the start of a call. If the closing tag is missing too, the line only counts as a call when the name matches a real tool.
- **Malformed calls.** If a call's JSON doesn't parse, it reaches pi as a tool call with an error. The model sees the error and can retry.
- **Context use.** Tool definitions are sent as prompt text on every request. They use about as much context as native tool definitions would.
- **Smaller models** follow the format less reliably than Hermes.

## Debug

```bash
PI_TEXT_TOOLS_DEBUG=/tmp/text-tools.jsonl pi --model openrouter-tt/nousresearch/hermes-4-405b
```

This appends each request body and each raw model response (text and thinking) to the file. Request bodies include your prompts and file contents, so only turn it on while debugging.

## Develop

```bash
npm test            # unit tests (Node ≥ 22.18, no dependencies)
pi -e ./src/index.ts --model openrouter-tt/nousresearch/hermes-4-405b
```

`src/parser.ts` is the streaming `<tool_call>` parser, `src/transcript.ts` rewrites the transcript, `src/events.ts` turns the text stream into pi tool-call events, and `src/index.ts` registers the provider.

## License

MIT

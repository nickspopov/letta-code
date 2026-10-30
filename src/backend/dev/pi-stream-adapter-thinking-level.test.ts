import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  AssistantMessage,
  AssistantMessageEvent,
  Context,
} from "@earendil-works/pi-ai";
import type { ConversationMessageCreateBody } from "@/backend";
import type { PiStreamFunction } from "@/backend/dev/pi-stream-adapter";
import { LocalBackend } from "@/backend/local/local-backend";
import { emptyLocalUsage } from "@/backend/local/local-message";
import { LocalStore } from "@/backend/local/local-store";

function response(step: number, level: string | undefined): AssistantMessage {
  return {
    role: "assistant",
    content: [
      {
        type: "thinking",
        thinking: `step ${step}`,
        thinkingSignature: "signature",
      },
      ...(step < 2
        ? [
            {
              type: "toolCall" as const,
              id: `call-${step}`,
              name: "Read",
              arguments: { path: "example.txt" },
            },
          ]
        : [{ type: "text" as const, text: "done" }]),
    ],
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-sonnet-4-6",
    ...(level !== undefined ? { providerThinkingLevel: level } : {}),
    usage: emptyLocalUsage(),
    stopReason: step < 2 ? "toolUse" : "stop",
    timestamp: Date.now(),
  };
}

function streamResponse(
  message: AssistantMessage,
): ReturnType<PiStreamFunction> {
  async function* iterator() {
    yield {
      type: "done",
      reason: message.stopReason,
      message,
    } as AssistantMessageEvent;
  }
  return Object.assign(iterator(), { result: async () => message });
}

describe("provider thinking level replay", () => {
  for (const levels of [
    ["high", "low", "max"],
    [undefined, undefined, undefined],
  ]) {
    test(`preserves ${levels[0] === undefined ? "absent" : "historical"} effort across stored tool steps`, async () => {
      const storageDir = await mkdtemp(
        join(tmpdir(), "thinking-level-replay-"),
      );
      try {
        const contexts: Context[] = [];
        const stream: PiStreamFunction = (_model, context) => {
          const step = contexts.length;
          contexts.push(context);
          return streamResponse(response(step, levels[step]));
        };
        const options = { storageDir, stream, memfsEnabled: false };
        let backend = new LocalBackend(options);
        const agent = await backend.createAgent({
          name: "Replay",
          model: "anthropic/claude-sonnet-4-6",
        });
        const conversation = await backend.createConversation({
          agent_id: agent.id,
        });
        for (let step = 0; step < 3; step++) {
          const body = {
            agent_id: agent.id,
            messages:
              step === 0
                ? [{ role: "user", content: "Read twice" }]
                : [
                    {
                      type: "approval",
                      approvals: [
                        {
                          type: "tool",
                          tool_call_id: `call-${step - 1}`,
                          status: "success",
                          tool_return: "contents",
                        },
                      ],
                    },
                  ],
          } as ConversationMessageCreateBody;
          for await (const chunk of await backend.createConversationMessageStream(
            conversation.id,
            body,
          )) {
            void chunk;
          }
          const stored = new LocalStore(agent.id, { storageDir })
            .listLocalMessages(conversation.id, agent.id)
            .filter((message) => message.role === "assistant");
          expect(stored).toHaveLength(step + 1);
          for (const [index, message] of stored.entries()) {
            expect(message.providerThinkingLevel).toBe(levels[index]);
            expect(Object.hasOwn(message, "providerThinkingLevel")).toBe(
              levels[index] !== undefined,
            );
          }
          // Force disk reload before each tool continuation, not just an in-memory replay.
          backend = new LocalBackend(options);
        }
        expect(contexts).toHaveLength(3);
        for (let step = 1; step < 3; step++) {
          const messages = contexts[step]?.messages ?? [];
          const assistants = messages.filter(
            (message) => message.role === "assistant",
          );
          expect(assistants).toHaveLength(step);
          for (const [index, message] of assistants.entries()) {
            expect(message.providerThinkingLevel).toBe(levels[index]);
            expect(Object.hasOwn(message, "providerThinkingLevel")).toBe(
              levels[index] !== undefined,
            );
            expect(message.content).toEqual(
              response(index, levels[index]).content,
            );
          }
          expect(messages.at(-1)).toMatchObject({
            role: "toolResult",
            toolCallId: `call-${step - 1}`,
          });
        }
      } finally {
        await rm(storageDir, { recursive: true, force: true });
      }
    });
  }
});

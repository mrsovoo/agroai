"use client";

import type { ChatMessage } from "@/hooks/useChat";
import { Markdown } from "./Markdown";
import { SourceCards } from "./SourceCards";

function Bubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-leaf-600 px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-sm md:max-w-[70%]">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 shadow-sm dark:border-red-900 dark:bg-red-950 dark:text-red-300">
          <p>⚠️ {message.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[92%] rounded-2xl rounded-bl-md border border-soil-200 bg-white px-4 py-3 shadow-sm dark:border-soil-700 dark:bg-soil-900 md:max-w-[80%]">
        <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-leaf-700 dark:text-leaf-400">
          <span aria-hidden>🌱</span> AgroAI
        </div>
        <Markdown text={message.content} />
        {message.warnings && message.warnings.length > 0 && (
          <div className="mt-3 space-y-1">
            {message.warnings.map((warning, index) => (
              <p
                key={index}
                className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-950 dark:text-amber-200"
              >
                ⚠️ {warning}
              </p>
            ))}
          </div>
        )}
        <SourceCards sources={message.sources || []} />
      </div>
    </div>
  );
}

export function MessageList({ messages, isLoading }: { messages: ChatMessage[]; isLoading: boolean }) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6">
      {messages.map((message) => (
        <Bubble key={message.id} message={message} />
      ))}
      {isLoading && (
        <div className="flex justify-start">
          <div className="flex items-center gap-2 rounded-2xl rounded-bl-md border border-soil-200 bg-white px-4 py-3 shadow-sm dark:border-soil-700 dark:bg-soil-900">
            <span className="h-2 w-2 animate-bounce rounded-full bg-leaf-500 [animation-delay:-0.2s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-leaf-500 [animation-delay:-0.1s]" />
            <span className="h-2 w-2 animate-bounce rounded-full bg-leaf-500" />
            <span className="sr-only">AgroAI javob tayyorlamoqda…</span>
          </div>
        </div>
      )}
    </div>
  );
}

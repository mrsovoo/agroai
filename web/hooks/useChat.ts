"use client";

import { useCallback, useRef, useState } from "react";
import {
  ApiUnavailableError,
  api,
  type ChatResponse,
  type SourceItem,
} from "../lib/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  warnings?: string[];
  modelStatus?: string;
  error?: string;
}

export function useChat(initialConversationId: string | null = null) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(
    initialConversationId
  );
  const [isLoading, setIsLoading] = useState(false);
  const [lastFailed, setLastFailed] = useState<string | null>(null);
  const counter = useRef(0);

  const nextId = () => `local-${++counter.current}`;

  const send = useCallback(
    async (text: string) => {
      const clean = text.trim();
      if (!clean || isLoading) return;

      setLastFailed(null);
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "user", content: clean },
      ]);
      setIsLoading(true);

      try {
        const response: ChatResponse = await api.chat(conversationId, clean);
        if (response.conversation_id && response.conversation_id !== conversationId) {
          setConversationId(response.conversation_id);
        }
        setMessages((prev) => [
          ...prev,
          {
            id: response.message_id || nextId(),
            role: "assistant",
            content: response.answer,
            sources: response.sources,
            warnings: [
              ...(response.safety?.disclaimers || []),
              ...(response.limitations || []),
            ],
            modelStatus: response.status,
          },
        ]);
      } catch (error) {
        const message =
          error instanceof ApiUnavailableError
            ? "AgroAI serveriga ulanib bo'lmadi. Server ishga tushirilganini tekshiring."
            : error instanceof Error
              ? error.message
              : "Noma'lum xatolik.";
        setLastFailed(clean);
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "assistant", content: "", error: message },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [conversationId, isLoading]
  );

  const retry = useCallback(() => {
    if (lastFailed) {
      const text = lastFailed;
      setMessages((prev) => {
        const copy = [...prev];
        while (copy.length && copy[copy.length - 1].role !== "user") copy.pop();
        if (copy.length && copy[copy.length - 1].role === "user") copy.pop();
        return copy;
      });
      setLastFailed(null);
      void send(text);
    }
  }, [lastFailed, send]);

  const reset = useCallback(() => {
    setMessages([]);
    setConversationId(null);
    setLastFailed(null);
  }, []);

  return { messages, conversationId, isLoading, send, retry, reset, hasError: lastFailed !== null };
}

/** Typed API client for the AgroAI backend (the one shared core). */

export const API_URL =
  process.env.NEXT_PUBLIC_AGROAI_API_URL || "http://127.0.0.1:8080";

export interface SourceItem {
  index: number;
  title: string;
  source: string;
  url: string;
  document_id: string;
  score: number;
}

export interface ChatResponse {
  conversation_id?: string;
  message_id?: string;
  answer: string;
  status: string;
  grounded: boolean;
  sources: SourceItem[];
  safety: { blocked: boolean; disclaimers: string[] };
  limitations: string[];
  plan?: { skills: string[]; steps: string[] };
}

export interface ConversationSummary {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface StoredMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface HealthReport {
  status: string;
  version: string;
  retrieval?: { ready: boolean; documents: number; chunks: number; search_mode: string };
  llm?: { configured: boolean; provider: string; detail: string };
  knowledge?: Record<string, unknown>;
}

export class ApiUnavailableError extends Error {
  constructor() {
    super("AgroAI serveriga ulanib bo'lmadi.");
    this.name = "ApiUnavailableError";
  }
}

export class QuotaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "QuotaError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      signal: AbortSignal.timeout?.(60_000),
    });
  } catch {
    throw new ApiUnavailableError();
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = (body as { error?: { code?: string; message?: string } }).error;
    if (error?.code === "quota_exceeded") throw new QuotaError(error?.message || "Limit tugadi.");
    throw new Error(error?.message || `HTTP ${response.status}`);
  }
  return body as T;
}

export const api = {
  health: () => request<HealthReport>("/api/v1/health"),

  chat(conversationId: string | null, message: string, language = "uz") {
    return request<ChatResponse>("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ conversation_id: conversationId, message, language }),
    });
  },

  listConversations() {
    return request<{ count: number; conversations: ConversationSummary[] }>(
      "/api/v1/conversations"
    );
  },

  createConversation(title = "") {
    return request<{ conversation: ConversationSummary }>("/api/v1/conversations", {
      method: "POST",
      body: JSON.stringify({ title }),
    });
  },

  getConversation(id: string) {
    return request<{ conversation: ConversationSummary; messages: StoredMessage[] }>(
      `/api/v1/conversations/${id}`
    );
  },

  deleteConversation(id: string) {
    return request<{ deleted: boolean }>(`/api/v1/conversations/${id}`, { method: "DELETE" });
  },
};

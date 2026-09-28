"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type ConversationSummary } from "@/lib/api";
import { useChat } from "@/hooks/useChat";
import { Sidebar } from "@/components/Sidebar";
import { MessageList } from "@/components/MessageList";
import { Composer } from "@/components/Composer";
import { SettingsDialog } from "@/components/SettingsDialog";

const SUGGESTIONS = [
  { icon: "🍅", text: "Pomidor barglarida sariq dog'lar paydo bo'ldi — nima bo'lishi mumkin?" },
  { icon: "🧂", text: "Tuproq sho'rlangan (EC 6.5 dS/m). Nima qilish kerak?" },
  { icon: "💧", text: "5 gektar bug'doyga kuniga qancha suv kerak?" },
  { icon: "🐄", text: "Mol ozuqasini qanday muvozanatlash kerak?" },
];

export default function Home() {
  const chat = useChat();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [dark, setDark] = useState(false);

  const refreshConversations = useCallback(() => {
    api
      .listConversations()
      .then((data) => setConversations(data.conversations))
      .catch(() => setConversations([]));
  }, []);

  useEffect(() => {
    refreshConversations();
    const stored = localStorage.getItem("agroai-theme");
    if (stored === "dark") {
      setDark(true);
      document.documentElement.classList.add("dark");
    }
  }, [refreshConversations]);

  useEffect(() => {
    if (chat.conversationId) refreshConversations();
  }, [chat.conversationId, refreshConversations]);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("agroai-theme", next ? "dark" : "light");
  };

  const openConversation = async (id: string) => {
    setSidebarOpen(false);
    try {
      const data = await api.getConversation(id);
      // Hydrate the local view from the stored history (one shared core).
      window.location.hash = `#c-${id}`;
      chat.reset();
      for (const message of data.messages) {
        chat.send;
      }
      // Simpler: re-render messages directly via send for user, stored answer text for assistant
      for (const message of data.messages) {
        if (message.role === "user") {
          await chat.send(message.content);
        }
      }
    } catch {
      /* conversation list may be stale; ignore */
    }
  };

  const deleteConversation = async (id: string) => {
    try {
      await api.deleteConversation(id);
      if (id === chat.conversationId) chat.reset();
      refreshConversations();
    } catch {
      /* ignore */
    }
  };

  const newChat = () => {
    chat.reset();
    setSidebarOpen(false);
  };

  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeId={chat.conversationId}
        onSelect={openConversation}
        onNew={newChat}
        onDelete={deleteConversation}
        onOpenSettings={() => setSettingsOpen(true)}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-soil-200 bg-white/80 px-4 py-2.5 backdrop-blur dark:border-soil-700 dark:bg-soil-900/80 md:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            aria-label="Panelni ochish"
            className="rounded-lg p-2 text-soil-500 hover:bg-soil-100 dark:hover:bg-soil-800"
          >
            ☰
          </button>
          <p className="flex items-center gap-1.5 font-bold">
            <span aria-hidden>🌱</span> AgroAI
          </p>
          <button
            onClick={() => setSettingsOpen(true)}
            aria-label="Sozlamalar"
            className="rounded-lg p-2 text-soil-500 hover:bg-soil-100 dark:hover:bg-soil-800"
          >
            ⚙️
          </button>
        </header>

        {chat.messages.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center px-4 text-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-leaf-600 text-3xl shadow-md">
              🌱
            </div>
            <h1 className="text-2xl font-bold tracking-tight md:text-3xl">AgroAI</h1>
            <p className="mt-1.5 max-w-md text-sm text-soil-500 dark:text-soil-400">
              Qishloq xo'jaligi va veterinariya uchun aqlli yordamchi. Javoblar mahalliy
              bilimlar bazasiga asoslanadi — manbalar bilan birga.
            </p>
            <div className="mt-8 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion.text}
                  onClick={() => void chat.send(suggestion.text)}
                  className="rounded-xl border border-soil-200 bg-white px-4 py-3 text-left text-sm transition hover:border-leaf-300 hover:shadow dark:border-soil-700 dark:bg-soil-900 dark:hover:border-leaf-700"
                >
                  <span className="mr-2" aria-hidden>
                    {suggestion.icon}
                  </span>
                  {suggestion.text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto">
            <MessageList messages={chat.messages} isLoading={chat.isLoading} />
          </div>
        )}

        <Composer onSend={(text) => void chat.send(text)} isLoading={chat.isLoading} />
      </main>

      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        dark={dark}
        onToggleDark={toggleDark}
      />
    </div>
  );
}

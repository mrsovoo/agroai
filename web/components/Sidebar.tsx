"use client";

import type { ConversationSummary } from "@/lib/api";

function groupByDate(conversations: ConversationSummary[]) {
  const now = Date.now();
  const dayMs = 86_400_000;
  const groups: { label: string; items: ConversationSummary[] }[] = [
    { label: "Bugun", items: [] },
    { label: "Oldingi", items: [] },
  ];
  for (const conversation of conversations) {
    const created = new Date(conversation.updated_at).getTime();
    const isToday = now - created < dayMs;
    (isToday ? groups[0] : groups[1]).items.push(conversation);
  }
  return groups.filter((group) => group.items.length > 0);
}

export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onOpenSettings,
  open,
  onClose,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
  open: boolean;
  onClose: () => void;
}) {
  const groups = groupByDate(conversations);
  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={onClose}
          aria-hidden
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-30 flex w-72 flex-col border-r border-soil-200 bg-white transition-transform dark:border-soil-700 dark:bg-soil-900 md:static md:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
        aria-label="Suhbatlar paneli"
      >
        <div className="flex items-center justify-between px-4 pt-5 pb-2">
          <div className="flex items-center gap-2">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-leaf-600 text-lg text-white shadow-sm">
              🌱
            </span>
            <div>
              <p className="text-base font-bold leading-none tracking-tight">AgroAI</p>
              <p className="mt-0.5 text-[11px] text-soil-500 dark:text-soil-400">
                agro · veterinary yordamchi
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-soil-500 hover:bg-soil-100 dark:hover:bg-soil-800 md:hidden"
            aria-label="Panelni yopish"
          >
            ✕
          </button>
        </div>

        <div className="px-3 pb-2">
          <button
            onClick={onNew}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-leaf-200 bg-leaf-50 px-3 py-2.5 text-sm font-semibold text-leaf-800 transition hover:bg-leaf-100 dark:border-leaf-800 dark:bg-leaf-950 dark:text-leaf-200 dark:hover:bg-leaf-900"
          >
            <span aria-hidden>＋</span> Yangi suhbat
          </button>
        </div>

        <nav className="flex-1 space-y-4 overflow-y-auto px-3 pb-4" aria-label="Suhbatlar tarixi">
          {groups.length === 0 && (
            <p className="px-2 pt-6 text-center text-xs text-soil-400">
              Hozircha suhbatlar yo'q.
              <br />
              Birinchi savolingizni yuboring 🌾
            </p>
          )}
          {groups.map((group) => (
            <div key={group.label}>
              <p className="px-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-soil-400 dark:text-soil-500">
                {group.label}
              </p>
              <ul className="space-y-0.5">
                {group.items.map((conversation) => (
                  <li key={conversation.id} className="group flex items-center">
                    <button
                      onClick={() => onSelect(conversation.id)}
                      className={`flex-1 truncate rounded-lg px-2.5 py-2 text-left text-sm transition ${
                        conversation.id === activeId
                          ? "bg-leaf-100 font-medium text-leaf-900 dark:bg-leaf-900 dark:text-leaf-100"
                          : "text-soil-700 hover:bg-soil-100 dark:text-soil-300 dark:hover:bg-soil-800"
                      }`}
                      title={conversation.title || "Suhbat"}
                    >
                      {conversation.title || "Suhbat"}
                    </button>
                    <button
                      onClick={() => onDelete(conversation.id)}
                      aria-label="Suhbatni o'chirish"
                      className="ml-1 hidden rounded-md p-1.5 text-soil-400 hover:bg-red-50 hover:text-red-500 group-hover:block dark:hover:bg-red-950"
                    >
                      🗑
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="space-y-1 border-t border-soil-200 p-3 dark:border-soil-700">
          <button
            onClick={onOpenSettings}
            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-soil-600 transition hover:bg-soil-100 dark:text-soil-300 dark:hover:bg-soil-800"
          >
            ⚙️ Sozlamalar
          </button>
          <a
            href="https://github.com/mrsovoo/agroai"
            target="_blank"
            rel="noopener noreferrer"
            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-soil-600 transition hover:bg-soil-100 dark:text-soil-300 dark:hover:bg-soil-800"
          >
            ℹ️ Loyiha haqida
          </a>
        </div>
      </aside>
    </>
  );
}

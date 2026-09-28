"use client";

import { useEffect, useState } from "react";
import { api, type HealthReport } from "@/lib/api";

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
        ok
          ? "bg-leaf-100 text-leaf-800 dark:bg-leaf-900 dark:text-leaf-200"
          : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200"
      }`}
    >
      {label}
    </span>
  );
}

export function SettingsDialog({
  open,
  onClose,
  dark,
  onToggleDark,
}: {
  open: boolean;
  onClose: () => void;
  dark: boolean;
  onToggleDark: () => void;
}) {
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setError(null);
    api
      .health()
      .then(setHealth)
      .catch((err: Error) => setError(err.message));
  }, [open]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Sozlamalar"
    >
      <div
        className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl dark:bg-soil-900"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-bold">⚙️ Sozlamalar</h2>
          <button
            onClick={onClose}
            aria-label="Yopish"
            className="rounded-lg p-1.5 text-soil-400 hover:bg-soil-100 dark:hover:bg-soil-800"
          >
            ✕
          </button>
        </div>

        <button
          onClick={onToggleDark}
          className="mb-4 flex w-full items-center justify-between rounded-xl border border-soil-200 px-3 py-2.5 text-sm dark:border-soil-700"
        >
          <span>Mavzu</span>
          <span className="font-semibold">{dark ? "🌙 Tungi" : "☀️ Kunduzgi"}</span>
        </button>

        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-soil-400">
          Backend holati (haqiqiy tekshiruv)
        </h3>
        {error && (
          <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
            ⚠️ {error}
          </p>
        )}
        {health && (
          <ul className="space-y-2 text-sm">
            <li className="flex items-center justify-between">
              <span>AgroAI v{health.version}</span>
              <StatusPill
                ok={health.status !== "error"}
                label={health.status === "ok" ? "OK" : health.status.toUpperCase()}
              />
            </li>
            <li className="flex items-center justify-between">
              <span>Retrieval (SQLite FTS5)</span>
              <StatusPill
                ok={!!health.retrieval?.ready}
                label={`${health.retrieval?.documents ?? 0} hujjat`}
              />
            </li>
            <li className="flex items-center justify-between">
              <span>Lokal AI model</span>
              <StatusPill
                ok={!!health.llm?.configured}
                label={health.llm?.configured ? "SOZLANGAN" : "SOZLANMAGAN"}
              />
            </li>
          </ul>
        )}
        <p className="mt-4 text-xs text-soil-400">
          Model sozlanmaganda ham AgroAI bilimlar bazasidan topilgan manbalar bilan javob beradi —
          hech qachon soxta javob bermaydi.
        </p>
      </div>
    </div>
  );
}

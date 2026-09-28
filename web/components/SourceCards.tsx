"use client";

import type { SourceItem } from "@/lib/api";

export function SourceCards({ sources }: { sources: SourceItem[] }) {
  if (!sources?.length) return null;
  return (
    <section aria-label="Manbalar" className="mt-3">
      <h4 className="mb-1.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-soil-500 dark:text-soil-400">
        <span aria-hidden>📚</span> Manbalar
      </h4>
      <ul className="flex flex-wrap gap-2">
        {sources.map((source) => (
          <li key={`${source.document_id}-${source.index}`}>
            {source.url ? (
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-lg border border-soil-200 bg-white px-3 py-2 text-xs shadow-sm transition hover:border-leaf-300 hover:shadow dark:border-soil-700 dark:bg-soil-900 dark:hover:border-leaf-700"
              >
                <span className="block font-semibold text-soil-800 dark:text-soil-100">
                  {source.title || source.source || "Manba"}
                </span>
                <span className="text-soil-500 dark:text-soil-400">{source.source}</span>
              </a>
            ) : (
              <div className="rounded-lg border border-soil-200 bg-white px-3 py-2 text-xs shadow-sm dark:border-soil-700 dark:bg-soil-900">
                <span className="block font-semibold text-soil-800 dark:text-soil-100">
                  {source.title || source.source || "Manba"}
                </span>
                <span className="text-soil-500 dark:text-soil-400">{source.source}</span>
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

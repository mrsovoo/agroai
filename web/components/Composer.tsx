"use client";

import { useRef, useState, type KeyboardEvent } from "react";

export function Composer({
  onSend,
  isLoading,
  disabled,
}: {
  onSend: (text: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const clean = value.trim();
    if (!clean || isLoading || disabled) return;
    onSend(clean);
    setValue("");
    ref.current?.focus();
  };

  const handleKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-soil-200 bg-white/80 backdrop-blur dark:border-soil-700 dark:bg-soil-900/80">
      <div className="mx-auto flex w-full max-w-3xl items-end gap-2 px-4 py-3">
        <label htmlFor="composer" className="sr-only">
          AgroAI ga savol yozing
        </label>
        <textarea
          id="composer"
          ref={ref}
          rows={1}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKey}
          placeholder="Savolingizni yozing… (Enter — yuborish, Shift+Enter — yangi qator)"
          disabled={disabled}
          className="max-h-40 min-h-[44px] flex-1 resize-none rounded-xl border border-soil-200 bg-soil-50 px-3 py-2.5 text-[15px] outline-none transition placeholder:text-soil-400 focus:border-leaf-400 focus:ring-2 focus:ring-leaf-200 disabled:opacity-50 dark:border-soil-700 dark:bg-soil-800 dark:focus:ring-leaf-900"
          style={{ height: "auto" }}
          onInput={(event) => {
            const el = event.currentTarget;
            el.style.height = "auto";
            el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
          }}
        />
        <button
          type="button"
          onClick={submit}
          disabled={isLoading || disabled || !value.trim()}
          aria-label="Yuborish"
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-leaf-600 text-white shadow-sm transition hover:bg-leaf-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-leaf-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          ↑
        </button>
      </div>
    </div>
  );
}

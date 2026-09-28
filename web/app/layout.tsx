import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AgroAI — Qishloq xo'jaligi va veterinariya yordamchisi",
  description:
    "Ochiq oflayn AI yordamchi: ekinlar, kasalliklar, tuproq, sug'orish, chorva va veterinariya savollari uchun manbaqa asoslangan javoblar.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="uz" suppressHydrationWarning>
      <body className="min-h-dvh bg-soil-50 font-sans text-soil-900 antialiased dark:bg-soil-950 dark:text-soil-100">
        {children}
      </body>
    </html>
  );
}

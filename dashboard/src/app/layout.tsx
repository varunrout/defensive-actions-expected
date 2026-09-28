import type { Metadata } from "next";
import "./globals.css";
import NavHeader from "@/components/NavHeader";

export const metadata: Metadata = {
  title: "Defensive Actions Expected",
  description: "DAx — a defensive-analytics case study across six evaluated models.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-full flex flex-col" style={{ background: "var(--bg)" }}>
        <NavHeader />
        <main className="flex-1 flex flex-col">{children}</main>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Newsreader } from "next/font/google";

import Nav from "@/components/Nav";
import "./globals.css";

// Three faces, each doing one job: Newsreader for anything meant to be read as prose
// (the question, the answer, headings), Inter for interface chrome, JetBrains Mono for
// anything numeric. A document tool that sets its answers in the same UI font as its
// buttons reads like a settings panel.
const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });

const serif = Newsreader({
  subsets: ["latin"],
  variable: "--font-serif",
  display: "swap",
  weight: ["400", "500", "600"],
});

const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "Document QA",
  description:
    "Ask questions about an indexed document corpus. Every answer cites the page it came from.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable} ${mono.variable}`}>
      <body>
        <Nav />
        {children}
      </body>
    </html>
  );
}

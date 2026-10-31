"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Ask" },
  { href: "/search", label: "Search" },
  { href: "/evaluation", label: "Evaluation" },
] as const;

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="topbar">
      <Link href="/" className="wordmark">
        Document QA
      </Link>
      <nav aria-label="Main" className="tabs">
        {LINKS.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className="tab"
            // Announced by screen readers; the background tint alone reaches sighted
            // users only.
            aria-current={pathname === href ? "page" : undefined}
          >
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}

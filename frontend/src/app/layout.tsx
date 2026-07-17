import type { Metadata } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";
import "../styles/tokens.css";
import "./globals.css";

// Variable names INTENTIONALLY kept as --font-inter / --font-jetbrains-mono so
// that tokens.css and the 30+ components consuming those CSS vars don't need to
// change. Semantically, Inter → Geist and JetBrains Mono → Geist Mono. A future
// rename PR can swap the variable names, but it's not launch-blocking.
const geist = Geist({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter"
});

const geistMono = Geist_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-jetbrains-mono"
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: ["400"],
  style: ["normal", "italic"],
  variable: "--font-serif"
});

export const metadata: Metadata = {
  title: "OpenResearch — verified research, not summaries",
  description:
    "Other tools summarize papers. OpenResearch re-runs them: it reads the literature, rebuilds the experiments, and runs them — returning a cited brief where every number is checked against a result it produced, and every contradiction is flagged."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable} ${instrumentSerif.variable}`}>
      <body className={geist.className}>{children}</body>
    </html>
  );
}

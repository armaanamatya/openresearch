import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        // Map design tokens (src/styles/tokens.css) into Tailwind utilities
        // so PaperBench (which uses Tailwind) consumes the same palette as
        // the CSS-module-driven lab UI.
        ink: "var(--ink)",
        "ink-2": "var(--ink-2)",
        muted: "var(--muted)",
        "muted-2": "var(--muted-2)",
        line: "var(--line)",
        "line-2": "var(--line-2)",
        chip: "var(--chip)",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        "accent-ink": "var(--accent-ink)",
        warn: "var(--warn)",
        "warn-soft": "var(--warn-soft)",
        "warn-ink": "var(--warn-ink)",
        err: "var(--err)",
        "err-soft": "var(--err-soft)"
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "ui-monospace", "monospace"]
      }
    }
  },
  plugins: []
};

export default config;

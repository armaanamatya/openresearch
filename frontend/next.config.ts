import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  outputFileTracingExcludes: {
    "/api/demo": [
      "build.log",
      "next.config.ts",
      "eslint.config.mjs",
      "postcss.config.mjs",
      "tailwind.config.ts",
      "tsconfig.json",
      "vitest.config.ts",
      "**/*.test.ts",
      "**/*.test.tsx"
    ],
    "/lab": [
      "build.log",
      "next.config.ts",
      "eslint.config.mjs",
      "postcss.config.mjs",
      "tailwind.config.ts",
      "tsconfig.json",
      "vitest.config.ts",
      "**/*.test.ts",
      "**/*.test.tsx"
    ]
  },
  turbopack: {
    root: import.meta.dirname
  }
};

export default nextConfig;

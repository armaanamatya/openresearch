"use client";

import { createContext, useContext, type ReactNode } from "react";

export type PresentationMode = "internal" | "demo";
const Ctx = createContext<PresentationMode>("internal");

export function PresentationModeProvider({
  mode,
  children
}: {
  mode: PresentationMode;
  children: ReactNode;
}) {
  return <Ctx.Provider value={mode}>{children}</Ctx.Provider>;
}

export function usePresentationMode(): PresentationMode {
  return useContext(Ctx);
}

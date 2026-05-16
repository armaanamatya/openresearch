"use client";

import { createContext, useContext, useMemo, type ReactNode } from "react";

import { layoutTopology, type Layout } from "./layout";
import type { PipelineTopology } from "./topology";

interface TopologyContextValue {
  topology: PipelineTopology;
  layout: Layout;
}

const Ctx = createContext<TopologyContextValue | null>(null);

/**
 * Wraps any subtree that needs the pipeline topology + its computed
 * layout. The layout is memoised on the topology reference so adding a
 * sibling consumer does NOT re-run Kahn's algorithm.
 */
export function TopologyProvider({
  topology,
  children
}: {
  topology: PipelineTopology;
  children: ReactNode;
}) {
  const layout = useMemo(() => layoutTopology(topology), [topology]);
  return <Ctx.Provider value={{ topology, layout }}>{children}</Ctx.Provider>;
}

export function useTopologyContext(): TopologyContextValue {
  const value = useContext(Ctx);
  if (!value) {
    throw new Error("useTopologyContext must be inside a <TopologyProvider>");
  }
  return value;
}

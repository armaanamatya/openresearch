"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RlmDashboardEvent } from "@/lib/events/rlm-events";

// Pacing bounds: a replay reveals events at their REAL inter-event timing (scaled
// by speed), but clamped so micro-bursts stay watchable and long idle gaps don't
// stall the playback.
const MIN_DELAY_MS = 120;
const MAX_DELAY_MS = 2500;
const DEFAULT_GAP_MS = 400; // used when an event lacks a parseable timestamp
const DEFAULT_SPEED = 1;

export interface ReplayDriverState {
  /** Number of events currently revealed (0..total). */
  index: number;
  total: number;
  playing: boolean;
  speed: number;
  atEnd: boolean;
}

export interface ReplayDriver {
  /** fullEvents.slice(0, index) — fed to <RlmLab events=…> (the same sink as live). */
  events: RlmDashboardEvent[];
  state: ReplayDriverState;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  seek: (index: number) => void;
  step: (delta: number) => void;
  setSpeed: (speed: number) => void;
  restart: () => void;
}

function eventTs(e: RlmDashboardEvent | undefined): number | null {
  if (!e) return null;
  const raw =
    (e as { timestamp?: string }).timestamp ??
    (e as { data?: { timestamp?: string } }).data?.timestamp;
  if (!raw) return null;
  const t = Date.parse(raw);
  return Number.isNaN(t) ? null : t;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Drives UI timeline replay of a completed run's persisted events. It owns only a
 * cursor (how many events are revealed); the revealed SLICE is fed to the SAME
 * RlmLab reducer that live SSE feeds, so the entire rendering pipeline is reused —
 * advancing the cursor "plays" the run, dragging it back "rewinds" (RlmLab resets
 * and re-folds when the array shrinks).
 *
 * Opens showing the full run (index = total), paused. Play restarts from 0 when at
 * the end, so the button always does something sensible.
 */
export function useReplayDriver(fullEvents: RlmDashboardEvent[]): ReplayDriver {
  const total = fullEvents.length;
  const [index, setIndex] = useState(total);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeedState] = useState(DEFAULT_SPEED);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  // Play loop: while playing and not at the end, reveal the next event after a
  // delay derived from the real timestamp gap (÷ speed, clamped).
  useEffect(() => {
    if (!playing || index >= total) {
      clearTimer();
      return;
    }
    const prevTs = eventTs(fullEvents[index - 1]);
    const nextTs = eventTs(fullEvents[index]);
    const rawGap = prevTs !== null && nextTs !== null ? nextTs - prevTs : DEFAULT_GAP_MS;
    const delay = clamp(rawGap / Math.max(speed, 0.1), MIN_DELAY_MS, MAX_DELAY_MS);
    timer.current = setTimeout(() => {
      setIndex((i) => Math.min(i + 1, total));
      // Reached the end on this step — stop. In the async timer callback (not the
      // effect body), so it does not trip the cascading-render lint rule.
      if (index + 1 >= total) setPlaying(false);
    }, delay);
    return clearTimer;
  }, [playing, index, speed, total, fullEvents, clearTimer]);

  const play = useCallback(() => {
    setIndex((i) => (i >= total ? 0 : i)); // restart from the beginning if at the end
    setPlaying(true);
  }, [total]);
  const pause = useCallback(() => setPlaying(false), []);
  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (p) return false;
      setIndex((i) => (i >= total ? 0 : i));
      return true;
    });
  }, [total]);
  const seek = useCallback(
    (i: number) => {
      setPlaying(false);
      setIndex(clamp(Math.round(i), 0, total));
    },
    [total],
  );
  const step = useCallback(
    (delta: number) => {
      setPlaying(false);
      setIndex((i) => clamp(i + delta, 0, total));
    },
    [total],
  );
  const setSpeed = useCallback((s: number) => setSpeedState(s), []);
  const restart = useCallback(() => {
    setIndex(0);
    setPlaying(true);
  }, []);

  const events = useMemo(() => fullEvents.slice(0, index), [fullEvents, index]);

  return {
    events,
    state: { index, total, playing, speed, atEnd: index >= total },
    play,
    pause,
    toggle,
    seek,
    step,
    setSpeed,
    restart,
  };
}

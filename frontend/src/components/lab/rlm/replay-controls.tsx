"use client";

import type { ReplayDriver } from "@/hooks/use-replay-driver";
import styles from "./replay-controls.module.css";

const SPEEDS = [0.5, 1, 2, 4, 8];

/**
 * Transport bar for UI timeline replay — play/pause, step, seek slider, speed, and a
 * position counter. Pure presenter: all state lives in the {@link ReplayDriver}.
 */
export function ReplayControls({ driver }: { driver: ReplayDriver }) {
  const { state, toggle, step, seek, setSpeed } = driver;
  const { index, total, playing, speed, atEnd } = state;

  const playLabel = playing ? "Pause" : atEnd ? "Replay from start" : "Play";

  return (
    <div className={styles.bar} role="group" aria-label="Replay controls">
      <span className={styles.badge}>REPLAY</span>

      <button type="button" className={styles.btn} onClick={toggle} aria-label={playLabel} title={playLabel}>
        {playing ? "❚❚" : atEnd ? "↺" : "▶"}
      </button>
      <button
        type="button"
        className={styles.btn}
        onClick={() => step(-1)}
        disabled={index <= 0}
        aria-label="Step back"
        title="Step back"
      >
        ⟪
      </button>
      <button
        type="button"
        className={styles.btn}
        onClick={() => step(1)}
        disabled={index >= total}
        aria-label="Step forward"
        title="Step forward"
      >
        ⟫
      </button>

      <input
        className={styles.slider}
        type="range"
        min={0}
        max={total}
        value={index}
        onChange={(e) => seek(Number(e.target.value))}
        aria-label="Timeline position"
      />

      <span className={styles.counter} aria-live="polite">
        {index} / {total}
      </span>

      <label className={styles.speed}>
        <span className={styles.srOnly}>Replay speed</span>
        <select
          className={styles.select}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          aria-label="Replay speed"
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

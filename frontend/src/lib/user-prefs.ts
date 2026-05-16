const KEY = "reprolab:user-prefs";

export interface UserPrefs {
  model?: "sonnet" | "opus";
  sandbox?: "auto" | "local" | "docker" | "runpod";
  executionMode?: "efficient" | "max";
  splitRatio?: number;
}

export function readUserPrefs(): UserPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as UserPrefs) : {};
  } catch {
    return {};
  }
}

export function writeUserPref<K extends keyof UserPrefs>(key: K, value: UserPrefs[K]): void {
  if (typeof window === "undefined") return;
  try {
    const prefs = readUserPrefs();
    prefs[key] = value;
    window.localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    // localStorage may be disabled (private mode) — non-fatal.
  }
}

/**
 * Stubbed "sign in" flag for the autonomous-reproduction flow (T11).
 *
 * There is no real authentication yet. The paper-landing screen
 * (`PaperLanding`) sets this flag when the user clicks "Sign in to start",
 * and the repo-confirm screen's launch cost-guard dialog (`RepoConfirm`)
 * requires it before it will actually launch a run. sessionStorage (not
 * localStorage) so it resets when the tab closes, the way a real sign-in
 * session eventually would.
 */

const DEV_SESSION_KEY = "openresearch:devSession:signedIn";

export function markDevSessionSignedIn(): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(DEV_SESSION_KEY, "1");
  } catch {
    // sessionStorage may be disabled (private mode etc.) — non-fatal.
  }
}

export function isDevSessionSignedIn(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.sessionStorage.getItem(DEV_SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

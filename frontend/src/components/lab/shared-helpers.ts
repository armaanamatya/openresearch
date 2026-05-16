/**
 * Small shared helpers used by multiple split lab components.
 * Kept tiny on purpose — when one of these grows, fold it into a
 * domain-specific file instead.
 */

export function issueText(value?: string | null): string {
  if (!value) return "";
  return value
    .replace(/\bfailed\b/gi, "needs attention")
    .replace(/\bfailure\b/gi, "issue");
}

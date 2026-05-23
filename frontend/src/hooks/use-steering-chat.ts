/**
 * useSteeringChat — derives chat messages from the SSE event stream and
 * provides a `send` function that POSTs to the messages proxy route.
 *
 * Messages are sourced from `user_message` and `user_message_response` events
 * in the existing SSE stream — no new stream or polling needed.
 *
 * Optimistic UI: when the user submits, an optimistic entry is appended
 * immediately. If the server echoes a matching `user_message` SSE event the
 * optimistic entry is replaced; if not (e.g., no backend yet) it stays until
 * a real event replaces it.
 */

import { useMemo, useState, useCallback, useRef } from "react";
import type { RlmDashboardEvent } from "../lib/events/rlm-events";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: string;
  /** True for a locally-added optimistic entry not yet confirmed by the server. */
  optimistic?: boolean;
}

export function useSteeringChat(
  projectId: string,
  events: RlmDashboardEvent[]
): {
  messages: ChatMessage[];
  send: (content: string) => Promise<void>;
  sending: boolean;
  error: string | null;
} {
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Optimistic messages keyed by a locally-generated id, removed when replaced.
  const [optimistic, setOptimistic] = useState<ChatMessage[]>([]);
  // Track content of confirmed server-echoed user messages so we can drop
  // the matching optimistic entry. We use the content string as the key because
  // the backend echo event has no client-side id.
  const confirmedContents = useRef<Set<string>>(new Set());

  // Derive confirmed messages from the event stream.
  const serverMessages = useMemo<ChatMessage[]>(() => {
    const out: ChatMessage[] = [];
    for (const ev of events) {
      if (ev.event === "user_message") {
        confirmedContents.current.add(ev.content);
        out.push({
          id: `server-user-${ev.timestamp}`,
          role: "user",
          content: ev.content,
          ts: ev.timestamp,
        });
      } else if (ev.event === "user_message_response") {
        out.push({
          id: `server-assistant-${ev.timestamp}`,
          role: "assistant",
          content: ev.message,
          ts: ev.timestamp,
        });
      }
    }
    return out;
  }, [events]);

  // Drop optimistic entries whose content has been confirmed by an SSE echo.
  const pendingOptimistic = useMemo(
    () => optimistic.filter((m) => !confirmedContents.current.has(m.content)),
    [optimistic]
  );

  // Merge: server messages first (in stream order), then any still-pending
  // optimistic messages appended at the end so they appear newest-last.
  const messages = useMemo<ChatMessage[]>(
    () => [...serverMessages, ...pendingOptimistic],
    [serverMessages, pendingOptimistic]
  );

  const send = useCallback(
    async (content: string) => {
      if (!content.trim() || sending) return;
      const optimisticId = `optimistic-${Date.now()}`;
      const optimisticMsg: ChatMessage = {
        id: optimisticId,
        role: "user",
        content,
        ts: new Date().toISOString(),
        optimistic: true,
      };
      setOptimistic((prev) => [...prev, optimisticMsg]);
      setSending(true);
      setError(null);
      try {
        const res = await fetch(
          `/api/demo/runs/${encodeURIComponent(projectId)}/messages`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ role: "user", content }),
          }
        );
        if (!res.ok) {
          const text = await res.text().catch(() => "");
          throw new Error(text || `HTTP ${res.status}`);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to send message");
        // Remove the failed optimistic entry so the user can retry.
        setOptimistic((prev) => prev.filter((m) => m.id !== optimisticId));
      } finally {
        setSending(false);
      }
    },
    [projectId, sending]
  );

  return { messages, send, sending, error };
}

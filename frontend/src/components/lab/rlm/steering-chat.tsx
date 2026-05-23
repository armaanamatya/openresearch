"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../../../hooks/use-steering-chat";
import styles from "./steering-chat.module.css";

export interface SteeringChatProps {
  projectId: string;
  messages: ChatMessage[];
  onSend: (content: string) => Promise<void>;
  disabled?: boolean;
}

/**
 * SteeringChat — the chat panel docked at the bottom of the NodeDetailSidebar.
 *
 * Optimistic UI is handled by the parent hook (useSteeringChat); this
 * component just renders the message log and the input form.
 */
export function SteeringChat({
  messages,
  onSend,
  disabled = false,
}: SteeringChatProps) {
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const logRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom whenever messages change.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const content = draft.trim();
    if (!content || sending || disabled) return;
    setDraft("");
    setSending(true);
    try {
      await onSend(content);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className={styles.chat} data-testid="steering-chat">
      {/* Message log */}
      <div className={styles.log} ref={logRef} role="log" aria-live="polite">
        {messages.length === 0 ? (
          <p className={styles.empty}>ask the RLM a question or steer it</p>
        ) : (
          messages.map((m) => (
            <div
              key={m.id}
              className={
                m.role === "user" ? styles.msgUser : styles.msgAssistant
              }
              aria-label={m.role === "user" ? "you" : "RLM"}
              data-optimistic={m.optimistic ? "true" : undefined}
            >
              <span className={styles.msgRole}>
                {m.role === "user" ? "you" : "RLM"}
              </span>
              <span className={styles.msgContent}>{m.content}</span>
            </div>
          ))
        )}
      </div>

      {/* Input */}
      <form className={styles.inputRow} onSubmit={handleSubmit}>
        <input
          className={styles.input}
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Send a message…"
          disabled={disabled || sending}
          aria-label="Chat message"
        />
        <button
          type="submit"
          className={styles.sendBtn}
          disabled={disabled || sending || !draft.trim()}
          aria-label={sending ? "Sending…" : "Send"}
        >
          {sending ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}

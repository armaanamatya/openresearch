import { ChevronLeft, Plus } from "lucide-react";
import styles from "./SessionRail.module.css";

export interface SessionRailSession {
  id: string;
  title: string;
  active?: boolean;
}

interface SessionRailProps {
  projectName: string;
  sessions: SessionRailSession[];
  onNewAgent?: () => void;
  /** Not in the alphaXiv screen's literal prop list but a session row with
   * no way to switch to it is a dead end for T13 (SessionReasoningView) —
   * optional, no-op if omitted, mirrors PrimaryButton's optional onClick. */
  onSelectSession?: (id: string) => void;
  className?: string;
}

/** Left session rail (alphaXiv screen `_5`): a project header, a "New
 * agent" control, and a session list where the active session carries a
 * live dot. Pure presentational — T13 owns wiring real session data in. */
export function SessionRail({
  projectName,
  sessions,
  onNewAgent,
  onSelectSession,
  className,
}: SessionRailProps) {
  const classes = [styles.rail, className].filter(Boolean).join(" ");

  return (
    <nav className={classes} aria-label={`${projectName} sessions`}>
      <div className={styles.header}>
        <ChevronLeft size={16} aria-hidden="true" className={styles.backGlyph} />
        <span className={styles.projectName}>{projectName}</span>
      </div>
      <button type="button" className={styles.newAgent} onClick={onNewAgent}>
        <Plus size={14} aria-hidden="true" />
        <span>New agent</span>
      </button>
      <ul className={styles.sessions}>
        {sessions.map((session) => {
          const rowClasses = [styles.session, session.active ? styles.active : ""]
            .filter(Boolean)
            .join(" ");

          return (
            <li key={session.id}>
              <button
                type="button"
                className={rowClasses}
                data-active={session.active ? "true" : "false"}
                onClick={() => onSelectSession?.(session.id)}
              >
                <span className={styles.sessionTitle}>{session.title}</span>
                {session.active && (
                  <span className={styles.liveDot} aria-hidden="true" />
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

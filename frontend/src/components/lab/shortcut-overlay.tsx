"use client";

import "./shortcut-overlay.css";

type ShortcutOverlayProps = {
  open: boolean;
  setOpen: (open: boolean) => void;
};

const ROWS: Array<{ keys: string[]; desc: string }> = [
  { keys: ["Cmd/Ctrl", "K"], desc: "Command palette" },
  { keys: ["j", "↓", "→"], desc: "Next node" },
  { keys: ["k", "↑", "←"], desc: "Previous node" },
  { keys: ["Enter"], desc: "Inspect selected node" },
  { keys: ["Esc"], desc: "Close panels / overlays" },
  { keys: ["?"], desc: "Toggle this help" }
];

export function ShortcutOverlay({ open, setOpen }: ShortcutOverlayProps) {
  if (!open) return null;

  const close = () => setOpen(false);

  return (
    <div
      className="shortcut-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) close();
      }}
      role="presentation"
    >
      <div
        className="shortcut-dialog"
        role="dialog"
        aria-label="Keyboard shortcuts"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="shortcut-head">
          <span className="shortcut-title">Keyboard shortcuts</span>
          <button
            type="button"
            className="shortcut-close"
            onClick={close}
            aria-label="Close shortcuts"
          >
            ×
          </button>
        </div>
        <div className="shortcut-body">
          {ROWS.map((row) => (
            <div key={row.desc} className="shortcut-row">
              <span className="shortcut-keys">
                {row.keys.map((k, i) => (
                  <span key={`${row.desc}-${i}`} className="shortcut-key">
                    {k}
                  </span>
                ))}
              </span>
              <span className="shortcut-desc">{row.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

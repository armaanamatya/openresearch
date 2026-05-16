import type { ReactNode } from "react";

function icon(children: ReactNode, size = 18) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 18 18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export const ICONS = {
  logo: (
    <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
      <path
        d="M4 6.5L11 3l7 3.5M4 6.5v9L11 19l7-3.5v-9M4 6.5L11 10l7-3.5M11 10v9"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
  lab: icon(
    <>
      <path d="M7 2.5v4L3.5 12a1.5 1.5 0 0 0 1.3 2.5h8.4A1.5 1.5 0 0 0 14.5 12L11 6.5v-4" />
      <path d="M6.5 2.5h5" />
    </>
  ),
  papers: icon(
    <>
      <path d="M5 2.5h6l3 3v10H5z" />
      <path d="M11 2.5v3h3" />
      <path d="M7 9h6M7 12h4" />
    </>
  ),
  hermes: icon(
    <>
      <path d="M9 2l5.5 2v5c0 3.5-2.5 6.5-5.5 7.5C6 15.5 3.5 12.5 3.5 9V4z" />
      <path d="M6.5 9l2 2 3-4" />
    </>
  ),
  feedback: icon(<path d="M3 4h12v8H8l-3 3v-3H3z" />),
  help: icon(
    <>
      <circle cx="9" cy="9" r="6.5" />
      <path d="M7.5 7c.4-1 1.4-1.5 2.4-1.2 1 .3 1.6 1.4 1.2 2.4-.3.7-1.1 1.3-1.6 1.3v.8" />
      <circle cx="9" cy="13" r=".6" fill="currentColor" />
    </>
  ),
  settings: icon(
    <>
      <circle cx="9" cy="9" r="2" />
      <path d="M14.5 9c0 .4 0 .8-.1 1.1l1.4 1-1.6 2.7-1.7-.5c-.5.5-1.1.9-1.7 1.1L10.5 16h-3l-.3-1.6c-.6-.2-1.2-.6-1.7-1.1l-1.7.5L2.2 11l1.4-1c-.1-.3-.1-.7-.1-1.1s0-.8.1-1.1l-1.4-1L3.8 4l1.7.5c.5-.5 1.1-.9 1.7-1.1L7.5 2h3l.3 1.6c.6.2 1.2.6 1.7 1.1L14.2 4l1.6 2.7-1.4 1c.1.4.1.8.1 1.2z" />
    </>
  ),
  upload: icon(
    <>
      <path d="M9 11V3.5M9 3.5l-2.5 2.5M9 3.5l2.5 2.5" />
      <path d="M3.5 12v1.5A1.5 1.5 0 0 0 5 15h8a1.5 1.5 0 0 0 1.5-1.5V12" />
    </>
  ),
  play: icon(<path d="M5 3.5v11l9-5.5z" fill="currentColor" stroke="none" />),
  pause: icon(
    <>
      <rect x="5.5" y="4" width="2.2" height="10" rx="1" fill="currentColor" stroke="none" />
      <rect x="10.3" y="4" width="2.2" height="10" rx="1" fill="currentColor" stroke="none" />
    </>
  ),
  spark: icon(
    <>
      <path d="M9 2v3M9 13v3M2 9h3M13 9h3M4 4l2 2M12 12l2 2M4 14l2-2M12 6l2-2" />
    </>
  ),
  doc: icon(
    <>
      <path d="M5 2.5h6l3 3v10H5z" />
      <path d="M11 2.5v3h3" />
    </>
  ),
  brain: icon(
    <>
      <path d="M9 3.5a2.5 2.5 0 0 0-2.5 2.5v0a2 2 0 0 0-1 3.5 2 2 0 0 0 1 3.5v0A2.5 2.5 0 0 0 9 15.5" />
      <path d="M9 3.5a2.5 2.5 0 0 1 2.5 2.5v0a2 2 0 0 1 1 3.5 2 2 0 0 1-1 3.5v0a2.5 2.5 0 0 1-2.5 2.5" />
    </>
  ),
  beaker: icon(
    <>
      <path d="M7 2.5v4L3.5 12a1.5 1.5 0 0 0 1.3 2.5h8.4A1.5 1.5 0 0 0 14.5 12L11 6.5v-4" />
      <path d="M6.5 2.5h5" />
      <circle cx="9" cy="11" r=".7" fill="currentColor" />
      <circle cx="7" cy="9" r=".5" fill="currentColor" />
    </>
  ),
  shield: icon(
    <>
      <path d="M9 2l5.5 2v5c0 3.5-2.5 6.5-5.5 7.5C6 15.5 3.5 12.5 3.5 9V4z" />
    </>
  ),
  zap: icon(<path d="M10 2L4.5 10h3l-1 6 5.5-8h-3l1-6z" fill="currentColor" stroke="none" />),
  copy: icon(
    <>
      <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" />
      <path d="M3.5 11V4A1.5 1.5 0 0 1 5 2.5h7" />
    </>
  ),
  graph: icon(
    <>
      <path d="M2.5 14.5l4-5 3 2 6-7" />
      <path d="M10 4.5h5.5V10" />
    </>
  ),
  flag: icon(
    <>
      <path d="M4 2v14" />
      <path d="M4 3h9l-2 3 2 3H4" />
    </>
  ),
  compute: icon(
    <>
      <rect x="3" y="3" width="12" height="12" rx="2" />
      <rect x="6" y="6" width="6" height="6" />
      <path d="M3 6.5h-1M3 11.5h-1M16 6.5h-1M16 11.5h-1M6.5 3v-1M11.5 3v-1M6.5 16v-1M11.5 16v-1" />
    </>
  )
};

export type IconKey = keyof typeof ICONS;

/**
 * The hero exploration-tree figure — pure SVG, no interactivity.
 * Sourced from docs/design/landing-source/index.html lines 762-858.
 * Style classes (.nlbl, .edge, .node-bg, .pulse) are defined inside
 * the SVG <defs> so they only apply to this SVG and don't pollute
 * the document.
 */
export function HeroTree(): React.JSX.Element {
  return (
    <svg viewBox="0 0 1200 360" width="100%" preserveAspectRatio="xMidYMid meet" aria-hidden>
      <defs>
        <style>{`
          .nlbl { font-family: var(--font-jetbrains-mono), ui-monospace, monospace; font-size: 10px; fill: oklch(0.56 0.005 90); letter-spacing: 0.04em; }
          .nlbl-em { fill: oklch(0.96 0.005 90); }
          .nlbl-accent { fill: oklch(0.80 0.16 70); }
          .edge { stroke: oklch(0.28 0.004 270); stroke-width: 1; fill: none; }
          .edge-live { stroke: oklch(0.80 0.16 70 / .8); }
          .node-bg { fill: oklch(0.205 0.004 270); stroke: oklch(0.28 0.004 270); }
          .node-bg-live { fill: oklch(0.205 0.004 270); stroke: oklch(0.80 0.16 70 / .9); }
          .node-bg-done { fill: oklch(0.205 0.004 270); stroke: oklch(0.40 0.005 90); }
          .pulse { animation: pulse 2.2s ease-out infinite; }
          @keyframes pulse { 0% { r: 4; opacity: 1; } 100% { r: 14; opacity: 0; } }
        `}</style>
      </defs>

      {/* edges */}
      <path className="edge" d="M150,180 C 230,180 230,80 310,80"/>
      <path className="edge" d="M150,180 C 230,180 230,180 310,180"/>
      <path className="edge" d="M150,180 C 230,180 230,280 310,280"/>

      <path className="edge" d="M450,80  C 530,80  530,40  610,40"/>
      <path className="edge" d="M450,80  C 530,80  530,120 610,120"/>
      <path className="edge edge-live" d="M450,180 C 530,180 530,180 610,180"/>
      <path className="edge" d="M450,280 C 530,280 530,240 610,240"/>
      <path className="edge" d="M450,280 C 530,280 530,320 610,320"/>

      <path className="edge edge-live" d="M750,180 C 830,180 830,140 910,140"/>
      <path className="edge edge-live" d="M750,180 C 830,180 830,220 910,220"/>

      {/* nodes */}
      <g>
        <rect className="node-bg-done" x="60" y="160" width="180" height="44" rx="5"/>
        <text className="nlbl" x="78" y="180">§ 1.0  COMPREHENSION</text>
        <text className="nlbl nlbl-em" x="78" y="196">Paper parsed · 47 claims</text>
      </g>

      <g>
        <rect className="node-bg-done" x="310" y="60" width="140" height="40" rx="5"/>
        <text className="nlbl" x="324" y="78">§ 2.0 ENV · CUDA</text>
        <text className="nlbl nlbl-em" x="324" y="92">torch 2.4 · py3.11</text>
      </g>
      <g>
        <rect className="node-bg-done" x="310" y="160" width="140" height="40" rx="5"/>
        <text className="nlbl" x="324" y="178">§ 3.0 IMPL · baseline</text>
        <text className="nlbl nlbl-em" x="324" y="192">commit 7e3a · diff +428</text>
      </g>
      <g>
        <rect className="node-bg-done" x="310" y="260" width="140" height="40" rx="5"/>
        <text className="nlbl" x="324" y="278">§ 3.2 EXPLORE</text>
        <text className="nlbl nlbl-em" x="324" y="292">12 branches</text>
      </g>

      {/* branches */}
      <g>
        <rect className="node-bg" x="610" y="20" width="140" height="40" rx="5"/>
        <text className="nlbl" x="624" y="38">3.2a · lr ×2</text>
        <text className="nlbl" x="624" y="52">acc 0.611 → 0.624</text>
      </g>
      <g>
        <rect className="node-bg" x="610" y="100" width="140" height="40" rx="5"/>
        <text className="nlbl" x="624" y="118">3.2b · bs 256</text>
        <text className="nlbl" x="624" y="132">acc 0.598</text>
      </g>
      <g>
        <rect className="node-bg-live" x="610" y="160" width="140" height="40" rx="5"/>
        <text className="nlbl" x="624" y="178">§ 4.0 EXPERIMENTS</text>
        <text className="nlbl nlbl-accent" x="624" y="192">running · seed 0…4</text>
        <circle cx="740" cy="180" r="4" fill="oklch(0.80 0.16 70)"/>
        <circle cx="740" cy="180" r="4" fill="oklch(0.80 0.16 70)" className="pulse"/>
      </g>
      <g>
        <rect className="node-bg" x="610" y="220" width="140" height="40" rx="5"/>
        <text className="nlbl" x="624" y="238">3.2c · adamw → lion</text>
        <text className="nlbl" x="624" y="252">acc 0.582</text>
      </g>
      <g>
        <rect className="node-bg" x="610" y="300" width="140" height="40" rx="5"/>
        <text className="nlbl" x="624" y="318">3.2d · cosine sched</text>
        <text className="nlbl" x="624" y="332">acc 0.607</text>
      </g>

      {/* verify */}
      <g>
        <rect className="node-bg-live" x="910" y="120" width="160" height="40" rx="5"/>
        <text className="nlbl" x="924" y="138">§ 5.0 VERIFY · seed 0</text>
        <text className="nlbl nlbl-accent" x="924" y="152">0.627 vs 0.631 paper</text>
      </g>
      <g>
        <rect className="node-bg-live" x="910" y="200" width="160" height="40" rx="5"/>
        <text className="nlbl" x="924" y="218">§ 5.0 VERIFY · seed 1</text>
        <text className="nlbl nlbl-accent" x="924" y="232">0.623 vs 0.631 paper</text>
      </g>
    </svg>
  );
}

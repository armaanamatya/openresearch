import type { Citation } from "../../lib/dashboard/contracts";

export function CitationPanel({
  citations,
  heading
}: {
  citations: Citation[];
  heading: string;
}) {
  return (
    <section className="panel">
      <div className="panel-inner">
        <h2 className="section-title">Citation panel</h2>
        <p className="section-copy">
          Every decision surface in this shell is wired to citations first. If the selected
          item has no direct citations, the panel falls back to task-level evidence.
        </p>
        <div className="detail-card" style={{ marginTop: 16 }}>
          <strong>Selected context</strong>
          <p className="section-copy" style={{ marginTop: 10, marginBottom: 0 }}>
            {heading}
          </p>
        </div>
        <ul className="citation-list" style={{ marginTop: 18 }}>
          {citations.length ? (
            citations.map((citation) => (
              <li key={citation.source_id} className="citation-card">
                <div className="chip-row">
                  <span className="pill">{citation.source_id}</span>
                  <span className="trust-badge">{citation.trust_level}</span>
                  <span className="pill">{citation.source_kind}</span>
                </div>
                <p className="section-copy" style={{ marginTop: 12, marginBottom: 0 }}>
                  {citation.content}
                </p>
              </li>
            ))
          ) : (
            <li className="citation-card">
              <p className="section-copy" style={{ marginBottom: 0 }}>
                No citations are attached yet. When the real event stream lands, uncited
                decisions should be shown as warnings rather than hidden.
              </p>
            </li>
          )}
        </ul>
      </div>
    </section>
  );
}

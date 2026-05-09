import type { FeedItem } from "../../lib/dashboard/normalize";

export function ActivityFeed({
  items,
  selectedFeedId,
  onSelectFeed
}: {
  items: FeedItem[];
  selectedFeedId: string;
  onSelectFeed: (feedId: string, taskId: string) => void;
}) {
  return (
    <section className="panel">
      <div className="panel-inner">
        <h2 className="section-title">Activity and message feed</h2>
        <p className="section-copy">
          The feed keeps lifecycle events, reasoning steps, shared-state publications, and
          verification decisions in one chronological stream.
        </p>
        <ul className="feed-list" style={{ marginTop: 18 }}>
          {items.map((item) => (
            <li key={item.id} className="feed-item">
              <button
                className="feed-button"
                data-selected={item.id === selectedFeedId}
                onClick={() => onSelectFeed(item.id, item.taskId)}
                type="button"
              >
                <div className="chip-row">
                  <span className="pill">{item.event}</span>
                  <span className="pill">{item.agentId}</span>
                </div>
                <div className="feed-title" style={{ marginTop: 12 }}>
                  {item.title}
                </div>
                <p className="section-copy" style={{ marginBottom: 0 }}>
                  {item.summary}
                </p>
                <div className="feed-meta">
                  <span>{item.timestamp}</span>
                  <span>Task {item.taskId}</span>
                  <span>{item.citations.length} citations</span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

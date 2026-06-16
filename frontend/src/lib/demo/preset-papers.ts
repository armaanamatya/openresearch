// The default reproduction targets, bundled in-repo (papers/registry.json) so they
// are available + selectable on a fresh clone with no network fetch. Selecting a chip
// fills the arXiv id; the backend paper-registry resolves it to the in-repo PDF and
// auto-applies the paper's hint (SDAR's future-dated id only works because it's bundled).
// Mirror of papers/registry.json — keep in sync (or fetch GET /papers for the dynamic list).
export const PRESET_PAPERS: ReadonlyArray<{ id: string; arxivId: string; title: string; short: string }> = [
  { id: "sdar",   arxivId: "2605.15155", title: "Self-Distilled Agentic Reinforcement Learning",     short: "SDAR (bundled)" },
  { id: "adam",   arxivId: "1412.6980",  title: "Adam: A Method for Stochastic Optimization",         short: "Adam (bundled)" },
  { id: "allcnn", arxivId: "1412.6806",  title: "Striving for Simplicity: The All Convolutional Net", short: "All-CNN (bundled)" },
  { id: "omnizip", arxivId: "2511.14582", title: "OmniZip: Audio-Guided Dynamic Token Compression",      short: "OmniZip (bundled)" },
];

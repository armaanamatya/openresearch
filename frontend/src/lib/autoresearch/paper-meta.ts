/**
 * Shared placeholder copy for the paper-landing / repo-confirm screens.
 *
 * There is no backend endpoint yet for arbitrary-arXiv paper metadata
 * (title, authors, abstract) ahead of ingestion — the arXiv id is the only
 * reliable datum available before a run starts. Both screens fall back to
 * this honest, paper-agnostic heading instead of fabricating a title; a
 * `title` prop lets either screen show the real one once a metadata source
 * exists, with no further changes needed here.
 */
export const FALLBACK_PAPER_TITLE = "Reproduce this paper";

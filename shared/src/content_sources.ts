/**
 * A first-page crop cannot represent a question continued on another page.
 * Decide the source set before loading images so an existing crop never hides
 * later working or the teacher's mark on a continuation.
 */
export function planContentSources(
  spans: readonly { page: number }[],
  cropAvailable: boolean,
): { kind: "crop" | "pages"; pageNumbers: number[] } {
  const pageNumbers = [...new Set(spans.map((span) => span.page))];
  return {
    kind: cropAvailable && pageNumbers.length === 1 ? "crop" : "pages",
    pageNumbers,
  };
}

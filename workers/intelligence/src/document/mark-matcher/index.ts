import type { Box } from "../../schemas";

export interface MarkCandidate { id: string; pageId: string; box: Box; commentRegionId?: string }
export interface QuestionCandidate { id: string; pageIds: string[]; box: Box; order: number; continuationPageIds?: string[]; commentRegionIds?: string[] }
export interface MarkAssignment { markId: string; questionId: string | null; confidence: number; secondBestGap: number; features: Record<string, number> }

function center(box: Box): { x: number; y: number } { return { x: box.x + box.width / 2, y: box.y + box.height / 2 }; }
function cost(mark: MarkCandidate, question: QuestionCandidate): { total: number; features: Record<string, number> } {
  const m = center(mark.box); const q = center(question.box);
  const vertical = Math.abs(m.y - q.y); const horizontal = Math.abs(m.x - q.x);
  const samePage = question.pageIds.includes(mark.pageId) ? 1 : 0;
  const continuation = question.continuationPageIds?.includes(mark.pageId) ? 1 : 0;
  const comment = mark.commentRegionId && question.commentRegionIds?.includes(mark.commentRegionId) ? 1 : 0;
  const margin = mark.box.x > 0.75 || mark.box.x < 0.1 ? 1 : 0;
  const features = { vertical, horizontal, samePage, continuation, comment, margin, order: question.order };
  return { total: vertical * 4 + horizontal * 0.8 + (samePage ? 0 : 3) - continuation * 1.2 - comment * 2 - margin * 0.15, features };
}

export function matchMarks(marks: readonly MarkCandidate[], questions: readonly QuestionCandidate[]): MarkAssignment[] {
  if (marks.length === 0) return [];
  if (questions.length === 0) return marks.map((mark) => ({ markId: mark.id, questionId: null, confidence: 0, secondBestGap: 0, features: {} }));
  const columns = Math.max(marks.length, questions.length);
  const matrix = marks.map((mark) => Array.from({ length: columns }, (_, index) => index < questions.length ? cost(mark, questions[index]).total : 4.5));
  // Hungarian algorithm: minimum-cost global one-to-one assignment, including dummy
  // columns for marks that cannot be attributed safely.
  const u = new Array<number>(marks.length + 1).fill(0);
  const v = new Array<number>(columns + 1).fill(0);
  const p = new Array<number>(columns + 1).fill(0);
  const way = new Array<number>(columns + 1).fill(0);
  for (let row = 1; row <= marks.length; row += 1) {
    p[0] = row;
    let column = 0;
    const min = new Array<number>(columns + 1).fill(Number.POSITIVE_INFINITY);
    const used = new Array<boolean>(columns + 1).fill(false);
    do {
      used[column] = true;
      const currentRow = p[column] ?? 0;
      let delta = Number.POSITIVE_INFINITY;
      let next = 0;
      for (let candidate = 1; candidate <= columns; candidate += 1) {
        if (used[candidate]) continue;
        const reduced = (matrix[currentRow - 1]?.[candidate - 1] ?? 4.5) - (u[currentRow] ?? 0) - (v[candidate] ?? 0);
        if (reduced < (min[candidate] ?? Number.POSITIVE_INFINITY)) { min[candidate] = reduced; way[candidate] = column; }
        if ((min[candidate] ?? Number.POSITIVE_INFINITY) < delta) { delta = min[candidate] ?? delta; next = candidate; }
      }
      for (let candidate = 0; candidate <= columns; candidate += 1) {
        if (used[candidate]) { const assigned = p[candidate] ?? 0; u[assigned] = (u[assigned] ?? 0) + delta; v[candidate] = (v[candidate] ?? 0) - delta; }
        else min[candidate] = (min[candidate] ?? 0) - delta;
      }
      column = next;
    } while ((p[column] ?? 0) !== 0);
    do {
      const previous = way[column] ?? 0;
      p[column] = p[previous] ?? 0;
      column = previous;
    } while (column !== 0);
  }
  const assignedColumn = new Array<number>(marks.length).fill(-1);
  for (let column = 1; column <= columns; column += 1) {
    const row = p[column] ?? 0;
    if (row > 0) assignedColumn[row - 1] = column - 1;
  }
  return marks.map((mark, index) => {
    const questionIndex = assignedColumn[index] ?? -1;
    if (questionIndex < 0 || questionIndex >= questions.length || (matrix[index]?.[questionIndex] ?? 4.5) >= 4.5) {
      return { markId: mark.id, questionId: null, confidence: 0, secondBestGap: 0, features: {} };
    }
    const question = questions[questionIndex];
    const selected = cost(mark, question);
    const alternatives = questions.map((item, alternativeIndex) => alternativeIndex === questionIndex ? Number.POSITIVE_INFINITY : cost(mark, item).total);
    const gap = Math.min(...alternatives, selected.total + 2) - selected.total;
    return { markId: mark.id, questionId: question.id, confidence: Math.max(0, Math.min(1, 1 - selected.total / 5 + gap / 10)), secondBestGap: gap, features: selected.features };
  });
}

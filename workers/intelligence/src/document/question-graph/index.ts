import type { LayoutRegion } from "../layout/types";

export interface QuestionNode { id: string; label: string; parentId?: string; pageIds: string[]; regionIds: string[]; continuationIds: string[] }
export interface QuestionGraph { questions: QuestionNode[]; unassignedRegionIds: string[] }

export function buildQuestionGraph(regions: readonly LayoutRegion[], pageOrder: ReadonlyMap<string, number> = new Map()): QuestionGraph {
  const ordered = [...regions].sort((a, b) => (pageOrder.get(a.pageId) ?? Number.MAX_SAFE_INTEGER) - (pageOrder.get(b.pageId) ?? Number.MAX_SAFE_INTEGER) || a.pageId.localeCompare(b.pageId) || a.box.y - b.box.y || a.box.x - b.box.x);
  const questions: QuestionNode[] = [];
  const unassignedRegionIds: string[] = [];
  let current: QuestionNode | undefined;
  for (const region of ordered) {
    if (region.class === "question_number" || region.class === "subquestion_number") {
      const label = region.text?.trim() || `unknown-${questions.length + 1}`;
      const parent = region.class === "subquestion_number" ? [...questions].reverse().find((item) => !item.parentId) : undefined;
      current = { id: `question:${label}:${region.id}`, label, ...(parent ? { parentId: parent.id } : {}), pageIds: [region.pageId], regionIds: [region.id], continuationIds: [] };
      questions.push(current); continue;
    }
    if (!current) { unassignedRegionIds.push(region.id); continue; }
    current.regionIds.push(region.id);
    if (!current.pageIds.includes(region.pageId)) current.pageIds.push(region.pageId);
    if (region.class === "continuation_region") current.continuationIds.push(region.id);
  }
  return { questions, unassignedRegionIds };
}

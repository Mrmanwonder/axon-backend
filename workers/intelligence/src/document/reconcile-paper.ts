import type { Box } from "../schemas";
import type { LayoutRegion, RegionClass } from "./layout/types";
import { matchMarks, type MarkAssignment, type QuestionCandidate } from "./mark-matcher";
import { buildQuestionGraph, type QuestionGraph } from "./question-graph";

interface StoredRegion { id: string; page_id: string; class: RegionClass; box_json: string; confidence: number; text_value: string | null; page_index: number }

function parseBox(value: string): Box {
  const parsed = JSON.parse(value) as Partial<Box>;
  if (![parsed.x, parsed.y, parsed.width, parsed.height].every((item) => typeof item === "number" && Number.isFinite(item))) throw new Error("Invalid stored layout box");
  return parsed as Box;
}

function unionBox(regions: readonly LayoutRegion[]): Box {
  const x = Math.min(...regions.map((item) => item.box.x));
  const y = Math.min(...regions.map((item) => item.box.y));
  const right = Math.max(...regions.map((item) => item.box.x + item.box.width));
  const bottom = Math.max(...regions.map((item) => item.box.y + item.box.height));
  return { x, y, width: right - x, height: bottom - y };
}

export async function reconcilePaperTopology(db: D1Database, paperId: string): Promise<{ graph: QuestionGraph; assignments: MarkAssignment[] }> {
  const rows = await db.prepare(`SELECT lr.id, lr.page_id, lr.class, lr.box_json, lr.confidence, lr.text_value, pp.page_index
    FROM layout_region lr JOIN paper_page pp ON pp.page_id = lr.page_id
    WHERE pp.paper_id = ? ORDER BY pp.page_index, pp.created_at, lr.id`).bind(paperId).all<StoredRegion>();
  const regions: LayoutRegion[] = rows.results.map((row) => ({ id: row.id, pageId: row.page_id, class: row.class, box: parseBox(row.box_json), confidence: row.confidence, ...(row.text_value !== null ? { text: row.text_value } : {}) }));
  const pageOrder = new Map(rows.results.map((row) => [row.page_id, row.page_index]));
  const graph = buildQuestionGraph(regions, pageOrder);
  const byId = new Map(regions.map((region) => [region.id, region]));
  const questions: QuestionCandidate[] = graph.questions.flatMap((question, order) => {
    const owned = question.regionIds.map((id) => byId.get(id)).filter((item): item is LayoutRegion => Boolean(item));
    if (owned.length === 0) return [];
    return [{ id: question.id, pageIds: question.pageIds, box: unionBox(owned), order, continuationPageIds: question.pageIds, commentRegionIds: owned.filter((item) => item.class === "teacher_comment").map((item) => item.id) }];
  });
  const marks = regions.filter((region) => region.class === "marginal_mark").map((region) => ({ id: region.id, pageId: region.pageId, box: region.box }));
  const assignments = matchMarks(marks, questions);
  const statements: D1PreparedStatement[] = [
    db.prepare("UPDATE mark_assignment SET question_id = NULL, trust_state = 'UNVERIFIED' WHERE mark_region_id IN (SELECT lr.id FROM layout_region lr JOIN paper_page pp ON pp.page_id = lr.page_id WHERE pp.paper_id = ?)").bind(paperId),
    db.prepare("DELETE FROM question_region WHERE question_id IN (SELECT id FROM question_node WHERE paper_id = ?)").bind(paperId),
    db.prepare("DELETE FROM question_node WHERE paper_id = ?").bind(paperId)
  ];
  for (const [order, question] of graph.questions.entries()) {
    statements.push(db.prepare("INSERT INTO question_node (id, paper_id, label, parent_id, page_ids_json, reading_order) VALUES (?, ?, ?, ?, ?, ?)")
      .bind(question.id, paperId, question.label, question.parentId ?? null, JSON.stringify(question.pageIds), order));
    for (const regionId of question.regionIds) statements.push(db.prepare("INSERT INTO question_region (question_id, region_id, relationship) VALUES (?, ?, 'contains')").bind(question.id, regionId));
  }
  for (const assignment of assignments) statements.push(db.prepare(`INSERT OR REPLACE INTO mark_assignment
    (mark_region_id, question_id, confidence, second_best_gap, features_json, trust_state) VALUES (?, ?, ?, ?, ?, ?)`)
    .bind(assignment.markId, assignment.questionId, assignment.confidence, assignment.secondBestGap, JSON.stringify(assignment.features), assignment.confidence >= 0.9 && assignment.secondBestGap >= 0.2 ? "AUTO_VERIFIED" : "UNVERIFIED"));
  await db.batch(statements);
  return { graph, assignments };
}

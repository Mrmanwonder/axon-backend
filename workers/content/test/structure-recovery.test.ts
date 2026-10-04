import { beforeEach, expect, test, vi } from "vitest";
const fixture = vi.hoisted(() => ({
  page: { id: "page", paper_id: "paper", student_id: "student", page_number: 2, r2_key: "student/paper/page/2", r2_bucket: "derived", structure_status: "pending", teacher_marks: [] },
  markers: [{ paper_id: "paper", student_id: "student", page_number: 2 }, { paper_id: "paper", student_id: "student", page_number: 3 }],
  status: "pending", deleteError: false, events: [] as string[],
}));
vi.mock("@mastery/shared/worker.js", () => ({ consumeQueue: (handle: any) => handle, failRun: vi.fn() }));
vi.mock("@mastery/shared/structure-page.js", () => ({ loadStructurePage: async () => fixture.page }));
vi.mock("@mastery/shared/r2.js", () => ({ imageRef: async () => ({ type: "image_url", image_url: { url: "fixture" } }) }));
vi.mock("@mastery/shared/page.js", () => ({ pageDimensions: () => ({ width: 1000, height: 2000 }), UNPLACEABLE_PAGE_REASON: "no dimensions" }));
vi.mock("@mastery/shared/model-client.js", () => ({ callModel: async () => ({ parsed: { is_graded_exam_paper: true, regions: [] } }) }));
vi.mock("@mastery/shared/structure_plan.js", async importOriginal => ({
  ...await importOriginal<any>(),
  planPage: () => [{ row: { order_index: 0 }, order_index: 0, span: { page: 2, box: { x: 0, y: 0, w: 100, h: 100 } } }],
}));
import worker from "../../structure/src/index.js";
function db() {
  return {
    rpc: async () => ({ data: { advanced: false }, error: null }),
    from(table: string) {
      let action = "read";
      let values: any;
      const filters: Record<string, any> = {};
      const b: any = {
        select: () => b, eq: (key: string, value: any) => { filters[key] = value; return b; },
        in: () => b, delete: () => { action = "delete"; return b; },
        insert: (v: any) => { action = "insert"; values = v; return b; },
        update: (v: any) => { action = "update"; values = v; return b; },
        maybeSingle: async () => ({ data: { status: "structure", route_override: null }, error: null }),
        then: (yes: any) => {
          let data: any = [];
          let error: any = null;
          if (table === "question_region" && action === "insert") { fixture.events.push("regions"); data = [{ id: "region", order_index: 0 }]; }
          if (table === "page_unreadable" && action === "delete") {
            fixture.events.push("retire");
            if (fixture.deleteError) error = { code: "08006", message: "connection failed" };
            else fixture.markers = fixture.markers.filter(marker => Object.entries(filters).some(([key, value]) => (marker as any)[key] !== value));
          }
          if (table === "paper_page" && action === "update") { fixture.status = values.structure_status; fixture.events.push(values.structure_status); }
          return Promise.resolve({ data, error, count: 2 }).then(yes);
        },
      };
      return b;
    },
  };
}
beforeEach(() => {
  fixture.markers = [{ paper_id: "paper", student_id: "student", page_number: 2 }, { paper_id: "paper", student_id: "student", page_number: 3 }];
  fixture.status = "pending"; fixture.deleteError = false; fixture.events = [];
});
test("a successful retry retires only its page's unreadability before durable completion", async () => {
  await (worker.queue as any)({ env: {}, sb: db(), msg: { run_id: "run", page_id: "page" }, attempt: 1, beat: async () => {} });
  expect(fixture.status).toBe("done");
  expect(fixture.markers.map(m => m.page_number)).toEqual([3]);
  expect(fixture.events.indexOf("retire")).toBeGreaterThan(fixture.events.indexOf("regions"));
  expect(fixture.events.indexOf("done")).toBeGreaterThan(fixture.events.indexOf("retire"));
});
test("failed unreadability cleanup cannot stamp the page done", async () => {
  fixture.deleteError = true;
  await expect((worker.queue as any)({ env: {}, sb: db(), msg: { run_id: "run", page_id: "page" }, attempt: 1, beat: async () => {} })).rejects.toThrow();
  expect(fixture.status).toBe("running");
  expect(fixture.markers).toHaveLength(2);
});

import test from "node:test";
import assert from "node:assert/strict";
import { resolveDocument } from "../cbse-scheme-registry.mjs";

class FakeQuery {
  constructor(db) {
    this.db = db;
    this.op = "select";
    this.filters = [];
    this.payload = null;
  }

  select() { return this; }
  eq(column, value) { this.filters.push(row => row[column] === value); return this; }
  neq(column, value) { this.filters.push(row => row[column] !== value); return this; }
  is(column, value) { this.filters.push(row => row[column] === value); return this; }
  insert(payload) { this.op = "insert"; this.payload = payload; return this; }
  update(payload) { this.op = "update"; this.payload = payload; return this; }

  matching() {
    return this.db.rows.filter(row => this.filters.every(filter => filter(row)));
  }

  async execute() {
    if (this.op === "insert") {
      const row = {
        id: "doc-" + this.db.nextId++,
        ...structuredClone(this.payload),
      };
      this.db.rows.push(row);
      return { data: [structuredClone(row)], error: null };
    }

    if (this.op === "update") {
      const matches = this.matching();
      for (const row of matches) Object.assign(row, structuredClone(this.payload));
      return { data: matches.map(row => structuredClone(row)), error: null };
    }

    return { data: this.matching().map(row => structuredClone(row)), error: null };
  }

  async maybeSingle() {
    const result = await this.execute();
    if (result.error) return result;
    if (result.data.length > 1) return { data: null, error: new Error("multiple rows") };
    return { data: result.data[0] ?? null, error: null };
  }

  async single() {
    const result = await this.execute();
    if (result.error) return result;
    if (result.data.length !== 1) return { data: null, error: new Error("expected one row") };
    return { data: result.data[0], error: null };
  }

  then(resolve, reject) {
    return this.execute().then(resolve, reject);
  }
}

class FakeSupabase {
  constructor() {
    this.rows = [];
    this.nextId = 1;
  }

  from(table) {
    assert.equal(table, "scheme_document");
    return new FakeQuery(this);
  }
}

const context = {
  provider: { id: "provider-cbse" },
  policy: {
    id: "policy-cbse",
    copyright_access_class: "public_official",
    terms_url: "https://www.cbse.gov.in/cbsenew/documents/WEBSITE_POLICY_U.pdf",
    policy_version: "verified-test",
  },
};

const identity = { id: "assessment-physics" };

const pair = {
  header: {
    session: "2026-27",
    subjectCode: "042",
    classLevel: 12,
  },
  coverage: 0.9,
  parsed: {
    sqpBlocks: 39,
    msBlocks: 39,
    coveredBaseQuestions: 30,
    expectedQuestions: 33,
  },
};

const url = "https://cbseacademic.nic.in/Physics-MS.pdf";
const hashA = "a".repeat(64);
const hashB = "b".repeat(64);
const hashC = "c".repeat(64);

test("unchanged official scheme hash is idempotent and never downgrades ready state", async () => {
  const sb = new FakeSupabase();

  const first = await resolveDocument(sb, context, identity, pair, url, hashA, true);
  assert.equal(sb.rows.length, 1);
  assert.equal(first.id, "doc-1");
  assert.equal(first.extraction_status, "pending");

  sb.rows[0].extraction_status = "ready";

  const second = await resolveDocument(sb, context, identity, pair, url, hashA, true);
  assert.equal(sb.rows.length, 1);
  assert.equal(second.id, first.id);
  assert.equal(second.extraction_status, "ready");
  assert.equal(sb.rows[0].superseded_by_id, null);
});

test("changed bytes create an immutable version chain instead of overwriting history", async () => {
  const sb = new FakeSupabase();

  const first = await resolveDocument(sb, context, identity, pair, url, hashA, true);
  sb.rows.find(row => row.id === first.id).extraction_status = "ready";

  const second = await resolveDocument(sb, context, identity, pair, url, hashB, true);
  assert.equal(sb.rows.length, 2);
  assert.equal(sb.rows.find(row => row.id === first.id).extraction_status, "superseded");
  assert.equal(sb.rows.find(row => row.id === first.id).superseded_by_id, second.id);

  sb.rows.find(row => row.id === second.id).extraction_status = "ready";
  const third = await resolveDocument(sb, context, identity, pair, url, hashC, true);

  assert.equal(sb.rows.length, 3);
  assert.equal(sb.rows.find(row => row.id === first.id).superseded_by_id, second.id);
  assert.equal(sb.rows.find(row => row.id === second.id).superseded_by_id, third.id);
  assert.equal(sb.rows.find(row => row.id === second.id).extraction_status, "superseded");
  assert.equal(sb.rows.find(row => row.id === third.id).extraction_status, "pending");
});

test("ingestion cannot silently reactivate superseded or revoked scheme versions", async () => {
  const sb = new FakeSupabase();

  const first = await resolveDocument(sb, context, identity, pair, url, hashA, true);
  const second = await resolveDocument(sb, context, identity, pair, url, hashB, true);

  await assert.rejects(
    () => resolveDocument(sb, context, identity, pair, url, hashA, true),
    /superseded and cannot be reactivated/,
  );

  const current = sb.rows.find(row => row.id === second.id);
  current.extraction_status = "revoked";
  current.revoked_at = "2026-09-26T00:00:00.000Z";
  current.revocation_reason = "test";

  await assert.rejects(
    () => resolveDocument(sb, context, identity, pair, url, hashB, true),
    /revoked and cannot be reactivated/,
  );

  assert.equal(sb.rows.length, 2);
  assert.equal(current.revoked_at, "2026-09-26T00:00:00.000Z");
});

#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createClient } from "@supabase/supabase-js";
import {
  CBSE_INDEXES,
  CBSE_OFFICIAL_HOST,
  CBSE_PARSER_VERSION,
  assertOfficialCbseUrl,
  discoverCbseIndex,
  normalizedSubjectName,
  pairOfficialQuestions,
  sha256,
} from "./lib/cbse-scheme.mjs";

const RIGHTS_KIND = "marking_scheme";

function parseArgs(argv) {
  const out = {
    command: "discover",
    classLevel: 12,
    subject: null,
    write: false,
    documentId: null,
    reason: null,
    bundlePath: null,
  };
  for (const arg of argv) {
    if (["discover", "verify", "ingest", "revoke"].includes(arg)) out.command = arg;
    else if (arg === "--write") out.write = true;
    else if (arg.startsWith("--class=")) out.classLevel = Number(arg.slice(8));
    else if (arg.startsWith("--subject=")) out.subject = arg.slice(10);
    else if (arg.startsWith("--document=")) out.documentId = arg.slice(11);
    else if (arg.startsWith("--reason=")) out.reason = arg.slice(9);
    else if (arg.startsWith("--bundle=")) out.bundlePath = arg.slice(9);
    else throw new Error("Unknown argument: " + arg);
  }
  if (![10, 12].includes(out.classLevel)) throw new Error("--class must be 10 or 12");
  return out;
}

async function fetchOfficial(url) {
  const source = assertOfficialCbseUrl(url);
  const response = await fetch(source, {
    redirect: "follow",
    signal: AbortSignal.timeout(30_000),
    headers: {
      "user-agent": "Axon official CBSE scheme registry/1.0 (+https://axonstudy.online/)",
      "accept": "*/*",
    },
  });
  if (!response.ok) throw new Error(source + " returned " + response.status);
  return Buffer.from(await response.arrayBuffer());
}

function pdfText(bytes) {
  const dir = mkdtempSync(join(tmpdir(), "axon-cbse-scheme-"));
  const pdf = join(dir, "document.pdf");
  const txt = join(dir, "document.txt");
  try {
    writeFileSync(pdf, bytes);
    execFileSync("pdftotext", ["-layout", "-nopgbrk", pdf, txt], { stdio: "pipe" });
    return readFileSync(txt, "utf8");
  } catch (error) {
    throw new Error("Install poppler-utils/pdftotext before CBSE registry ingestion: " + String(error));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function adminClient() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) throw new Error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required");
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

async function one(query, label) {
  const { data, error } = await query;
  if (error) throw new Error(label + ": " + error.message);
  if (!data) throw new Error(label + ": not found");
  return data;
}

async function loadContext(sb, classLevel) {
  const provider = await one(
    sb.from("curriculum_provider").select("id,key").eq("key", "cbse").maybeSingle(),
    "CBSE provider",
  );
  const stageKey = classLevel === 10 ? "cbse_10" : "cbse_12";
  const stage = await one(
    sb.from("curriculum_stage").select("id,programme_id,key").eq("key", stageKey).maybeSingle(),
    stageKey,
  );
  const policy = await one(
    sb.from("scheme_source_policy")
      .select("id,hostname,copyright_access_class,reproduction_permitted,active,terms_url,policy_version")
      .eq("provider_id", provider.id)
      .eq("source_kind", RIGHTS_KIND)
      .eq("hostname", CBSE_OFFICIAL_HOST)
      .maybeSingle(),
    "CBSE marking-scheme policy",
  );
  if (!policy.active || !policy.reproduction_permitted || policy.copyright_access_class !== "public_official") {
    throw new Error("CBSE source policy does not currently permit scheme ingestion");
  }
  return { provider, stage, policy };
}

function offeringMatches(row, names, code) {
  if (row.external_code === code) return true;
  if (names.has(normalizedSubjectName(row.display_name))) return true;
  return (row.aliases ?? []).some(alias => names.has(normalizedSubjectName(alias)));
}

async function resolveOffering(sb, stage, indexSubject, header, write) {
  const { data, error } = await sb.from("subject_offering")
    .select("id,display_name,external_code,external_code_kind,aliases,availability")
    .eq("programme_id", stage.programme_id)
    .eq("stage_id", stage.id)
    .eq("availability", "active");
  if (error) throw error;

  const names = new Set([
    normalizedSubjectName(indexSubject),
    normalizedSubjectName(header.subject),
  ]);
  const matches = (data ?? []).filter(row => offeringMatches(row, names, header.subjectCode));
  if (matches.length !== 1) {
    throw new Error(
      "Expected one " + stage.key + " offering for "
      + indexSubject + " / code " + header.subjectCode + ", found " + matches.length,
    );
  }
  const offering = matches[0];
  if (offering.external_code && offering.external_code !== header.subjectCode) {
    throw new Error(
      "Subject-code conflict for " + offering.display_name + ": "
      + offering.external_code + " vs " + header.subjectCode,
    );
  }

  if (write && !offering.external_code) {
    const { error: updateError } = await sb.from("subject_offering")
      .update({
        external_code: header.subjectCode,
        external_code_kind: "cbse_subject_code",
        updated_at: new Date().toISOString(),
      })
      .eq("id", offering.id)
      .is("external_code", null);
    if (updateError) throw updateError;
  }
  return offering;
}

async function resolveIdentity(sb, context, offering, pair, sqpUrl, sqpHash, write) {
  const { data, error } = await sb.from("assessment_identity")
    .select("*")
    .eq("programme_id", context.stage.programme_id)
    .eq("subject_offering_id", offering.id)
    .eq("exam_year", pair.header.examYear)
    .eq("session", pair.header.session)
    .eq("assessment_route", "sample_paper")
    .is("paper_code", null)
    .is("component_code", null);
  if (error) throw error;
  if ((data ?? []).length > 1) throw new Error("Duplicate assessment identities already exist");

  const row = {
    programme_id: context.stage.programme_id,
    subject_offering_id: offering.id,
    level: null,
    exam_year: pair.header.examYear,
    session: pair.header.session,
    paper_code: null,
    component_code: null,
    variant: null,
    zone: null,
    assessment_route: "sample_paper",
    title: offering.display_name + " Sample Question Paper " + pair.header.session,
    official_source_url: sqpUrl,
    source_document_version: pair.header.session + ":" + sqpHash.slice(0, 16),
    metadata: {
      provider: "cbse",
      class_level: pair.header.classLevel,
      subject_code: pair.header.subjectCode,
      sqp_sha256: sqpHash,
      source_kind: "sample_question_paper",
    },
    updated_at: new Date().toISOString(),
  };

  if (!write) return data?.[0] ?? { id: "dry-run-assessment", ...row };

  if (data?.[0]) {
    const result = await sb.from("assessment_identity")
      .update(row).eq("id", data[0].id).select("*").single();
    if (result.error) throw result.error;
    return result.data;
  }
  const result = await sb.from("assessment_identity").insert(row).select("*").single();
  if (result.error) throw result.error;
  return result.data;
}

export async function resolveDocument(sb, context, identity, pair, msUrl, msHash, write) {
  const version = pair.header.session + ":" + msHash.slice(0, 16);
  if (!write) return { id: "dry-run-document", source_url: msUrl, source_version: version };

  const existing = await sb.from("scheme_document")
    .select("*")
    .eq("provider_id", context.provider.id)
    .eq("sha256", msHash)
    .eq("source_kind", RIGHTS_KIND)
    .maybeSingle();
  if (existing.error) throw existing.error;
  if (existing.data && existing.data.assessment_identity_id !== identity.id) {
    throw new Error("Official scheme hash is already bound to another assessment identity");
  }
  if (existing.data?.revoked_at || existing.data?.extraction_status === "revoked") {
    throw new Error("Official scheme version is revoked and cannot be reactivated by ingestion");
  }
  if (existing.data?.superseded_by_id || existing.data?.extraction_status === "superseded") {
    throw new Error("Official scheme version is superseded and cannot be reactivated by ingestion");
  }

  const payload = {
    provider_id: context.provider.id,
    assessment_identity_id: identity.id,
    policy_id: context.policy.id,
    source_url: msUrl,
    source_kind: RIGHTS_KIND,
    source_version: version,
    sha256: msHash,
    retrieved_at: new Date().toISOString(),
    copyright_access_class: context.policy.copyright_access_class,
    extraction_status: "pending",
    parser_version: CBSE_PARSER_VERSION,
    metadata: {
      subject_code: pair.header.subjectCode,
      class_level: pair.header.classLevel,
      session: pair.header.session,
      coverage: pair.coverage,
      parsed: pair.parsed,
      terms_url: context.policy.terms_url,
      policy_version: context.policy.policy_version,
    },
  };

  let document;
  if (existing.data) {
    // An unchanged source is the same immutable version. Refresh provenance
    // metadata without downgrading a document that already completed parsing.
    const stableStatus = ["ready", "complete", "extracted"].includes(existing.data.extraction_status)
      ? existing.data.extraction_status
      : "pending";
    const result = await sb.from("scheme_document")
      .update({ ...payload, extraction_status: stableStatus })
      .eq("id", existing.data.id)
      .select("*")
      .single();
    if (result.error) throw result.error;
    document = result.data;
  } else {
    const result = await sb.from("scheme_document")
      .insert({
        ...payload,
        revoked_at: null,
        revocation_reason: null,
        superseded_by_id: null,
      })
      .select("*")
      .single();
    if (result.error) throw result.error;
    document = result.data;
  }

  // Preserve the immutable version chain: only the currently active predecessor
  // may point to this new version. Older already-superseded rows keep the link
  // that was written when their direct successor was created.
  const older = await sb.from("scheme_document")
    .select("id")
    .eq("assessment_identity_id", identity.id)
    .eq("source_kind", RIGHTS_KIND)
    .neq("id", document.id)
    .is("revoked_at", null)
    .is("superseded_by_id", null);
  if (older.error) throw older.error;
  for (const prior of older.data ?? []) {
    const result = await sb.from("scheme_document")
      .update({ extraction_status: "superseded", superseded_by_id: document.id })
      .eq("id", prior.id);
    if (result.error) throw result.error;
  }
  return document;
}

async function storeQuestions(sb, identity, document, offering, pair, write) {
  if (!write) return pair.questions.length;
  let count = 0;
  for (const question of pair.questions) {
    const canonicalId = [
      "CBSE",
      pair.header.session,
      pair.header.classLevel,
      pair.header.subjectCode,
      "SQP",
      question.label,
    ].join(":");

    const result = await sb.from("canonical_question").upsert({
      board: "CBSE",
      exam_year: pair.header.examYear,
      subject: offering.display_name,
      question_text: question.questionText,
      max_marks: question.maxMarks,
      marking_scheme: question.markingScheme,
      scheme_source: document.source_url,
      scheme_version: document.source_version,
      syllabus_code: null,
      qualification_level: null,
      series: null,
      paper_number: null,
      variant: null,
      component_code: null,
      canonical_id: canonicalId,
      assessment_identity_id: identity.id,
      question_label: question.label,
      scheme_document_id: document.id,
      updated_at: new Date().toISOString(),
    }, { onConflict: "canonical_id" });
    if (result.error) throw result.error;
    count++;
  }
  return count;
}

async function verifyOne(entry, includeBundle = false) {
  const [sqpBytes, msBytes] = await Promise.all([
    fetchOfficial(entry.sqpUrl),
    fetchOfficial(entry.msUrl),
  ]);
  const sqpHash = sha256(sqpBytes);
  const msHash = sha256(msBytes);
  const pair = pairOfficialQuestions(pdfText(sqpBytes), pdfText(msBytes));
  if (pair.header.classLevel !== entry.classLevel) {
    throw new Error("Index/PDF class mismatch for " + entry.subject);
  }
  if (!normalizedSubjectName(entry.subject).includes(normalizedSubjectName(pair.header.subject))
      && !normalizedSubjectName(pair.header.subject).includes(normalizedSubjectName(entry.subject))) {
    throw new Error("Index/PDF subject mismatch for " + entry.subject + " vs " + pair.header.subject);
  }
  const summary = {
    subject: entry.subject,
    code: pair.header.subjectCode,
    classLevel: pair.header.classLevel,
    session: pair.header.session,
    examYear: pair.header.examYear,
    coverage: pair.coverage,
    canonicalQuestions: pair.questions.length,
    parsed: pair.parsed,
    sqpSha256: sqpHash,
    msSha256: msHash,
  };
  if (!includeBundle) return { summary, bundle: null };
  return {
    summary,
    bundle: {
      schemaVersion: 1,
      provider: "cbse",
      parserVersion: CBSE_PARSER_VERSION,
      subject: entry.subject,
      sqpUrl: entry.sqpUrl,
      msUrl: entry.msUrl,
      sqpSha256: sqpHash,
      msSha256: msHash,
      header: pair.header,
      coverage: pair.coverage,
      parsed: pair.parsed,
      questions: pair.questions,
    },
  };
}

async function ingestOne(sb, entry, write) {
  const [sqpBytes, msBytes] = await Promise.all([
    fetchOfficial(entry.sqpUrl),
    fetchOfficial(entry.msUrl),
  ]);
  const sqpHash = sha256(sqpBytes);
  const msHash = sha256(msBytes);
  const pair = pairOfficialQuestions(pdfText(sqpBytes), pdfText(msBytes));
  if (pair.header.classLevel !== entry.classLevel) {
    throw new Error("Index/PDF class mismatch for " + entry.subject);
  }

  const context = await loadContext(sb, entry.classLevel);
  const offering = await resolveOffering(sb, context.stage, entry.subject, pair.header, write);
  const identity = await resolveIdentity(sb, context, offering, pair, entry.sqpUrl, sqpHash, write);
  const document = await resolveDocument(sb, context, identity, pair, entry.msUrl, msHash, write);
  const stored = await storeQuestions(sb, identity, document, offering, pair, write);

  if (write) {
    const result = await sb.from("scheme_document").update({
      extraction_status: "ready",
      metadata: {
        subject_code: pair.header.subjectCode,
        class_level: pair.header.classLevel,
        session: pair.header.session,
        coverage: pair.coverage,
        parsed: pair.parsed,
        canonical_questions: stored,
        terms_url: context.policy.terms_url,
        policy_version: context.policy.policy_version,
      },
    }).eq("id", document.id);
    if (result.error) throw result.error;
  }

  return {
    subject: entry.subject,
    code: pair.header.subjectCode,
    classLevel: entry.classLevel,
    session: pair.header.session,
    coverage: pair.coverage,
    canonicalQuestions: stored,
    sqpSha256: sqpHash,
    msSha256: msHash,
    write,
  };
}

async function revoke(sb, args) {
  if (!args.documentId || !args.reason?.trim()) {
    throw new Error("revoke requires --document=<uuid> and --reason=<text>");
  }
  if (!args.write) return { documentId: args.documentId, reason: args.reason, write: false };
  const result = await sb.from("scheme_document").update({
    revoked_at: new Date().toISOString(),
    revocation_reason: args.reason.trim(),
    extraction_status: "revoked",
  }).eq("id", args.documentId).select("id,source_url,revoked_at,revocation_reason").single();
  if (result.error) throw result.error;
  return result.data;
}

async function indexRows(classLevel, subject) {
  const indexUrl = CBSE_INDEXES[classLevel];
  const response = await fetch(indexUrl, {
    signal: AbortSignal.timeout(20_000),
    headers: { "user-agent": "Axon official CBSE scheme registry/1.0 (+https://axonstudy.online/)" },
  });
  if (!response.ok) throw new Error(indexUrl + " returned " + response.status);
  let rows = discoverCbseIndex(await response.text(), indexUrl, classLevel);
  if (subject) {
    const needle = normalizedSubjectName(subject);
    rows = rows.filter(row => normalizedSubjectName(row.subject).includes(needle));
  }
  return rows;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.command === "discover") {
    console.log(JSON.stringify(await indexRows(args.classLevel, args.subject), null, 2));
    return;
  }

  if (args.command === "verify") {
    const rows = await indexRows(args.classLevel, args.subject);
    if (!rows.length) throw new Error("No matching CBSE SQP/MS rows discovered");
    const results = [];
    const bundles = [];
    const failures = [];
    for (const row of rows) {
      try {
        const verified = await verifyOne(row, Boolean(args.bundlePath));
        results.push(verified.summary);
        if (verified.bundle) bundles.push(verified.bundle);
      } catch (error) {
        failures.push({
          subject: row.subject,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    if (args.bundlePath && bundles.length) {
      writeFileSync(args.bundlePath, JSON.stringify({
        schemaVersion: 1,
        generatedAt: new Date().toISOString(),
        parserVersion: CBSE_PARSER_VERSION,
        bundles,
      }, null, 2));
    }
    console.log(JSON.stringify({
      results,
      failures,
      bundleWritten: args.bundlePath ? bundles.length > 0 : false,
    }, null, 2));
    if (failures.length) process.exitCode = 2;
    return;
  }

  const sb = adminClient();
  if (args.command === "revoke") {
    console.log(JSON.stringify(await revoke(sb, args), null, 2));
    return;
  }

  const rows = await indexRows(args.classLevel, args.subject);
  if (!rows.length) throw new Error("No matching CBSE SQP/MS rows discovered");

  const results = [];
  const failures = [];
  for (const row of rows) {
    try {
      results.push(await ingestOne(sb, row, args.write));
    } catch (error) {
      failures.push({
        subject: row.subject,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  console.log(JSON.stringify({ write: args.write, results, failures }, null, 2));
  if (failures.length) process.exitCode = 2;
}

if (import.meta.url === new URL(process.argv[1], "file:").href) {
  main().catch(error => {
    console.error(error instanceof Error ? error.stack : error);
    process.exitCode = 1;
  });
}

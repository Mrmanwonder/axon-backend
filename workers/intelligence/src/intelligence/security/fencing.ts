import type { Evidence } from "../../schemas";

export function fenceEvidence(evidence: readonly Evidence[]): string {
  return evidence.map((item) => {
    const serialized = JSON.stringify(item.value).replaceAll("</UNTRUSTED_EVIDENCE>", "&lt;/UNTRUSTED_EVIDENCE&gt;");
    return `<UNTRUSTED_EVIDENCE id=${JSON.stringify(item.id)} source=${JSON.stringify(item.source)}>\n${serialized}\n</UNTRUSTED_EVIDENCE>`;
  }).join("\n");
}

export function minimizeEvidence(evidence: readonly Evidence[], allowedIds: ReadonlySet<string>): Evidence[] {
  return evidence.filter((item) => allowedIds.has(item.id)).map((item) => structuredClone(item));
}

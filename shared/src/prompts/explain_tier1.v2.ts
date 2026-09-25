// Versioned replacement for paper feedback. The schema and deterministic
// validator remain shared with v1; the prompt version changes because the
// teacher-intent doctrine and evidence contract changed materially.
export { SYSTEM, instruction, SCHEMA, validate } from "./explain_tier1.v1.js";
export type {
  Cause,
  ErrorType,
  ExplainInstructionOptions,
  ExplainResult,
  LossReason,
} from "./explain_tier1.v1.js";

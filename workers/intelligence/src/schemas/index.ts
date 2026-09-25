import { Type, type Static, type TSchema } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";

export const InformationClassSchema = Type.Union([
  Type.Literal("OBSERVED"), Type.Literal("DERIVED"),
  Type.Literal("VERIFIED_EXTERNAL"), Type.Literal("STABLE_KNOWLEDGE"),
  Type.Literal("INFERRED"), Type.Literal("UNKNOWN")
]);

export const BoxSchema = Type.Object({
  x: Type.Number({ minimum: 0, maximum: 1 }),
  y: Type.Number({ minimum: 0, maximum: 1 }),
  width: Type.Number({ minimum: 0, maximum: 1 }),
  height: Type.Number({ minimum: 0, maximum: 1 })
}, { additionalProperties: false });

export const EvidenceSchema = Type.Object({
  id: Type.String({ minLength: 1, maxLength: 256 }),
  informationClass: InformationClassSchema,
  source: Type.Union([
    Type.Literal("student"), Type.Literal("paper"), Type.Literal("teacher"),
    Type.Literal("axon_db"), Type.Literal("tool"), Type.Literal("retrieval"),
    Type.Literal("official_source"), Type.Literal("stable_knowledge")
  ]),
  authority: Type.Union([Type.Literal("primary"), Type.Literal("secondary"), Type.Literal("derived"), Type.Literal("low")]),
  value: Type.Unknown(),
  provenance: Type.Object({
    paperId: Type.Optional(Type.String({ maxLength: 128 })), pageId: Type.Optional(Type.String({ maxLength: 128 })),
    region: Type.Optional(BoxSchema), url: Type.Optional(Type.String({ format: "uri" })),
    retrievedAt: Type.Optional(Type.String({ format: "date-time" })),
    publishedAt: Type.Optional(Type.String({ format: "date-time" })),
    toolId: Type.Optional(Type.String()), artifactHash: Type.Optional(Type.String())
  }, { additionalProperties: false }),
  verification: Type.Union([Type.Literal("verified"), Type.Literal("probable"), Type.Literal("unverified")]),
  confidence: Type.Optional(Type.Number({ minimum: 0, maximum: 1 }))
}, { additionalProperties: false });

export const ClaimSchema = Type.Object({
  id: Type.String({ minLength: 1, maxLength: 256 }),
  text: Type.String({ minLength: 1 }),
  type: Type.Union([
    Type.Literal("observed"), Type.Literal("derived"), Type.Literal("stable"),
    Type.Literal("retrieved"), Type.Literal("interpretation"), Type.Literal("calculation")
  ]),
  evidenceIds: Type.Array(Type.String()),
  risk: Type.Union([Type.Literal("low"), Type.Literal("medium"), Type.Literal("high"), Type.Literal("critical")]),
  verificationStatus: Type.Union([Type.Literal("pending"), Type.Literal("verified"), Type.Literal("rejected"), Type.Literal("uncertain")])
}, { additionalProperties: false });

export const IntentSchema = Type.Union([
  Type.Literal("direct_answer"), Type.Literal("concept_explanation"),
  Type.Literal("problem_solving"), Type.Literal("hint"),
  Type.Literal("mistake_diagnosis"), Type.Literal("paper_feedback"),
  Type.Literal("work_check"), Type.Literal("comparison"),
  Type.Literal("socratic"), Type.Literal("current_information"),
  Type.Literal("source_question"), Type.Literal("casual")
]);

export const ReasoningResultSchema = Type.Object({
  status: Type.Union([Type.Literal("supported"), Type.Literal("partially_supported"), Type.Literal("insufficient_evidence")]),
  intent: IntentSchema,
  claims: Type.Array(ClaimSchema),
  conceptIds: Type.Array(Type.String()),
  misconception: Type.Optional(Type.Object({ text: Type.String(), evidenceIds: Type.Array(Type.String()) }, { additionalProperties: false })),
  teachingStrategy: Type.Union([
    Type.Literal("direct"), Type.Literal("worked_example"), Type.Literal("contrast"),
    Type.Literal("analogy"), Type.Literal("socratic"), Type.Literal("step_by_step")
  ]),
  nextAction: Type.Optional(Type.String()),
  uncertaintyReason: Type.Optional(Type.String())
}, { additionalProperties: false });

export const VerificationResultSchema = Type.Object({
  passed: Type.Boolean(),
  failures: Type.Array(Type.Object({
    code: Type.Union([
      Type.Literal("unsupported_claim"), Type.Literal("contradiction"), Type.Literal("calculation_error"),
      Type.Literal("missing_uncertainty"), Type.Literal("invented_context"), Type.Literal("invented_source"),
      Type.Literal("incorrect_attribution"), Type.Literal("prompt_injection"), Type.Literal("other")
    ]),
    claimId: Type.Optional(Type.String()),
    detail: Type.String()
  }, { additionalProperties: false }))
}, { additionalProperties: false });

export const TutorRequestSchema = Type.Object({
  requestId: Type.Optional(Type.String({ maxLength: 128 })),
  studentId: Type.String({ minLength: 1, maxLength: 256 }),
  message: Type.String({ minLength: 1, maxLength: 20_000 }),
  grade: Type.Optional(Type.Integer({ minimum: 1, maximum: 16 })),
  board: Type.Optional(Type.String({ maxLength: 100 })),
  subject: Type.Optional(Type.String({ maxLength: 100 })),
  topic: Type.Optional(Type.String({ maxLength: 200 })),
  paperId: Type.Optional(Type.String({ maxLength: 128 })),
  depth: Type.Optional(Type.Union([Type.Literal("BRIEF"), Type.Literal("NORMAL"), Type.Literal("DEEP")])),
  evidence: Type.Optional(Type.Array(EvidenceSchema, { maxItems: 100 }))
}, { additionalProperties: false });

export const TutorResponseSchema = Type.Object({
  traceId: Type.String(),
  status: Type.Union([Type.Literal("supported"), Type.Literal("partially_supported"), Type.Literal("insufficient_evidence"), Type.Literal("controlled_failure")]),
  answer: Type.String(),
  citations: Type.Array(Type.Object({ title: Type.String(), url: Type.String({ format: "uri" }) }, { additionalProperties: false })),
  verification: Type.Object({ passed: Type.Boolean(), repaired: Type.Boolean(), failures: Type.Array(Type.String()) }, { additionalProperties: false })
}, { additionalProperties: false });

export const CorrectionEventSchema = Type.Object({
  field: Type.String({ minLength: 1, maxLength: 128 }), predicted: Type.Unknown(), corrected: Type.Unknown(),
  acceptedValue: Type.Unknown(), artifactId: Type.String({ minLength: 1, maxLength: 256 }), pipelineVersion: Type.String({ minLength: 1, maxLength: 64 }),
  model: Type.String({ minLength: 1, maxLength: 128 }), promptHash: Type.String({ minLength: 1, maxLength: 128 }), contextMetadata: Type.Record(Type.String(), Type.Unknown())
}, { additionalProperties: false });

export type InformationClass = Static<typeof InformationClassSchema>;
export type Box = Static<typeof BoxSchema>;
export type Evidence = Static<typeof EvidenceSchema>;
export type Claim = Static<typeof ClaimSchema>;
export type Intent = Static<typeof IntentSchema>;
export type ReasoningResult = Static<typeof ReasoningResultSchema>;
export type VerificationResult = Static<typeof VerificationResultSchema>;
export type TutorRequest = Static<typeof TutorRequestSchema>;
export type TutorResponse = Static<typeof TutorResponseSchema>;
export type CorrectionEvent = Static<typeof CorrectionEventSchema>;

export function parseSchema<T extends TSchema>(schema: T, value: unknown): Static<T> {
  if (!Value.Check(schema, value)) {
    const issues = [...Value.Errors(schema, value)].map((issue) => `${issue.path || "/"}: ${issue.message}`);
    throw new SchemaValidationError(issues);
  }
  return value;
}

export class SchemaValidationError extends Error {
  constructor(readonly issues: string[]) {
    super(`Schema validation failed: ${issues.join("; ")}`);
    this.name = "SchemaValidationError";
  }
}

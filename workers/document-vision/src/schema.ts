import { Type, type Static, type TSchema } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";

export const BoxSchema = Type.Object({
  x: Type.Number({ minimum: 0, maximum: 1 }),
  y: Type.Number({ minimum: 0, maximum: 1 }),
  width: Type.Number({ minimum: 0.0001, maximum: 1 }),
  height: Type.Number({ minimum: 0.0001, maximum: 1 })
}, { additionalProperties: false });

export const REGION_CLASSES = [
  "question_number", "subquestion_number", "printed_question", "student_answer", "teacher_annotation",
  "teacher_comment", "marginal_mark", "marks_available", "reported_total", "diagram", "table",
  "working_area", "header", "footer", "page_number", "crossed_out_work", "continuation_region"
] as const;

const RegionClassSchema = Type.Union(REGION_CLASSES.map((value) => Type.Literal(value)));
const InkLayerSchema = Type.Union(["PRINTED", "STUDENT", "TEACHER", "UNKNOWN"].map((value) => Type.Literal(value)));

export const ReaderOutputSchema = Type.Object({
  orientationDegrees: Type.Union([0, 90, 180, 270].map((value) => Type.Literal(value))),
  perspectiveDegrees: Type.Number({ minimum: -45, maximum: 45 }),
  cropCompleteness: Type.Number({ minimum: 0, maximum: 1 }),
  regions: Type.Array(Type.Object({
    class: RegionClassSchema,
    box: BoxSchema,
    confidence: Type.Number({ minimum: 0, maximum: 1 }),
    text: Type.Union([Type.String({ maxLength: 10_000 }), Type.Null()]),
    layer: InkLayerSchema
  }, { additionalProperties: false }), { maxItems: 300 })
}, { additionalProperties: false });

export const RequestSchema = Type.Object({
  contractVersion: Type.Literal("axon-document-vision.v1"),
  pageId: Type.String({ minLength: 1, maxLength: 128 }),
  mimeType: Type.Union([Type.Literal("image/jpeg"), Type.Literal("image/png"), Type.Literal("image/webp")]),
  dataBase64: Type.String({ minLength: 4, maxLength: 16_000_000 })
}, { additionalProperties: false });

const QualityMetricsSchema = Type.Object({
  blur: Type.Number({ minimum: 0, maximum: 1 }),
  glareFraction: Type.Number({ minimum: 0, maximum: 1 }),
  perspectiveDegrees: Type.Number({ minimum: -90, maximum: 90 }),
  resolution: Type.Number({ minimum: 0, maximum: 1 }),
  compression: Type.Number({ minimum: 0, maximum: 1 }),
  cropCompleteness: Type.Number({ minimum: 0, maximum: 1 }),
  shadowFraction: Type.Number({ minimum: 0, maximum: 1 })
}, { additionalProperties: false });

const InkSignalsSchema = Type.Object({
  printedProbability: Type.Number({ minimum: 0, maximum: 1 }),
  colourDistanceFromPrint: Type.Number({ minimum: 0, maximum: 1 }),
  strokeDifference: Type.Number({ minimum: 0, maximum: 1 }),
  marginTendency: Type.Number({ minimum: 0, maximum: 1 }),
  annotationOverlap: Type.Number({ minimum: 0, maximum: 1 }),
  handwritingDifference: Type.Number({ minimum: 0, maximum: 1 })
}, { additionalProperties: false });

const ReadSchema = Type.Object({
  value: Type.Union([Type.String(), Type.Null()]),
  alternatives: Type.Array(Type.String(), { maxItems: 10 }),
  status: Type.Union([Type.Literal("read"), Type.Literal("ambiguous"), Type.Literal("unreadable")]),
  region: BoxSchema,
  readerIds: Type.Array(Type.String(), { minItems: 1, maxItems: 5 })
}, { additionalProperties: false });

export const VisionAnalysisSchema = Type.Object({
  qualityMetrics: QualityMetricsSchema,
  orientationDegrees: Type.Number({ minimum: -360, maximum: 360 }),
  regions: Type.Array(Type.Object({
    id: Type.String({ minLength: 1, maxLength: 128 }),
    class: RegionClassSchema,
    box: BoxSchema,
    confidence: Type.Number({ minimum: 0, maximum: 1 }),
    text: Type.Optional(Type.String({ maxLength: 10_000 })),
    inkSignals: Type.Optional(InkSignalsSchema)
  }, { additionalProperties: false }), { maxItems: 2_000 }),
  reads: Type.Array(Type.Object({
    regionId: Type.String(),
    reads: Type.Array(ReadSchema, { minItems: 1, maxItems: 5 })
  }, { additionalProperties: false }), { maxItems: 2_000 }),
  conditionedImageBase64: Type.Optional(Type.String({ maxLength: 30_000_000 })),
  conditionedImageMimeType: Type.Optional(Type.Literal("image/webp"))
}, { additionalProperties: false });

export type Box = Static<typeof BoxSchema>;
export type ReaderOutput = Static<typeof ReaderOutputSchema>;
export type ReaderRegion = ReaderOutput["regions"][number];
export type VisionRequest = Static<typeof RequestSchema>;
export type VisionAnalysis = Static<typeof VisionAnalysisSchema>;
export type InkLayer = ReaderRegion["layer"];

export function parseSchema<T extends TSchema>(schema: T, value: unknown): Static<T> {
  if (!Value.Check(schema, value)) {
    const issues = [...Value.Errors(schema, value)].slice(0, 20).map((issue) => `${issue.path || "/"}: ${issue.message}`);
    throw new Error(`SCHEMA_VALIDATION_FAILED: ${issues.join("; ")}`);
  }
  return value;
}

import { Type, type Static } from "@sinclair/typebox";
import { parseSchema } from "../../schemas";
import { readBoundedJsonBody } from "../../shared/bounded-json";

const BoxSchema = Type.Object({
  x: Type.Number({ minimum: 0, maximum: 1 }), y: Type.Number({ minimum: 0, maximum: 1 }),
  width: Type.Number({ minimum: 0, maximum: 1 }), height: Type.Number({ minimum: 0, maximum: 1 })
}, { additionalProperties: false });

const RegionClassSchema = Type.Union([
  "question_number", "subquestion_number", "printed_question", "student_answer", "teacher_annotation",
  "teacher_comment", "marginal_mark", "marks_available", "reported_total", "diagram", "table",
  "working_area", "header", "footer", "page_number", "crossed_out_work", "continuation_region"
].map((value) => Type.Literal(value)));

const QualityMetricsSchema = Type.Object({
  blur: Type.Number({ minimum: 0, maximum: 1 }), glareFraction: Type.Number({ minimum: 0, maximum: 1 }),
  perspectiveDegrees: Type.Number({ minimum: -90, maximum: 90 }), resolution: Type.Number({ minimum: 0, maximum: 1 }),
  compression: Type.Number({ minimum: 0, maximum: 1 }), cropCompleteness: Type.Number({ minimum: 0, maximum: 1 }),
  shadowFraction: Type.Number({ minimum: 0, maximum: 1 })
}, { additionalProperties: false });

const InkSignalsSchema = Type.Object({
  printedProbability: Type.Number({ minimum: 0, maximum: 1 }), colourDistanceFromPrint: Type.Number({ minimum: 0, maximum: 1 }),
  strokeDifference: Type.Number({ minimum: 0, maximum: 1 }), marginTendency: Type.Number({ minimum: 0, maximum: 1 }),
  annotationOverlap: Type.Number({ minimum: 0, maximum: 1 }), handwritingDifference: Type.Number({ minimum: 0, maximum: 1 })
}, { additionalProperties: false });

const ReadSchema = Type.Object({
  value: Type.Union([Type.String(), Type.Null()]), alternatives: Type.Array(Type.String(), { maxItems: 10 }),
  status: Type.Union([Type.Literal("read"), Type.Literal("ambiguous"), Type.Literal("unreadable")]),
  region: BoxSchema, readerIds: Type.Array(Type.String(), { minItems: 1, maxItems: 5 })
}, { additionalProperties: false });

export const VisionAnalysisSchema = Type.Object({
  qualityMetrics: QualityMetricsSchema,
  orientationDegrees: Type.Number({ minimum: -360, maximum: 360 }),
  regions: Type.Array(Type.Object({
    id: Type.String({ minLength: 1, maxLength: 128 }),
    class: RegionClassSchema, box: BoxSchema, confidence: Type.Number({ minimum: 0, maximum: 1 }),
    text: Type.Optional(Type.String({ maxLength: 10_000 })), inkSignals: Type.Optional(InkSignalsSchema)
  }, { additionalProperties: false }), { maxItems: 2_000 }),
  reads: Type.Array(Type.Object({ regionId: Type.String(), reads: Type.Array(ReadSchema, { minItems: 1, maxItems: 5 }) }, { additionalProperties: false }), { maxItems: 2_000 }),
  conditionedImageBase64: Type.Optional(Type.String({ maxLength: 30_000_000 })),
  conditionedImageMimeType: Type.Optional(Type.Literal("image/webp"))
}, { additionalProperties: false });

export type VisionAnalysis = Static<typeof VisionAnalysisSchema>;

export interface DocumentVisionProvider {
  readonly id: string;
  readonly privacyMode: "zdr" | "unverified";
  analyze(input: { bytes: ArrayBuffer; mimeType: string; pageId: string; timeoutMs: number }): Promise<{ analysis: VisionAnalysis; latencyMs: number }>;
}

export class ServiceBindingDocumentVisionProvider implements DocumentVisionProvider {
  readonly id = "axon-document-vision";
  constructor(readonly service: Fetcher, readonly privacyMode: "zdr" | "unverified") {}

  async analyze(input: { bytes: ArrayBuffer; mimeType: string; pageId: string; timeoutMs: number }): Promise<{ analysis: VisionAnalysis; latencyMs: number }> {
    const started = Date.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), input.timeoutMs);
    try {
      const bytes = new Uint8Array(input.bytes);
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      const response = await this.service.fetch("https://axon-document-vision/v1/analyze", {
        method: "POST",
        signal: controller.signal,
        headers: { "content-type": "application/json", "x-axon-contract-version": "axon-document-vision.v1" },
        body: JSON.stringify({ pageId: input.pageId, mimeType: input.mimeType, dataBase64: btoa(binary), contractVersion: "axon-document-vision.v1" })
      });
      if (!response.ok) throw new Error(`VISION_PROVIDER_${response.status}`);
      const payload = await readBoundedJsonBody(response.body, 32_000_000);
      return { analysis: parseSchema(VisionAnalysisSchema, payload), latencyMs: Date.now() - started };
    } finally {
      clearTimeout(timer);
    }
  }
}

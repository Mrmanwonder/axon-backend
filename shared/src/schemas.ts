// JSON Schemas passed as response_format.json_schema to the model. Kept
// alongside the TS types the pipeline actually uses (see each worker's
// index.ts) rather than generated from them, because the schema is what
// constrains the model's output and needs to be readable in its own right.

const box = {
  type: ["object", "null"],
  additionalProperties: false,
  required: ["x", "y", "w", "h"],
  properties: {
    x: { type: "number", minimum: 0, maximum: 1000 },
    y: { type: "number", minimum: 0, maximum: 1000 },
    w: { type: "number", minimum: 0, maximum: 1000 },
    h: { type: "number", minimum: 0, maximum: 1000 },
  },
} as const;

const valueWithBox = (valueType: string) => ({
  type: ["object", "null"],
  additionalProperties: false,
  required: ["value", "box", "page_index"],
  properties: {
    value: { type: [valueType, "null"] },
    box,
    page_index: { type: "integer", minimum: 0 },
  },
});

export const STRUCTURE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["is_graded_exam_paper", "not_a_paper_reason", "reported_total", "stated_maximum", "regions"],
  properties: {
    // Students will upload homework, blank question papers, textbook pages, and
    // things that are not schoolwork at all. A system that dutifully extracts a
    // textbook page into the analytics quietly degrades every insight
    // downstream, so this is the first question asked, not the last.
    is_graded_exam_paper: { type: "boolean" },
    not_a_paper_reason: { type: ["string", "null"] },
    reported_total: valueWithBox("number"),
    stated_maximum: valueWithBox("number"),
    regions: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["candidate_number", "number_box", "box", "continues_from_previous", "structure_confidence"],
        properties: {
          candidate_number: { type: ["string", "null"] },
          number_box: box,
          box: { ...box, type: "object" },
          // A question that runs off the bottom of this page and picks up on the
          // next is normal, not a failure. Saying so here is what lets the
          // caller stitch the two halves into one region.
          continues_from_previous: { type: "boolean" },
          structure_confidence: { type: "string", enum: ["high", "low"] },
        },
      },
    },
  },
} as const;

export const CONTENT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "question_text",
    "student_answer",
    "marks_awarded",
    "marks_available",
    "teacher_remark",
    "region_type",
    "recognition_confidence",
    "unreadable",
    "unreadable_reason",
    "answer_block",
  ],
  properties: {
    question_text: valueWithBox("string"),
    student_answer: valueWithBox("string"),
    /**
     * The answer with its structure kept, alongside the flat string.
     *
     * `student_answer` being `text` is why a student described the screen as
     * "numbers, alphabets and signs paired together": there was nowhere to put
     * structure, so structure was destroyed at write time. Handwritten `8/2`
     * arrived as `8+1` — turning a correct step into a false one — and a
     * struck-through `32` arrived as nothing at all, which under CAIE marking
     * is mark-bearing evidence being discarded.
     *
     * Nullable: a diagram or an unreadable crop has no block to give, and an
     * invented one would be worse than none.
     */
    answer_block: {
      type: ["object", "null"],
      additionalProperties: false,
      required: ["lines", "notation_profile", "raw_text"],
      properties: {
        lines: {
          type: "array",
          items: {
            type: "object",
            additionalProperties: false,
            required: ["segments", "role"],
            properties: {
              role: { type: "string", enum: ["working", "final_answer", "restatement", "crossed_out"] },
              segments: {
                type: "array",
                items: {
                  type: "object",
                  additionalProperties: false,
                  required: ["type", "latex", "text", "annotations", "bbox", "confidence"],
                  properties: {
                    type: { type: "string", enum: ["math", "prose", "numeral", "binary", "label"] },
                    latex: { type: ["string", "null"] },
                    text: { type: ["string", "null"] },
                    annotations: {
                      type: "array",
                      items: {
                        type: "string",
                        enum: ["struck_through", "boxed", "circled", "underlined", "inserted", "overwritten"],
                      },
                    },
                    // Into the page image, so a student can tap a segment and
                    // see the handwriting it was read from.
                    bbox: {
                      type: ["object", "null"],
                      additionalProperties: false,
                      required: ["x", "y", "w", "h", "page_index"],
                      properties: {
                        x: { type: "number" }, y: { type: "number" },
                        w: { type: "number" }, h: { type: "number" },
                        page_index: { type: "number" },
                      },
                    },
                    confidence: { type: ["number", "null"] },
                  },
                },
              },
            },
          },
        },
        notation_profile: { type: "string" },
        raw_text: { type: "string" },
      },
    },
    marks_awarded: valueWithBox("number"),
    marks_available: valueWithBox("number"),
    teacher_remark: valueWithBox("string"),
    region_type: { type: "string", enum: ["prose", "math", "diagram", "table", "mcq", "mixed"] },
    recognition_confidence: { type: "string", enum: ["high", "medium", "low"] },
    unreadable: { type: "boolean" },
    unreadable_reason: { type: ["string", "null"] },
  },
} as const;

export const EXPLANATION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "can_explain", "cause", "marks_lost", "explanation", "do_this_next", "concepts",
    "command_word", "command_word_note", "model_answer", "loss_reasons",
  ],
  properties: {
    // If no reason for the deduction can be constructed, saying so plainly and
    // pointing at the teacher is an honest and genuinely useful outcome. It is
    // not a failure of the request.
    can_explain: { type: "boolean" },
    cause: {
      type: ["string", "null"],
      enum: [
        "conceptual_gap",
        "procedural_slip",
        "misread_question",
        "incomplete",
        "presentation",
        "keyword_miss",
        "timed_out",
        null,
      ],
    },
    marks_lost: { type: ["number", "null"] },
    explanation: { type: ["string", "null"] },
    // Null when the model cannot clear the quality floor. An empty slot is
    // honest; generic advice trains students to stop reading.
    do_this_next: { type: ["string", "null"] },
    concepts: { type: "array", items: { type: "string" } },

    // The command word the question was built around, and one line on what it
    // requires. Reading "Explain" as "State" is one of the most common and most
    // fixable ways a Cambridge mark goes, and it is invisible to a student who
    // was never told the word was doing work. Null when the stem could not be
    // read or the word is not one we recognise — see command_words.ts.
    command_word: { type: ["string", "null"] },
    command_word_note: { type: ["string", "null"] },

    // The corrected working, in the same steps as the student's own answer.
    // do_this_next names the fix; this one shows it carried through. Null
    // rather than a paraphrase of the instruction.
    model_answer: { type: ["string", "null"] },

    // A two-part mistake is two diagnoses. Averaging "skipped the inversion"
    // and "sign error in the exponent" into one "conceptual gap" loses the part
    // the student could have fixed on the day.
    loss_reasons: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["marks", "cause", "note", "error_type"],
        properties: {
          // Axon's own category for what the mistake looked like, as opposed to
          // `cause`, which is why it happened. No marking scheme needed, so it
          // is available on every paper. Deliberately not Cambridge's
          // vocabulary and deliberately not letter codes — see LossReason.
          error_type: {
            type: "string",
            enum: ["method", "final_answer", "omitted_step", "presentation", "other"],
          },
          marks: { type: "number" },
          cause: {
            type: ["string", "null"],
            enum: [
              "conceptual_gap",
              "procedural_slip",
              "misread_question",
              "incomplete",
              "presentation",
              "keyword_miss",
              "timed_out",
              null,
            ],
          },
          // Anchored in the student's own working where it can be — "between
          // your line 2 and line 3" — rather than in a scheme we do not have.
          note: { type: ["string", "null"] },
        },
      },
    },
  },
} as const;

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
  ],
  properties: {
    question_text: valueWithBox("string"),
    student_answer: valueWithBox("string"),
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
        required: ["marks", "cause", "note"],
        properties: {
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

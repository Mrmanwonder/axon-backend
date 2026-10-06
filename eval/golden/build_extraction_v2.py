# Generates eval/golden/content-v2.json and eval/golden/structure-v2.json, the golden sets for the
# content.v2 and structure.v2 prompts (drawn answers read; page furniture refused).
#
# Synthetic only. Every page below is described in words and invented for this eval: no student
# data, no question-paper text and no mark-scheme text from any board. The contexts (counters,
# buses, a school fete) and the numbers are original. Where a case imitates the LAYOUT of a
# Cambridge page (a lone page number at the top centre, a footer with a paper code), the paper
# code is an obviously fictional one.
#
# Content and structure are vision stages, so a case is a page description plus the expected
# reading. To run one, a person renders the description as a synthetic page image (handwriting
# on a printed template) and stores it with the eval case; the description is the specification
# that image must match. Labels are drafted by Claude and are DRAFT (needs_human_label) until a
# person confirms them.
#
#   python3 eval/golden/build_extraction_v2.py
import json, uuid

NS = uuid.UUID("6f1c2b7e-94a3-4d58-8b0e-1e7a3c9d5f62")
content_cases = []
structure_cases = []


def content(key, page, region_type, student_answer, traps=(), table=False, notes=None):
    content_cases.append({
        "id": str(uuid.uuid5(NS, "content:" + key)),
        "key": key,
        "stage": "content",
        "prompt_file": "shared/src/prompts/content.v2.ts",
        "page_description": page,
        "expected": {
            "region_type": region_type,
            # Exact lines of student_answer, or null when the student wrote nothing readable as text.
            "student_answer_lines": student_answer,
            "every_value_has_box": True,
            "is_latex_array": table,
        },
        "traps": list(traps),
        "notes": notes,
        "requires_image": True,
        "needs_human_label": True,
    })


def structure(key, page, printed_page_number, regions, forbidden_labels=(), traps=()):
    structure_cases.append({
        "id": str(uuid.uuid5(NS, "structure:" + key)),
        "key": key,
        "stage": "structure",
        "prompt_file": "shared/src/prompts/structure.v2.ts",
        "page_description": page,
        "expected": {
            "printed_page_number": printed_page_number,
            # In page order. candidate_number may be any value in `accept` (null allowed when listed).
            "regions": regions,
            # No region may carry any of these as, or at the start of, its candidate_number.
            "forbidden_labels": list(forbidden_labels),
        },
        "traps": list(traps),
        "requires_image": True,
        "needs_human_label": True,
    })


def region(accept, continues=False, confidence=None):
    return {"accept": list(accept), "continues_from_previous": continues, "structure_confidence": confidence}


# ---------------- content.v2: diagrams ----------------
content("tree-two-stage-values",
        "Question 2(a), 3 marks. Printed: a two-stage tree diagram about a bus. First stage printed "
        "labels 'Monday' then 'Late' and 'On time'; second stage printed 'Tuesday', with 'Late' and "
        "'On time' after each first branch. Student wrote in pen on the branches: 0.3 on Monday Late, "
        "0.7 on Monday On time, 0.45 on Tuesday Late after Late, 0.55 on Tuesday On time after Late, "
        "0.1 on Tuesday Late after On time, 0.9 on Tuesday On time after On time. Teacher wrote 3 in the margin.",
        "diagram",
        ["Monday Late: 0.3", "On time: 0.7",
         "Tuesday after Late, Late: 0.45", "Tuesday after Late, On time: 0.55",
         "Tuesday after On time, Late: 0.1", "Tuesday after On time, On time: 0.9"],
        traps=["v1 returned null here: the student sees 'Not read'",
               "never describe the tree or say whether the probabilities sum to 1"])
content("tree-fraction-values",
        "Question 5(i), 2 marks. Printed: one-stage tree for drawing a counter from a bag, branches "
        "labelled 'Red' and 'Blue'. Student wrote 3/8 on Red and 5/8 on Blue.",
        "diagram",
        ["Red: \\tfrac{3}{8}", "Blue: \\tfrac{5}{8}"],
        traps=["fractions stay fractions in LaTeX, not 0.375"])
content("tree-drawn-no-values",
        "Question 2(a), 3 marks. Student drew a tree diagram freehand with four branches and wrote no "
        "numbers or words on it. Teacher wrote 0 in the margin.",
        "diagram", None,
        traps=["student_answer stays null: nothing was written as text",
               "never describe the branches"])
content("box-plot-written-summary",
        "Question 3(b), 3 marks. Printed: a grid with a horizontal scale 'Time (minutes)' from 0 to 40. "
        "Student drew a box-and-whisker plot on the grid and wrote beside it: 'LQ = 14', 'median = 19', "
        "'UQ = 26'. Nothing else is written.",
        "diagram",
        ["LQ = 14", "median = 19", "UQ = 26"],
        traps=["the whisker ends are not written as text: never read them off the grid",
               "do not add min or max values"])
content("box-plot-drawn-only",
        "Question 3(b), 3 marks. Printed grid 'Mass (kg)' 0 to 60. Student drew a box-and-whisker plot "
        "with no numbers or words written anywhere.",
        "diagram", None,
        traps=["reading 12, 20, 31 off where the box edges fall is a measurement, not a transcription"])
content("histogram-written-fd",
        "Question 4(a), 4 marks. Printed: axes 'Height (cm)' and 'Frequency density'. Student drew four "
        "bars and wrote above them, left to right: 'fd = 0.8', 'fd = 2.4', 'fd = 1.5', 'fd = 0.3'. "
        "Under the grid the student wrote the working line '12 / 5 = 2.4'.",
        "diagram",
        ["fd = 0.8", "fd = 2.4", "fd = 1.5", "fd = 0.3", "\\tfrac{12}{5} = 2.4"],
        traps=["bar heights are never read off the drawing",
               "the working line under the grid is the student's text and is transcribed"])
content("histogram-drawn-only",
        "Question 4(a), 4 marks. Printed axes with a scale. Student drew five bars of different heights; "
        "nothing is written.",
        "diagram", None,
        traps=["no value may be read off the bar heights"])
content("labelled-axes-printed-only",
        "Question 6(c), 2 marks. Student sketched a curve on printed axes labelled 'x' and 'y'. The only "
        "text on the diagram is printed. The student wrote '(0, 4)' beside where the curve meets the y-axis.",
        "diagram",
        ["(0, 4)"],
        traps=["printed axis labels are not the student's text",
               "the intercept is transcribed because the student wrote it, not because it is on the curve"])

# ---------------- content.v2: tables ----------------
content("table-probability-distribution",
        "Question 1(b), 3 marks. Printed table with first row 'x | 0 | 1 | 2 | 3' and second row "
        "'P(X = x)' with four empty cells. Student filled the cells with 1/10, 3/10, 2/5, 1/5.",
        "table",
        ["\\begin{array}{|c|c|c|c|c|} \\hline x & 0 & 1 & 2 & 3 \\\\ \\hline P(X = x) & \\tfrac{1}{10} & "
         "\\tfrac{3}{10} & \\tfrac{2}{5} & \\tfrac{1}{5} \\\\ \\hline \\end{array}"],
        table=True,
        traps=["a table is not a diagram", "printed header cells are copied as printed"])
content("table-cell-left-blank",
        "Question 2(a), 2 marks. Printed frequency table 'Score | 1 | 2 | 3' with a 'Frequency' row. "
        "Student wrote 4 and 9 in the first two cells and left the third empty.",
        "table",
        ["\\begin{array}{|c|c|c|c|} \\hline Score & 1 & 2 & 3 \\\\ \\hline Frequency & 4 & 9 &  \\\\ "
         "\\hline \\end{array}"],
        table=True,
        traps=["an empty cell stays empty; never fill it from the total"])
content("table-unreadable-cell",
        "Question 2(a), 2 marks. Same printed table as table-cell-left-blank. Student wrote 4, then a "
        "heavily smudged digit, then 7.",
        "table",
        ["\\begin{array}{|c|c|c|c|} \\hline Score & 1 & 2 & 3 \\\\ \\hline Frequency & 4 &  & 7 \\\\ "
         "\\hline \\end{array}"],
        table=True,
        traps=["the smudged cell is left empty and recognition_confidence lowered, never guessed"])

# ---------------- content.v2: controls ----------------
content("control-prose-unchanged",
        "Question 7, 2 marks. Printed question asks for a reason. Student wrote one sentence: "
        "'Because the events are independent.' No diagram.",
        "prose",
        ["Because the events are independent."],
        traps=["v2 must not change how prose is read"])

# ---------------- structure.v2 ----------------
structure("lone-page-number-top-centre",
          "Printed page 6 of a fictional paper 'XX/00/Z/Z/99'. Lone '6' at the top centre. Question 4 "
          "is printed at the left margin a little below, with parts (a) and (b) and handwritten answers. "
          "Footer: a copyright line and the fictional paper code.",
          "6",
          [region(["4", "4(a)"]), region(["(b)", "4(b)"])],
          forbidden_labels=["6"],
          traps=["the page number is not question 6"])
structure("continued-part-top-of-page",
          "Printed page 7. Lone '7' at the top centre. Directly under it at the left margin, a printed "
          "'(b)' whose question number (3) was on page 6, followed by a handwritten answer. Then "
          "question 4 starts lower down.",
          "7",
          [region(["(b)", None], confidence=None), region(["4"])],
          forbidden_labels=["7", "7(b)", "7b"],
          traps=["never put the page number in front of a continued part label"])
structure("continuation-no-label",
          "Printed page 9. Lone '9' at the top centre. The top third of the page is handwriting with no "
          "printed number, continuing an answer from page 8. Question 5 is printed below it.",
          "9",
          [region([None], continues=True), region(["5"])],
          forbidden_labels=["9"])
structure("footer-furniture",
          "Printed page 11. Lone '11' at the top centre. Question 6 at the left margin. Footer: a "
          "copyright line, the fictional paper code 'XX/00/Z/Z/99' and '[Turn over' at the bottom right.",
          "11",
          [region(["6"])],
          forbidden_labels=["11", "99", "00"],
          traps=["nothing in the footer starts or numbers a region"])
structure("genuine-question-equals-page",
          "Printed page 6. Lone '6' at the top centre. Question 6 is printed at the LEFT margin about "
          "a fifth of the way down, with a handwritten answer.",
          "6",
          [region(["6"])],
          traps=["a real question 6 on page 6, at the left margin, must keep its number"])
structure("page-number-in-footer",
          "Printed page 12, page number printed alone at the bottom centre. Question 8 at the left "
          "margin at the top of the page.",
          "12",
          [region(["8"])],
          forbidden_labels=["12"])
structure("no-printed-page-number",
          "A school test with no printed page numbers. Handwritten question numbers '1.' and '2.' at "
          "the left margin.",
          None,
          [region(["1"]), region(["2"])],
          traps=["printed_page_number is null when none is printed; never inferred from upload order"])

out = __file__.replace("build_extraction_v2.py", "")
meta = {"labels": "DRAFT: drafted by Claude, need a person's confirmation before they are truth",
        "source": "synthetic page descriptions; images to be rendered from them before a live run"}
json.dump({"golden_set_version": "content-synthetic-v2", "stage": "content", **meta, "cases": content_cases},
          open(out + "content-v2.json", "w"), ensure_ascii=False, indent=1)
json.dump({"golden_set_version": "structure-synthetic-v2", "stage": "structure", **meta, "cases": structure_cases},
          open(out + "structure-v2.json", "w"), ensure_ascii=False, indent=1)
print(len(content_cases), "content cases,", len(structure_cases), "structure cases")

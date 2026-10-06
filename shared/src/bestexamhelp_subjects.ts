// bestexamhelp.com subject folders, keyed by Cambridge syllabus code.
//
// A mark scheme's address there is fully determined by the paper code once the
// subject folder is known:
//   https://bestexamhelp.com/exam/<level>/<slug>-<code>/<year>/<code>_<s|w|m><yy>_ms_<component>.pdf
// e.g. 9231/11/O/N/25 -> .../cambridge-international-a-level/mathematics-further-9231/2025/9231_w25_ms_11.pdf
// The year folder is the exam year for all three series (m, s, w).
//
// Read from the site's own subject indexes on 6 Oct 2026:
//   /exam/cambridge-international-a-level/, /exam/cambridge-igcse/, /exam/cambridge-o-level/
// A code missing here is looked up at runtime from the same index pages
// (cambridge_scheme.ts), so a subject the site adds later still resolves.

export type BestExamHelpLevel = "cambridge-international-a-level" | "cambridge-igcse" | "cambridge-o-level";

export const BESTEXAMHELP_LEVELS: readonly BestExamHelpLevel[] = [
  "cambridge-international-a-level", "cambridge-igcse", "cambridge-o-level",
];

/** code -> "<level>/<slug>-<code>" */
export const BESTEXAMHELP_SUBJECTS: Readonly<Record<string, string>> = {
  // Cambridge International AS & A Level
  "9706": "cambridge-international-a-level/accounting-9706",
  "9713": "cambridge-international-a-level/applied-information-and-communication-technology-9713",
  "9700": "cambridge-international-a-level/biology-9700",
  "9609": "cambridge-international-a-level/business-9609",
  "9707": "cambridge-international-a-level/business-studies-9707",
  "9701": "cambridge-international-a-level/chemistry-9701",
  "9618": "cambridge-international-a-level/computer-science-9618",
  "9608": "cambridge-international-a-level/computer-science-9608",
  "9691": "cambridge-international-a-level/computing-9691",
  "9708": "cambridge-international-a-level/economics-9708",
  "9489": "cambridge-international-a-level/history-9489",
  "9488": "cambridge-international-a-level/islamic-studies-9488",
  "9084": "cambridge-international-a-level/law-9084",
  "9709": "cambridge-international-a-level/mathematics-9709",
  "9231": "cambridge-international-a-level/mathematics-further-9231",
  "9607": "cambridge-international-a-level/media-studies-9607",
  "9702": "cambridge-international-a-level/physics-9702",
  "9990": "cambridge-international-a-level/psychology-9990",
  "9698": "cambridge-international-a-level/psychology-9698",
  "9699": "cambridge-international-a-level/sociology-9699",
  // Cambridge IGCSE
  "0452": "cambridge-igcse/accounting-0452",
  "0508": "cambridge-igcse/arabic-first-language-0508",
  "0400": "cambridge-igcse/art-and-design-0400",
  "0600": "cambridge-igcse/agriculture-0600",
  "0610": "cambridge-igcse/biology-0610",
  "0450": "cambridge-igcse/business-studies-0450",
  "0620": "cambridge-igcse/chemistry-0620",
  "0478": "cambridge-igcse/computer-science-0478",
  "0509": "cambridge-igcse/chinese-first-language-0509",
  "0411": "cambridge-igcse/drama-0411",
  "0455": "cambridge-igcse/economics-0455",
  "0500": "cambridge-igcse/english-first-language-0500",
  "0524": "cambridge-igcse/english-first-language-us-0524",
  "0475": "cambridge-igcse/english-literature-0475",
  "0680": "cambridge-igcse/environmental-management-0680",
  "0454": "cambridge-igcse/enterprise-0454",
  "0648": "cambridge-igcse/food-and-nutrition-0648",
  "0501": "cambridge-igcse/french-first-language-0501",
  "0460": "cambridge-igcse/geography-0460",
  "0457": "cambridge-igcse/global-perspectives-0457",
  "0470": "cambridge-igcse/history-0470",
  "0493": "cambridge-igcse/islamiyat-0493",
  "0580": "cambridge-igcse/mathematics-0580",
  "0606": "cambridge-igcse/mathematics-additional-0606",
  "0607": "cambridge-igcse/mathematics-international-0607",
  "0410": "cambridge-igcse/music-0410",
  "0696": "cambridge-igcse/malay-first-language-0696",
  "0625": "cambridge-igcse/physics-0625",
  "0413": "cambridge-igcse/physical-education-0413",
  "0490": "cambridge-igcse/religious-studies-0490",
  "0653": "cambridge-igcse/science-combined-0653",
  "0654": "cambridge-igcse/sciences-co-ordinated-0654",
  "0495": "cambridge-igcse/sociology-0495",
  "0502": "cambridge-igcse/spanish-first-language-0502",
  "0471": "cambridge-igcse/travel-and-tourism-0471",
  // Cambridge O Level
  "7707": "cambridge-o-level/accounting-7707",
  "5090": "cambridge-o-level/biology-5090",
  "7115": "cambridge-o-level/business-studies-7115",
  "3204": "cambridge-o-level/bengali-3204",
  "7094": "cambridge-o-level/bangladesh-studies-7094",
  "5070": "cambridge-o-level/chemistry-5070",
  "7100": "cambridge-o-level/commerce-7100",
  "2210": "cambridge-o-level/computer-science-2210",
  "7010": "cambridge-o-level/computer-studies-7010",
  "2281": "cambridge-o-level/economics-2281",
  "1123": "cambridge-o-level/english-language-1123",
  "4024": "cambridge-o-level/mathematics-d-4024",
  "4037": "cambridge-o-level/mathematics-additional-4037",
  "5054": "cambridge-o-level/physics-5054",
  "7110": "cambridge-o-level/principles-of-accounts-7110",
  "2059": "cambridge-o-level/pakistan-studies-2059",
  "4040": "cambridge-o-level/statistics-4040",
};

// lib/data/subject_data.dart
// Generated from CSV data in Curriculumn folder

class SubjectData {
  final String code;
  final String name;
  final String level;
  final List<ChapterData> chapters;
  final List<int> papers;
  final List<SubjectResource> resources;

  const SubjectData({
    required this.code,
    this.name = '',
    this.level = '',
    this.chapters = const [],
    this.papers = const [],
    this.resources = const [],
  });
}

class ChapterData {
  final int number;
  final String title;
  final List<String> subtopics;
  final String notes;

  const ChapterData({
    required this.number,
    required this.title,
    required this.subtopics,
    this.notes = '',
  });
}

class SubjectResource {
  final String name;
  final String? url;
  final String? localPath;
  final String type;
  final bool isNative;

  const SubjectResource({
    required this.name,
    this.url,
    this.localPath,
    required this.type,
    this.isNative = false,
  });
}

const List<SubjectData> allSubjectsData = [
  SubjectData(
    code: '0580',
    name: 'Mathematics',
    level: 'IGCSE',
    resources: [
      SubjectResource(
        name: 'Syllabus PDF',
        url: 'https://www.cambridgeinternational.org/Images/698481-2026-syllabus.pdf',
        type: 'notes',
      ),
    ],
    chapters: [
      ChapterData(
        number: 1,
        title: 'Number',
        subtopics: ['Integers'],
      ),
      ChapterData(
        number: 2,
        title: 'Algebra',
        subtopics: ['Expressions'],
      ),
    ],
  ),
  SubjectData(
    code: '0610',
    name: 'Biology',
    level: 'IGCSE',
    resources: [
      SubjectResource(
        name: 'Syllabus PDF',
        url: 'https://www.cambridgeinternational.org/Images/698377-2026-syllabus.pdf',
        type: 'notes',
      ),
    ],
    chapters: [
      ChapterData(
        number: 1,
        title: 'Cells',
        subtopics: ['Cell structure'],
      ),
    ],
  ),
  SubjectData(
    code: '4MA1',
    name: 'Mathematics A',
    level: 'IGCSE',
    resources: [
      SubjectResource(
        name: 'Syllabus PDF',
        url: 'https://qualifications.pearson.com/content/dam/pdf/International%20GCSE/Mathematics%20A/2016/specification-and-sample-assessments/international-gcse-mathematics-a-specification.pdf',
        type: 'notes',
      ),
    ],
    chapters: [
      ChapterData(
        number: 1,
        title: 'Algebra',
        subtopics: ['Linear equations'],
      ),
    ],
  ),
  SubjectData(
    code: '9203',
    name: 'Mathematics',
    level: 'IGCSE',
    resources: [
      SubjectResource(
        name: 'Syllabus PDF',
        url: 'https://www.oxfordaqa.com/wp-content/uploads/2024/02/oxfordaqa-igcse-mathematics-specification.pdf',
        type: 'notes',
      ),
    ],
    chapters: [
    ],
  ),
];
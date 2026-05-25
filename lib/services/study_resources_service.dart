import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'curriculum_catalog_service.dart';

enum ResourceLevel { igcse, asALevel, oLevel }

enum ResourceType { notes, video, flashcards, examPrep, pastPapers, syllabus }

class SubjectResources {
  final String code;
  final String name;
  final String level;
  final String syllabus;
  final List<String> notesUrls;
  final List<String> videoUrls;
  final List<String> flashcardsUrls;
  final List<String> examPrepUrls;
  final List<String> pastPaperUrls;
  final List<String> youtubeChannels;
  final List<String> otherUrls;
  final List<Map<String, String>> chapters;

  const SubjectResources({
    required this.code,
    required this.name,
    required this.level,
    required this.syllabus,
    this.notesUrls = const [],
    this.videoUrls = const [],
    this.flashcardsUrls = const [],
    this.examPrepUrls = const [],
    this.pastPaperUrls = const [],
    this.youtubeChannels = const [],
    this.otherUrls = const [],
    this.chapters = const [],
  });
}

class ChapterInfo {
  final String number;
  final String title;

  const ChapterInfo({required this.number, required this.title});
}

class StudyResourcesService {
  static final StudyResourcesService _instance =
      StudyResourcesService._internal();
  factory StudyResourcesService() => _instance;
  StudyResourcesService._internal();

  static const Map<String, SubjectResources> _resources = {
    // ============ IGCSE BIOLOGY 0610 ============
    '0610': SubjectResources(
      code: '0610',
      name: 'Biology',
      level: 'IGCSE',
      syllabus: '0610 / 0970',
      notesUrls: [
        'https://znotes.org/caie/igcse/biology-0610',
        'https://www.savemyexams.com/igcse/biology/notes/',
        'https://www.physicsandmathstutor.com/igcse/biology/revision-notes/',
        'https://www.freeexamacademy.com/260-2/',
        'https://studybliss.github.io/studybliss-official/bioigcse.html',
        'https://www.blitznotes.org/ig/biology',
      ],
      videoUrls: [
        'https://www.cognito.org.uk/learn/igcse-biology',
        'https://www.sciencewithhazel.com/igcse-biology',
      ],
      youtubeChannels: [
        'Science with Hazel',
        'Cognito',
        'Amoeba Sisters',
        'Cambridge In 5 Minutes',
        'Biology with Zhan Xuan',
      ],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/biology/flashcards/',
        'https://quizlet.com/subject/igcse-biology/',
      ],
      examPrepUrls: [
        'https://www.savemyexams.com/igcse/biology/mock-exams/',
        'https://www.physicsandmathstutor.com/igcse/biology/questions-by-topic/',
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/biology/0610',
        'https://www.gceguide.com/IGCSE/Biology/0610',
      ],
      otherUrls: ['https://www.bbc.co.uk/bitesize/examspecs/z2nywsg'],
      chapters: [
        {'1': 'Characteristics and Classification'},
        {'2': 'Organisation of the Organism'},
        {'3': 'Movement In and Out of Cells'},
        {'4': 'Biological Molecules'},
        {'5': 'Nutrition'},
        {'6': 'Respiration'},
        {'7': 'Transport'},
        {'8': 'Coordination and Response'},
        {'9': 'Reproduction'},
        {'10': 'Genetics'},
        {'11': 'Biotechnology'},
        {'12': 'Ecology and Environment'},
      ],
    ),

    // ============ IGCSE CHEMISTRY 0620 ============
    '0620': SubjectResources(
      code: '0620',
      name: 'Chemistry',
      level: 'IGCSE',
      syllabus: '0620 / 0971',
      notesUrls: [
        'https://znotes.org/caie/igcse/chemistry-0620',
        'https://www.savemyexams.com/igcse/chemistry/notes/',
        'https://www.physicsandmathstutor.com/igcse/chemistry/revision-notes/',
        'https://sites.google.com/view/crushchemistry/igcse-cambridge-cie',
        'https://www.blitznotes.org/ig/chemistry',
      ],
      videoUrls: [
        'https://www.cognito.org.uk/learn/igcse-chemistry',
        'https://www.sciencewithhazel.com/igcse-chemistry',
      ],
      youtubeChannels: [
        'Science with Hazel',
        'Cognito',
        'Cambridge In 5 Minutes',
        'Crush Chemistry'
      ],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/chemistry/flashcards/',
        'https://quizlet.com/subject/igcse-chemistry/',
      ],
      examPrepUrls: ['https://www.savemyexams.com/igcse/chemistry/mock-exams/'],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/chemistry/0620',
        'https://www.gceguide.com/IGCSE/Chemistry/0620',
      ],
      otherUrls: ['https://www.bbc.co.uk/bitesize/examspecs/zyjmprj'],
      chapters: [
        {'1': 'States of Matter'},
        {'2': 'Atomic Structure'},
        {'3': 'Chemical Bonding'},
        {'4': 'Chemical Formulae and Equations'},
        {'5': 'The Periodic Table'},
        {'6': 'Electrolysis'},
        {'7': 'Chemical Measurements'},
        {'8': 'Acids, Bases and Salts'},
        {'9': 'Chemical Reactivity'},
        {'10': 'Metals'},
        {'11': 'Organic Chemistry'},
        {'12': 'Chemical Analysis'},
      ],
    ),

    // ============ IGCSE PHYSICS 0625 ============
    '0625': SubjectResources(
      code: '0625',
      name: 'Physics',
      level: 'IGCSE',
      syllabus: '0625 / 0972',
      notesUrls: [
        'https://znotes.org/caie/igcse/physics-0625',
        'https://www.savemyexams.com/igcse/physics/notes/',
        'https://www.physicsandmathstutor.com/igcse/physics/revision-notes/',
        'https://igcse.guru/igcse-physics-cambridge/',
        'https://www.blitznotes.org/ig/physics',
        'https://onlinemathlearning.com/igcse-physics.html',
      ],
      videoUrls: [
        'https://www.cognito.org.uk/learn/igcse-physics',
        'https://www.sciencewithhazel.com/igcse-physics',
      ],
      youtubeChannels: [
        'Science with Hazel',
        'Cognito',
        'Cambridge In 5 Minutes',
        'Physics Online'
      ],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/physics/flashcards/',
        'https://quizlet.com/subject/igcse-physics/',
      ],
      examPrepUrls: ['https://www.savemyexams.com/igcse/physics/mock-exams/'],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/physics/0625',
        'https://www.gceguide.com/IGCSE/Physics/0625',
      ],
      otherUrls: ['https://www.bbc.co.uk/bitesize/examspecs/z2nywsg'],
      chapters: [
        {'1': 'Motion, Forces and Energy'},
        {'2': 'Thermal Physics'},
        {'3': 'Waves'},
        {'4': 'Electricity and Magnetism'},
        {'5': 'Nuclear Physics'},
        {'6': 'Space Physics'},
      ],
    ),

    // ============ IGCSE MATHEMATICS 0580 ============
    '0580': SubjectResources(
      code: '0580',
      name: 'Mathematics',
      level: 'IGCSE',
      syllabus: '0580 / 0980',
      notesUrls: [
        'https://znotes.org/caie/igcse/mathematics-0580',
        'https://www.savemyexams.com/igcse/maths/notes/',
        'https://www.physicsandmathstutor.com/igcse/maths/revision-notes/',
        'https://www.twinseducation.com/igcse-maths/',
        'https://www.igcse.guru/igcse-mathematics-cambridge/',
      ],
      videoUrls: ['https://www.cognito.org.uk/learn/igcse-maths'],
      youtubeChannels: [
        'Cognito',
        'Maths with James',
        'Hegarty Maths',
        'ExamSolutions'
      ],
      flashcardsUrls: ['https://www.savemyexams.com/igcse/maths/flashcards/'],
      examPrepUrls: [
        'https://www.savemyexams.com/igcse/maths/mock-exams/',
        'https://www.physicsandmathstutor.com/igcse/maths/questions-by-topic/',
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/mathematics/0580',
        'https://www.gceguide.com/IGCSE/Maths/0580',
      ],
      otherUrls: ['https://www.bbc.co.uk/bitesize/examspecs/zpqcc94'],
      chapters: [
        {'1': 'Number'},
        {'2': 'Algebra and Graphs'},
        {'3': 'Geometry'},
        {'4': 'Length and Area'},
        {'5': 'Trigonometry'},
        {'6': 'Vectors and Transformation'},
        {'7': 'Probability'},
        {'8': 'Statistics'},
      ],
    ),

    // ============ IGCSE COMPUTER SCIENCE 0478 ============
    '0478': SubjectResources(
      code: '0478',
      name: 'Computer Science',
      level: 'IGCSE',
      syllabus: '0478 / 0984',
      notesUrls: [
        'https://znotes.org/caie/igcse/computer-science-0478',
        'https://www.savemyexams.com/igcse/computer-science/notes/',
        'https://www.physicsandmathstutor.com/igcse/computer-science/revision-notes/',
      ],
      videoUrls: ['https://www.samyc2002.com/'],
      youtubeChannels: ['Computer Science Tutor', 'Tech with Tim'],
      flashcardsUrls: ['https://quizlet.com/subject/igcse-computer-science/'],
      examPrepUrls: [
        'https://www.savemyexams.com/igcse/computer-science/mock-exams/'
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/computer-science/0478'
      ],
      chapters: [
        {'1': 'Data Representation'},
        {'2': 'Communication and Internet Technologies'},
        {'3': 'Hardware'},
        {'4': 'Software'},
        {'5': 'Use of Technology'},
        {'6': 'Algorithm'},
        {'7': 'Program Design'},
        {'8': 'Coding'},
        {'9': 'Databases'},
      ],
    ),

    // ============ IGCSE ECONOMICS 0455 ============
    '0455': SubjectResources(
      code: '0455',
      name: 'Economics',
      level: 'IGCSE',
      syllabus: '0455 / 0987',
      notesUrls: [
        'https://znotes.org/caie/igcse/economics-0455',
        'https://www.savemyexams.com/igcse/economics/notes/',
        'https://www.twinseducation.com/igcse-economics/',
      ],
      videoUrls: ['https://www.econplusdal.com/'],
      youtubeChannels: ['EconPlusDal', 'Teachingconomics'],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/economics/flashcards/'
      ],
      examPrepUrls: ['https://www.savemyexams.com/igcse/economics/mock-exams/'],
      pastPaperUrls: ['https://www.papacambridge.com/igcse/economics/0455'],
      chapters: [
        {'1': 'The Basic Economic Problem'},
        {'2': 'Allocation of Resources'},
        {'3': 'Microeconomics'},
        {'4': 'Macroeconomics'},
        {'5': 'Economic Development'},
        {'6': 'International Trade and Globalisation'},
      ],
    ),

    // ============ IGCSE BUSINESS STUDIES 0450 ============
    '0450': SubjectResources(
      code: '0450',
      name: 'Business Studies',
      level: 'IGCSE',
      syllabus: '0450 / 0986',
      notesUrls: [
        'https://znotes.org/caie/igcse/business-studies-0450',
        'https://www.savemyexams.com/igcse/business-studies/notes/',
        'https://www.twinseducation.com/igcse-business-studies/',
      ],
      youtubeChannels: ['Business Guy'],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/business-studies/flashcards/'
      ],
      examPrepUrls: [
        'https://www.savemyexams.com/igcse/business-studies/mock-exams/'
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/business-studies/0450'
      ],
      chapters: [
        {'1': 'Business Activity'},
        {'2': 'Human Resources'},
        {'3': 'Business Finance'},
        {'4': 'Marketing'},
        {'5': 'Business Operations'},
      ],
    ),

    // ============ IGCSE ACCOUNTING 0452 ============
    '0452': SubjectResources(
      code: '0452',
      name: 'Accounting',
      level: 'IGCSE',
      syllabus: '0452 / 0987',
      notesUrls: [
        'https://znotes.org/caie/igcse/accounting-0452',
        'https://www.savemyexams.com/igcse/accounting/notes/',
        'https://www.twinseducation.com/igcse-accounting/',
      ],
      youtubeChannels: ['Accounting to Learn'],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/accounting/flashcards/'
      ],
      examPrepUrls: [
        'https://www.savemyexams.com/igcse/accounting/mock-exams/'
      ],
      pastPaperUrls: ['https://www.papacambridge.com/igcse/accounting/0452'],
      chapters: [
        {'1': 'The Accounting System'},
        {'2': 'Ledger Accounts and Trial Balance'},
        {'3': 'Correction of Errors'},
        {'4': 'Cash and Bank'},
        {'5': 'Financial Statements'},
        {'6': 'Analysis and Interpretation'},
      ],
    ),

    // ============ IGCSE ENGLISH FIRST LANGUAGE 0500 ============
    '0500': SubjectResources(
      code: '0500',
      name: 'English - First Language',
      level: 'IGCSE',
      syllabus: '0500 / 0990',
      notesUrls: [
        'https://znotes.org/caie/igcse/english-first-language-0500',
        'https://www.savemyexams.com/igcse/english-first-language/notes/',
      ],
      youtubeChannels: ['Mrs Whitham English', 'Glowscorpio'],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/english-first-language/flashcards/'
      ],
      examPrepUrls: [
        'https://www.savemyexams.com/igcse/english-first-language/mock-exams/'
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/english-first-language/0500'
      ],
      chapters: [
        {'1': 'Reading'},
        {'2': 'Writing'},
        {'3': 'Directed Writing'},
      ],
    ),

    // ============ IGCSE ADDITIONAL MATHEMATICS 0606 ============
    '0606': SubjectResources(
      code: '0606',
      name: 'Additional Mathematics',
      level: 'IGCSE',
      syllabus: '0606',
      notesUrls: [
        'https://znotes.org/caie/igcse/additional-mathematics-0606',
        'https://www.savemyexams.com/igcse/further-maths/notes/',
      ],
      youtubeChannels: ['Cognito', 'ExamSolutions'],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/additional-mathematics/0606'
      ],
      chapters: [
        {'1': 'Quadratics'},
        {'2': 'Functions'},
        {'3': 'Coordinate Geometry'},
        {'4': 'Circular Measure'},
        {'5': 'Trigonometry'},
        {'6': 'Series'},
        {'7': 'Differentiation'},
        {'8': 'Integration'},
      ],
    ),

    // ============ IGCSE INFORMATION TECHNOLOGY 0413 ============
    '0413': SubjectResources(
      code: '0413',
      name: 'Information Technology',
      level: 'IGCSE',
      syllabus: '0413 / 0983',
      notesUrls: [
        'https://znotes.org/caie/igcse/information-technology-0413',
        'https://www.savemyexams.com/igcse/information-and-communication-technology/notes/',
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/information-technology/0413'
      ],
      chapters: [
        {'1': 'Types and Components'},
        {'2': 'Networks'},
        {'3': 'Document Production'},
        {'4': 'Data Manipulation'},
        {'5': 'Integration'},
      ],
    ),

    // ============ IGCSE GLOBAL PERSPECTIVES 0459 ============
    '0459': SubjectResources(
      code: '0459',
      name: 'Global Perspectives',
      level: 'IGCSE',
      syllabus: '0459 / 0987',
      notesUrls: ['https://znotes.org/caie/igcse/global-perspectives-0459'],
      pastPaperUrls: [
        'https://www.papacambridge.com/igcse/global-perspectives/0459'
      ],
      chapters: [
        {'1': 'The Individual'},
        {'2': 'The Community'},
        {'3': 'Society'},
        {'4': 'The World'},
      ],
    ),

    // ============ IGCSE GEOGRAPHY 0460 ============
    '0460': SubjectResources(
      code: '0460',
      name: 'Geography',
      level: 'IGCSE',
      syllabus: '0460 / 0976',
      notesUrls: [
        'https://znotes.org/caie/igcse/geography-0460',
        'https://www.savemyexams.com/igcse/geography/notes/',
      ],
      youtubeChannels: ['Geography Now'],
      flashcardsUrls: [
        'https://www.savemyexams.com/igcse/geography/flashcards/'
      ],
      pastPaperUrls: ['https://www.papacambridge.com/igcse/geography/0460'],
      chapters: [
        {'1': 'Population'},
        {'2': 'Migration'},
        {'3': 'Settlement'},
        {'4': 'Climate and Climate Change'},
        {'5': 'Ecosystems'},
        {'6': 'Tectonic Processes'},
        {'7': 'Rivers'},
        {'8': 'Coasts'},
      ],
    ),

    // ============ IGCSE HISTORY 0470 ============
    '0470': SubjectResources(
      code: '0470',
      name: 'History',
      level: 'IGCSE',
      syllabus: '0470 / 0977',
      notesUrls: [
        'https://znotes.org/caie/igcse/history-0470',
        'https://www.savemyexams.com/igcse/history/notes/',
      ],
      pastPaperUrls: ['https://www.papacambridge.com/igcse/history/0470'],
      chapters: [
        {'1': 'World History'},
        {'2': '20th Century'},
      ],
    ),

    // ============ IGCSE FRENCH 0520 ============
    '0520': SubjectResources(
      code: '0520',
      name: 'French',
      level: 'IGCSE',
      syllabus: '0520 / 0976',
      notesUrls: ['https://znotes.org/caie/igcse/french-0520'],
      youtubeChannels: ['French with Alexa'],
      pastPaperUrls: ['https://www.papacambridge.com/igcse/french/0520'],
      chapters: [
        {'1': 'Theme 1: Identity and Culture'},
        {'2': 'Theme 2: Local Area, Holiday, Travel'},
        {'3': 'Theme 3: Health'},
        {'4': 'Theme 4: Environment'},
        {'5': 'Theme 5: Career'},
      ],
    ),

    // ============ AS & A LEVEL MATHEMATICS 9709 ============
    '9709': SubjectResources(
      code: '9709',
      name: 'Mathematics',
      level: 'AS & A Level',
      syllabus: '9709',
      notesUrls: [
        'https://znotes.org/caie/as-a-level/mathematics-9709',
        'https://www.savemyexams.com/alevel/maths/notes/',
        'https://www.physicsandmathstutor.com/a-level-maths/revision-notes/',
      ],
      videoUrls: ['https://www.cognito.org.uk/learn/a-level-maths'],
      youtubeChannels: [
        'Cognito',
        'Hegarty Maths',
        'ExamSolutions',
        'TL Maths'
      ],
      flashcardsUrls: ['https://www.savemyexams.com/alevel/maths/flashcards/'],
      examPrepUrls: ['https://www.savemyexams.com/alevel/maths/mock-exams/'],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/mathematics/9709'],
      chapters: [
        {'1': 'Quadratics'},
        {'2': 'Functions'},
        {'3': 'Coordinate Geometry'},
        {'4': 'Circular Measure'},
        {'5': 'Trigonometry'},
        {'6': 'Series'},
        {'7': 'Differentiation'},
        {'8': 'Integration'},
      ],
    ),

    // ============ AS & A LEVEL BIOLOGY 9700 ============
    '9700': SubjectResources(
      code: '9700',
      name: 'Biology',
      level: 'AS & A Level',
      syllabus: '9700',
      notesUrls: [
        'https://znotes.org/caie/as-a-level/biology-9700',
        'https://www.savemyexams.com/alevel/biology/notes/',
        'https://www.physicsandmathstutor.com/a-level-biology/revision-notes/',
      ],
      videoUrls: ['https://www.cognito.org.uk/learn/a-level-biology'],
      youtubeChannels: ['Cognito', 'Science with Hazel', 'Amoeba Sisters'],
      flashcardsUrls: [
        'https://www.savemyexams.com/alevel/biology/flashcards/'
      ],
      examPrepUrls: ['https://www.savemyexams.com/alevel/biology/mock-exams/'],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/biology/9700'],
      chapters: [
        {'1': 'Cell Structure'},
        {'2': 'Biological Molecules'},
        {'3': 'Enzymes'},
        {'4': 'Cell Membranes'},
        {'5': 'The Mitotic Cycle'},
        {'6': 'Nucleic Acids'},
        {'7': 'Energy and Respiration'},
        {'8': 'Photosynthesis'},
        {'9': 'Homeostasis'},
        {'10': 'Inheritance'},
        {'11': 'Selection and Evolution'},
        {'12': 'Classification'},
      ],
    ),

    // ============ AS & A LEVEL CHEMISTRY 9701 ============
    '9701': SubjectResources(
      code: '9701',
      name: 'Chemistry',
      level: 'AS & A Level',
      syllabus: '9701',
      notesUrls: [
        'https://znotes.org/caie/as-a-level/chemistry-9701',
        'https://www.savemyexams.com/alevel/chemistry/notes/',
        'https://www.physicsandmathstutor.com/a-level-chemistry/revision-notes/',
      ],
      videoUrls: ['https://www.cognito.org.uk/learn/a-level-chemistry'],
      youtubeChannels: ['Cognito', 'Organic Chemistry Tutor'],
      flashcardsUrls: [
        'https://www.savemyexams.com/alevel/chemistry/flashcards/'
      ],
      examPrepUrls: [
        'https://www.savemyexams.com/alevel/chemistry/mock-exams/'
      ],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/chemistry/9701'],
      chapters: [
        {'1': 'Atomic Structure'},
        {'2': 'Chemical Bonding'},
        {'3': 'States of Matter'},
        {'4': 'Chemical Energetics'},
        {'5': 'Electrochemistry'},
        {'6': 'Equilibria'},
        {'7': 'Reaction Kinetics'},
        {'8': 'Inorganic Chemistry'},
        {'9': 'Organic Chemistry'},
      ],
    ),

    // ============ AS & A LEVEL PHYSICS 9702 ============
    '9702': SubjectResources(
      code: '9702',
      name: 'Physics',
      level: 'AS & A Level',
      syllabus: '9702',
      notesUrls: [
        'https://znotes.org/caie/as-a-level/physics-9702',
        'https://www.savemyexams.com/alevel/physics/notes/',
        'https://www.physicsandmathstutor.com/a-level-physics/revision-notes/',
      ],
      videoUrls: ['https://www.cognito.org.uk/learn/a-level-physics'],
      youtubeChannels: ['Cognito', 'Physics Online', 'WaldomScience'],
      flashcardsUrls: [
        'https://www.savemyexams.com/alevel/physics/flashcards/'
      ],
      examPrepUrls: ['https://www.savemyexams.com/alevel/physics/mock-exams/'],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/physics/9702'],
      chapters: [
        {'1': 'Kinematics'},
        {'2': 'Dynamics'},
        {'3': 'Forces'},
        {'4': 'Work, Energy and Power'},
        {'5': 'Deformation of Solids'},
        {'6': 'Waves'},
        {'7': 'Superposition'},
        {'8': 'Electric Fields'},
        {'9': 'Capacitors'},
        {'10': 'Magnetic Fields'},
        {'11': 'Electromagnetic Induction'},
        {'12': 'Nuclear Physics'},
      ],
    ),

    // ============ AS & A LEVEL COMPUTER SCIENCE 9608 ============
    '9608': SubjectResources(
      code: '9608',
      name: 'Computer Science',
      level: 'AS & A Level',
      syllabus: '9608',
      notesUrls: [
        'https://znotes.org/caie/as-a-level/computer-science-9608',
        'https://www.savemyexams.com/alevel/computer-science/notes/',
      ],
      youtubeChannels: ['Computer Science Tutor', 'Tech with Tim'],
      flashcardsUrls: [
        'https://www.savemyexams.com/alevel/computer-science/flashcards/'
      ],
      pastPaperUrls: [
        'https://www.papacambridge.com/a-level/computer-science/9608'
      ],
      chapters: [
        {'1': 'Information Representation'},
        {'2': 'Communication and Internet Technologies'},
        {'3': 'Hardware'},
        {'4': 'Logic Gates'},
        {'5': 'Processor Fundamentals'},
        {'6': 'Assembly Language'},
        {'7': 'Operating Systems'},
        {'8': 'Security and Encryption'},
        {'9': 'Databases'},
      ],
    ),

    // ============ AS & A LEVEL ECONOMICS 9708 ============
    '9708': SubjectResources(
      code: '9708',
      name: 'Economics',
      level: 'AS & A Level',
      syllabus: '9708',
      notesUrls: [
        'https://znotes.org/caie/as-a-level/economics-9708',
        'https://www.savemyexams.com/alevel/economics/notes/',
        'https://www.econplusdal.com/',
      ],
      youtubeChannels: ['EconPlusDal', 'Teachingconomics'],
      flashcardsUrls: [
        'https://www.savemyexams.com/alevel/economics/flashcards/'
      ],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/economics/9708'],
      chapters: [
        {'1': 'The Economic Problem'},
        {'2': 'Microeconomics'},
        {'3': 'Macroeconomics'},
      ],
    ),

    // ============ AS & A LEVEL ACCOUNTING 9706 ============
    '9706': SubjectResources(
      code: '9706',
      name: 'Accounting',
      level: 'AS & A Level',
      syllabus: '9706',
      notesUrls: ['https://znotes.org/caie/as-a-level/accounting-9706'],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/accounting/9706'],
      chapters: [
        {'1': 'Introduction to Accounting'},
        {'2': 'The Accounting System'},
        {'3': 'Preparation of Financial Statements'},
        {'4': 'Analysis and Interpretation'},
      ],
    ),

    // ============ AS & A LEVEL PSYCHOLOGY 9698 ============
    '9698': SubjectResources(
      code: '9698',
      name: 'Psychology',
      level: 'AS & A Level',
      syllabus: '9698',
      notesUrls: ['https://znotes.org/caie/as-a-level/psychology-9698'],
      youtubeChannels: ['Psychology in 10 minutes'],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/psychology/9698'],
      chapters: [
        {'1': 'Social Psychology'},
        {'2': 'Cognitive Psychology'},
        {'3': 'Development Psychology'},
        {'4': 'Biopsychology'},
        {'5': 'Research Methods'},
      ],
    ),

    // ============ AS & A LEVEL SOCIOLOGY 9699 ============
    '9699': SubjectResources(
      code: '9699',
      name: 'Sociology',
      level: 'AS & A Level',
      syllabus: '9699',
      notesUrls: ['https://znotes.org/caie/as-a-level/sociology-9699'],
      pastPaperUrls: ['https://www.papacambridge.com/a-level/sociology/9699'],
      chapters: [
        {'1': 'Sociology of Education'},
        {'2': 'Social Stratification'},
        {'3': 'Work and Leisure'},
        {'4': 'Theorizing Social Change'},
      ],
    ),

    // ============ O LEVEL BIOLOGY 5090 ============
    '5090': SubjectResources(
      code: '5090',
      name: 'Biology',
      level: 'O Level',
      syllabus: '5090',
      notesUrls: ['https://znotes.org/caie/olevel/biology-5090'],
      chapters: [
        {'1': 'Cells'},
        {'2': 'Nutrition'},
        {'3': 'Respiration'},
        {'4': 'Gas Exchange'},
        {'5': 'Transport'},
        {'6': 'Homeostasis'},
        {'7': 'Reproduction'},
      ],
    ),

    // ============ O LEVEL CHEMISTRY 5070 ============
    '5070': SubjectResources(
      code: '5070',
      name: 'Chemistry',
      level: 'O Level',
      syllabus: '5070',
      notesUrls: ['https://znotes.org/caie/olevel/chemistry-5070'],
      chapters: [
        {'1': 'States of Matter'},
        {'2': 'Atomic Structure'},
        {'3': 'Chemical Bonding'},
        {'4': 'Chemical Formulae'},
        {'5': 'The Periodic Table'},
        {'6': 'Acids and Bases'},
        {'7': 'Metals'},
        {'8': 'Organic Chemistry'},
      ],
    ),

    // ============ O LEVEL PHYSICS 5054 ============
    '5054': SubjectResources(
      code: '5054',
      name: 'Physics',
      level: 'O Level',
      syllabus: '5054',
      notesUrls: ['https://znotes.org/caie/olevel/physics-5054'],
      chapters: [
        {'1': 'Measurement'},
        {'2': 'Kinematics'},
        {'3': 'Dynamics'},
        {'4': 'Force'},
        {'5': 'Energy'},
        {'6': 'Thermal Physics'},
        {'7': 'Light'},
        {'8': 'Electricity'},
        {'9': 'Magnetism'},
        {'10': 'Radioactivity'},
      ],
    ),

    // ============ O LEVEL MATHEMATICS 4024 ============
    '4024': SubjectResources(
      code: '4024',
      name: 'Mathematics',
      level: 'O Level',
      syllabus: '4024',
      notesUrls: ['https://znotes.org/caie/olevel/mathematics-4024'],
      chapters: [
        {'1': 'Number'},
        {'2': 'Algebra'},
        {'3': 'Geometry'},
        {'4': 'Mensuration'},
        {'5': 'Trigonometry'},
        {'6': 'Statistics'},
      ],
    ),
};

  SubjectResources? getResources(String code) => _resources[code];

  Future<SubjectResources?> getResourcesForSubject(String subject) async {
    if (_resources.containsKey(subject)) {
      return _resources[subject];
    }
    final code = await CurriculumCatalogService.instance.getSubjectCode(subject);
    if (code != null && _resources.containsKey(code)) {
      return _resources[code];
    }
    return null;
  }

  List<String> getAllCodes() => _resources.keys.toList();

  List<String> getCodesByLevel(String level) {
    return _resources.entries
        .where((e) => e.value.level.toLowerCase().contains(level.toLowerCase()))
        .map((e) => e.key)
        .toList();
  }

  Future<List<ChapterInfo>> getChaptersForSubject(String subject) async {
    final resources = await getResourcesForSubject(subject);
    if (resources == null) return [];

    return resources.chapters.map((c) {
      final number = c.keys.first;
      final title = c[number]!;
      return ChapterInfo(number: number, title: title);
    }).toList();
  }

  List<ChapterInfo> getChapters(String code) {
    final resources = _resources[code];
    if (resources == null) return [];

    return resources.chapters.map((c) {
      final number = c.keys.first;
      final title = c[number]!;
      return ChapterInfo(number: number, title: title);
    }).toList();
  }

  Future<void> openResource(String url) async {}
}

final studyResourcesServiceProvider = Provider<StudyResourcesService>((ref) {
  return StudyResourcesService();
});

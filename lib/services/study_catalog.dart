// lib/services/study_catalog.dart
import 'dart:convert';
import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'curriculum_catalog_service.dart';
import 'firestore_service.dart';

class StudyCatalog {
  static const _key = 'study_catalog';
  static const Map<String, List<String>> _boardSubjects = {
    'IGCSE': [
      'Mathematics',
      'Further Mathematics',
      'Additional Mathematics',
      'Statistics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Information and Communication Technology',
      'Information Technology',
      'Economics',
      'Business Studies',
      'Accounting',
      'Financial Accounting',
      'English Language',
      'English Literature',
      'First Language English',
      'English as a Second Language',
      'English - First Language',
      'English - Second Language',
      'History',
      'Geography',
      'Travel and Tourism',
      'Environmental Management',
      'Marine Science',
      'Cricket',
      'Psychology',
      'Sociology',
      'Law',
      'Philosophy',
      'Religious Studies',
      'Islamic Studies',
      'Pakistan Studies',
      'Bangladesh Studies',
      'Sri Lanka Studies',
      'India Studies',
      'Nepal Studies',
      'Malaysian Studies',
      'Singapore Studies',
      'Media Studies',
      'Film Studies',
      'Photography',
      'Art and Design',
      'Design and Technology',
      'Product Design',
      'Textiles',
      'Food and Nutrition',
      'Food Technology',
      ' Hospitality',
      'Child Development',
      'Health and Social Care',
      'Music',
      'Music Practice',
      'Drama',
      'Theatre Studies',
      'Dance',
      'Physical Education',
      'Sports Science',
      'French',
      'Spanish',
      'German',
      'Italian',
      'Portuguese',
      'Dutch',
      'Swedish',
      'Polish',
      'Russian',
      'Japanese',
      'Chinese',
      'Korean',
      'Hindi',
      'Urdu',
      'Bengali',
      'Tamil',
      'Telugu',
      'Malayalam',
      'Marathi',
      'Punjabi',
      'Gujarati',
      'Arabic',
      'Hebrew',
      'Thai',
      'Vietnamese',
      'Indonesian',
      'Malay',
      'Burmese',
      'Tagalog',
      'Khmer',
      ' Lao',
      'Nepali',
      'Sinhala',
      'Afrikaans',
      'Zulu',
      'Xhosa',
      'Swahili',
      'Amharic',
      'Turkish',
      'Greek',
      'Latin',
      'Sanskrit',
      'Classical Studies',
      'Ancient History',
      'Modern History',
      'European History',
      'World History',
      'Economics',
      'Commerce',
      'Office Practice',
      'Shorthand',
      'Typing',
      'Fashion and Textiles',
      'Building Construction',
      'Engineering',
      'Electricity and Electronics',
      'Metalwork',
      'Woodwork',
      'Technical Drawing',
      'Automotive',
      'Motor Vehicle Mechanics',
      'Air Conditioning and Refrigeration',
      'Plumbing',
      'Brickwork',
      'Carpentry and Joinery',
      'Painting and Decorating',
      'Catering and Hospitality',
      'Travel and Tourism',
      'Hotel and Catering',
      'Business Studies',
      'Business and Enterprise',
      'Enterprise and Marketing',
      'Accounting and Finance',
      'Applied Business',
      'Travel and Tourism',
      ' Leisure and Recreation',
      ' Leisur Studies',
      'Public Services',
      'Civic Education',
      'Citizenship',
      'Personal and Social Education',
      'Social Sciences',
      'Global Perspectives',
      'Thinking Skills',
      'Research Skills',
      'Project Work',
      'Cambridge International Diploma',
      'Cambridge International Certificate',
      'Digital Literacy',
      'Computer Applications',
      'Coding',
      'Robotics',
      'Artificial Intelligence',
      'Cyber Security',
      'Networking',
      'Database Management',
      'Software Development',
      'Web Development',
      'Game Design',
      'Animation',
      'Graphic Design',
      'Fashion Design',
      'Interior Design',
      'Landscape Design',
      'Architecture',
      'Agricultural Science',
      'Crop Science',
      'Horticulture',
      'Animal Husbandry',
      'Veterinary Science',
      'Forestry',
      'Fisheries',
      'Marine Biology',
      'Oceanography',
      'Geology',
      'Astronomy',
      'Earth Science',
      'Environmental Science',
      'Biochemistry',
      'Biotechnology',
      'Microbiology',
      'Genetics',
      'Zoology',
      'Botany',
      'Ecology',
      'Soil Science',
      'Hydrology',
      'Meteorology',
      ' Climatology',
    ],
    'O Level': [
      'Mathematics',
      'Additional Mathematics',
      'Elementary Mathematics',
      'Extended Mathematics',
      'Statistics',
      'Physics',
      'Chemistry',
      'Biology',
      'Science',
      'Combined Science',
      'Double Award Science',
      'Physical Science',
      'Computer Science',
      'Information Technology',
      'Computing',
      'Office Technology',
      'Economics',
      'Commerce',
      'Business Studies',
      'Accounting',
      'Principles of Accounting',
      'Book-keeping',
      'English Language',
      'English Literature',
      'English as a Second Language',
      'First Language English',
      'Second Language English',
      'History',
      'Geography',
      'Environmental Management',
      'Travel and Tourism',
      'Pakistan Studies',
      'Bangladesh Studies',
      'Sri Lanka Studies',
      'India and Pakistan (History)',
      'World History',
      'Social Studies',
      'Civics',
      'Religious Studies',
      'Islamic Studies',
      ' Islamiyat',
      'Christianity',
      'Hinduism',
      'Buddhism',
      'Sikhism',
      'Judaism',
      'Zoroastrianism',
      'Moral Education',
      'Ethics',
      'Psychology',
      'Sociology',
      'Arts',
      'Art and Design',
      'Fashion and Textiles',
      'Food and Nutrition',
      'Food Technology',
      'Home Economics',
      'Music',
      'Physical Education',
      'Health Education',
      'French',
      'Spanish',
      'German',
      'Arabic',
      'Hindi',
      'Urdu',
      'Bengali',
      'Tamil',
      'Telugu',
      'Malayalam',
      'Punjabi',
      'Gujarati',
      'Marathi',
      'Nepali',
      'Sinhala',
      'Burmese',
      'Thai',
      'Indonesian',
      'Malay',
      'Vietnamese',
      'Tagalog',
      'Khmer',
      'Chinese',
      'Japanese',
      'Korean',
      'Russian',
      'Portuguese',
      'Italian',
      'Dutch',
      'Greek',
      'Latin',
      'Hebrew',
      'Sanskrit',
      'Turkish',
      'Swedish',
      'Polish',
      'Czech',
      'Hungarian',
      'Finnish',
    ],
    'A Level': [
      'Mathematics',
      'Further Mathematics',
      'Additional Mathematics',
      'Pure Mathematics',
      'Applied Mathematics',
      'Statistics',
      'Mechanics',
      'Discrete Mathematics',
      'Decision Mathematics',
      'Numerical Methods',
      'Physics',
      'Advanced Physics',
      'Applied Physics',
      'Chemistry',
      'Advanced Chemistry',
      'Applied Chemistry',
      'Biology',
      'Advanced Biology',
      'Applied Biology',
      'Computer Science',
      'Computing',
      'Information Technology',
      'Software Development',
      'Database Systems',
      'Networks',
      'Artificial Intelligence',
      'Economics',
      'Advanced Economics',
      'Applied Economics',
      'Development Economics',
      'International Economics',
      'Quantitative Economics',
      'Business',
      'Business Studies',
      'Business Management',
      'Business Accounting',
      'Accounting',
      'Financial Accounting',
      'Management Accounting',
      'Taxation',
      'Auditing',
      'Finance',
      'Financial Management',
      'Investment',
      'Banking',
      'Insurance',
      'English Language',
      'English Language and Literature',
      'English Literature',
      'English as a Global Language',
      'Creative Writing',
      'Literature',
      'Poetry',
      'Prose',
      'Drama',
      'Shakespeare',
      'Classical Literature',
      'Comparative Literature',
      'World Literature',
      'History',
      'Modern History',
      'Contemporary History',
      'Ancient History',
      'Medieval History',
      'Early Modern History',
      'Late Modern History',
      'European History',
      'British History',
      'American History',
      'Asian History',
      'African History',
      'World History',
      'International History',
      'Political History',
      'Economic History',
      'Social History',
      'Cultural History',
      'Military History',
      'Geography',
      'Physical Geography',
      'Human Geography',
      'Regional Geography',
      'Urban Geography',
      'Rural Geography',
      'Economic Geography',
      'Population Geography',
      'Climatology',
      'Geomorphology',
      'Biogeography',
      'Oceanography',
      'Cartography',
      'Geographical Information Systems',
      'Environmental Geography',
      'Development Geography',
      'Psychology',
      'Abnormal Psychology',
      'Cognitive Psychology',
      'Developmental Psychology',
      'Social Psychology',
      'Educational Psychology',
      'Clinical Psychology',
      'Counselling Psychology',
      'Research Psychology',
      'Sociology',
      'Theoretical Sociology',
      'Applied Sociology',
      'Social Policy',
      'Social Anthropology',
      'Cultural Studies',
      'Gender Studies',
      'Criminology',
      'Deviance',
      'Social Stratification',
      'Philosophy',
      'Ethics',
      'Metaphysics',
      'Epistemology',
      'Logic',
      'Political Philosophy',
      'Philosophy of Mind',
      'Philosophy of Religion',
      'Aesthetics',
      'Jurisprudence',
      'Law',
      'Criminal Law',
      'Civil Law',
      'International Law',
      'Constitutional Law',
      'Contract Law',
      'Tort Law',
      'Property Law',
      'Company Law',
      'Tax Law',
      'Labour Law',
      'Administrative Law',
      'Human Rights Law',
      'Medical Law',
      'Theology',
      'Religious Studies',
      'Biblical Studies',
      'Islamic Studies',
      'Hindu Studies',
      'Buddhist Studies',
      'Comparative Religion',
      'World Religions',
      'Global Politics',
      'International Relations',
      'International Politics',
      'Political Science',
      'Political Theory',
      'Comparative Politics',
      'Public Policy',
      'Public Administration',
      'Governance',
      'Diplomacy',
      'Strategic Studies',
      'Security Studies',
      'Peace Studies',
      'Development Studies',
      'International Development',
      'Human Development',
      'Sustainable Development',
      'Digital Society',
      'Media Studies',
      'Communication Studies',
      'Journalism',
      'Broadcasting',
      'Film Studies',
      'Television Studies',
      'Media Production',
      'Media Management',
      'Environmental Systems',
      'Environmental Science',
      'Environmental Studies',
      'Ecology',
      'Conservation',
      'Biodiversity',
      'Wildlife',
      'Forestry',
      'Marine Science',
      'Ocean Science',
      'Atmospheric Science',
      'Geology',
      'Geophysics',
      'Geochemistry',
      'Astronomy',
      'Astrophysics',
      'Marine Biology',
      'Sports Science',
      'Sports Studies',
      'Sports Psychology',
      'Sports Management',
      'Exercise Science',
      'Physical Education',
      'Coaching',
      'Sports Medicine',
      'Physiotherapy',
      'Nutrition',
      'Dietetics',
      'Health Science',
      'Public Health',
      'Epidemiology',
      'Biostatistics',
      'Pharmacology',
      'Toxicology',
      'Neuroscience',
      'Biochemistry',
      'Biotechnology',
      'Genetics',
      'Molecular Biology',
      'Microbiology',
      'Cell Biology',
      'Plant Biology',
      'Animal Biology',
      'Zoology',
      'Entomology',
      'Ornithology',
      'Herpetology',
      'Mammalogy',
      'Ichthyology',
      'Primatology',
      'Anthropology',
      'Physical Anthropology',
      'Archaeology',
      'Paleontology',
      'Museum Studies',
      'Heritage Studies',
      'Conservation Studies',
      'Restoration',
      'Art History',
      'Visual Arts',
      'Fine Arts',
      'Art and Design',
      'Fashion Design',
      'Graphic Design',
      'Interior Design',
      'Product Design',
      'Industrial Design',
      'Textile Design',
      'Ceramic Design',
      'Jewelry Design',
      'Photography',
      'Sculpture',
      'Painting',
      'Drawing',
      'Printmaking',
      'Music',
      'Music Theory',
      'Music Performance',
      'Music Composition',
      'Music Technology',
      'Musicology',
      'Ethnomusicology',
      'Jazz Studies',
      'Popular Music',
      'Classical Music',
      'Film Music',
      'Theatre',
      'Drama',
      'Acting',
      'Directing',
      'Stagecraft',
      'Production Design',
      'Costume Design',
      'Lighting Design',
      'Sound Design',
      'Screenwriting',
      'Playwriting',
      'Dance',
      'Choreography',
      'Ballet',
      'Contemporary Dance',
      'Jazz Dance',
      'Traditional Dance',
      'French',
      'Spanish',
      'German',
      'Italian',
      'Portuguese',
      'Dutch',
      'Swedish',
      'Danish',
      'Norwegian',
      'Finnish',
      'Icelandic',
      'Polish',
      'Czech',
      'Slovak',
      'Hungarian',
      'Romanian',
      'Bulgarian',
      'Greek',
      'Turkish',
      'Arabic',
      'Hebrew',
      'Persian',
      'Urdu',
      'Hindi',
      'Bengali',
      'Tamil',
      'Telugu',
      'Malayalam',
      'Punjabi',
      'Marathi',
      'Gujarati',
      'Nepali',
      'Sinhala',
      'Burmese',
      'Thai',
      'Vietnamese',
      'Indonesian',
      'Malay',
      'Chinese',
      'Japanese',
      'Korean',
      'Mongolian',
      'Tibetan',
      'Sanskrit',
      'Pali',
      'Latin',
      'Classical Greek',
      'Classical Hebrew',
      'Old English',
      'Old Norse',
      'Celtic Languages',
      'Welsh',
      'Irish Gaelic',
      'Scots Gaelic',
      'Breton',
      'Cornish',
      'Extended Essay',
      'Theory of Knowledge',
      'Research Project',
      'Independent Research',
      'Capstone Project',
      'Thesis',
      'Dissertation',
      'Portfolio',
      'Work Experience',
      'Internship',
      'Community Service',
      'Volunteering',
      'Leadership',
      'Entrepreneurship',
      'Innovation',
      'Design Thinking',
      'Project Management',
      'Time Management',
      'Critical Thinking',
      'Problem Solving',
      'Decision Making',
      'Systems Thinking',
      'Quantitative Reasoning',
      'Qualitative Analysis',
      'Data Analysis',
      'Statistical Analysis',
      'Mathematical Modelling',
      'Operations Research',
      'Game Theory',
      'Information Systems',
      'Business Information Systems',
      'Knowledge Management',
      'Innovation Management',
      'Technology Management',
      'Operations Management',
      'Supply Chain Management',
      'Logistics',
      'Procurement',
      'Quality Management',
      'Production Management',
      'Risk Management',
      'Crisis Management',
      'Emergency Management',
      'Disaster Management',
      'Security Management',
      'Asset Management',
      'Facility Management',
      'Property Management',
      'Real Estate',
      'Construction Management',
      'Urban Planning',
      'Regional Planning',
      'Environmental Planning',
      'Transportation Planning',
      'Housing Policy',
      'Social Policy',
      'Education Policy',
      'Health Policy',
      'Economic Policy',
      'Fiscal Policy',
      'Monetary Policy',
      'Trade Policy',
      'Industrial Policy',
      'Agricultural Policy',
      'Energy Policy',
      'Environmental Policy',
      'Foreign Policy',
      'Defence Policy',
      'Security Policy',
      'Immigration Policy',
      'Citizenship Policy',
      'Human Rights',
      'International Law',
      'International Relations Theory',
      'International Organizations',
      'International Political Economy',
      'International Security',
      'Humanitarian Intervention',
      'Conflict Resolution',
      'Peacebuilding',
      'Reconciliation',
      'Post-Conflict Reconstruction',
      'Transitional Justice',
      'Rule of Law',
      'Governance',
      'Transparency',
      'Accountability',
      'Corruption',
      'Anti-Corruption',
      'Ethics',
      'Corporate Ethics',
      'Professional Ethics',
      'Bioethics',
      'Media Ethics',
      'Journalism Ethics',
      'Research Ethics',
    ],
  };

  static List<String> get allSubjects {
    return _boardSubjects.values.expand((list) => list).toList();
  }

  static String normalizeSubject(String subject) {
    final trimmed = subject.trim();
    switch (trimmed.toLowerCase()) {
      case 'math':
      case 'maths':
      case 'mathematics':
        return 'Mathematics';
      case 'further math':
      case 'further maths':
      case 'further mathematics':
        return 'Further Mathematics';
      case 'cs':
      case 'comp sci':
      case 'computer science':
        return 'Computer Science';
      case 'business management':
        return 'Business Management';
      case 'business':
        return 'Business';
      case 'accounting':
      case 'accounts':
        return 'Accounting';
      case 'english':
      case 'eng lang':
      case 'english language':
      case 'first language english':
        return 'First Language English';
      case 'esl':
      case 'english as a second language':
      case 'english as second language':
        return 'English as a Second Language';
      case 'english literature':
        return 'English Literature';
      case 'history':
        return 'History';
      case 'geography':
        return 'Geography';
      case 'psychology':
        return 'Psychology';
      case 'sociology':
        return 'Sociology';
      case 'religious studies':
      case 'rs':
      case 'religion':
        return 'Religious Studies';
      case 'physical education':
      case 'pe':
        return 'Physical Education';
      case 'art and design':
      case 'art':
        return 'Art and Design';
      case 'design and technology':
      case 'dt':
      case 'design technology':
        return 'Design and Technology';
      case 'drama':
        return 'Drama';
      case 'music':
        return 'Music';
      case 'media studies':
      case 'media':
        return 'Media Studies';
      case 'french':
        return 'French';
      case 'spanish':
        return 'Spanish';
      case 'german':
        return 'German';
      case 'additional mathematics':
      case 'add math':
        return 'Additional Mathematics';
      case 'statistics':
        return 'Statistics';
      case 'applied ict':
      case 'ict':
      case 'information technology':
        return 'Applied ICT';
      case 'food and nutrition':
      case 'food tech':
      case 'food technology':
        return 'Food and Nutrition';
      case 'environmental management':
        return 'Environmental Management';
      case 'travel and tourism':
      case 'travel tourism':
        return 'Travel and Tourism';
      case 'law':
        return 'Law';
      case 'philosophy':
        return 'Philosophy';
      case 'politics':
        return 'Politics';
      case 'global politics':
        return 'Global Politics';
      case 'marine science':
      case 'marine biology':
        return 'Marine Science';
      case 'physics':
        return 'Physics';
      case 'chemistry':
        return 'Chemistry';
      case 'biology':
        return 'Biology';
      case 'economics':
        return 'Economics';
      default:
        return trimmed;
    }
  }

  static List<String> normalizeSubjects(Iterable<dynamic> subjects) {
    return subjects
        .map((subject) => normalizeSubject(subject.toString()))
        .where((subject) => subject.trim().isNotEmpty)
        .toSet()
        .toList()
      ..sort();
  }

  static List<String> extractSubjects(dynamic raw) {
    if (raw is List) {
      return normalizeSubjects(raw);
    }
    if (raw is Map) {
      return normalizeSubjects(raw.keys);
    }
    return const <String>[];
  }

  static List<String> subjectsForBoard(String board) {
    debugPrint('StudyCatalog.subjectsForBoard called with: "$board"');
    if (board.isEmpty) {
      final result = _boardSubjects.values
          .expand((subjects) => subjects)
          .toSet()
          .toList()
        ..sort();
      debugPrint('Empty board, returning ${result.length} subjects');
      return result;
    }
    final exact = _boardSubjects[board];
    debugPrint('Exact match for "$board": ${exact != null}');
    if (exact != null) {
      return List<String>.from(exact);
    }

    final lower = board.toLowerCase();
    debugPrint('Lower case: "$lower"');
    if (lower.contains('igcse') || lower.contains('cambridge')) {
      final subjects = _boardSubjects['IGCSE'];
      debugPrint('IGCSE match: ${subjects?.length ?? 0}');
      return subjects != null ? List<String>.from(subjects) : [];
    }
    if (lower.contains('o level') ||
        lower.contains('olevel') ||
        lower.contains('o-level')) {
      final subjects = _boardSubjects['O Level'];
      debugPrint('O Level match: ${subjects?.length ?? 0}');
      return subjects != null ? List<String>.from(subjects) : [];
    }
    if (lower.contains('a level') ||
        lower.contains('alevel') ||
        lower.contains('as level')) {
      final subjects = _boardSubjects['A Level'];
      debugPrint('A Level match: ${subjects?.length ?? 0}');
      return subjects != null ? List<String>.from(subjects) : [];
    }
    final result = _boardSubjects.values
        .expand((subjects) => subjects)
        .toSet()
        .toList()
      ..sort();
    debugPrint('Fallback returning ${result.length} subjects');
    return result;
  }

  Future<Map<String, List<String>>> load() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_key);
    if (raw == null || raw.isEmpty) return {};
    try {
      final map = jsonDecode(raw) as Map<String, dynamic>;
      return map.map((k, v) => MapEntry(
            k,
            (v as List).map((e) => e.toString()).toList(),
          ));
    } catch (_) {
      return {};
    }
  }

  Future<void> save(Map<String, List<String>> data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(data));
    await prefs.setInt('${_key}_last_updated', DateTime.now().millisecondsSinceEpoch);
  }

  Future<void> addSubject(String subject) async {
    final data = await load();
    data.putIfAbsent(normalizeSubject(subject), () => <String>[]);
    await save(data);
  }

  Future<void> replaceSubjects(List<String> subjects) async {
    final normalized = normalizeSubjects(subjects);
    final existing = await load();
    final updated = <String, List<String>>{};
    for (final subject in normalized) {
      updated[subject] =
          List<String>.from(existing[subject] ?? const <String>[]);
    }
    await save(updated);
  }

  Future<void> addChapters(String subject, List<String> chapters) async {
    final data = await load();
    final list = data.putIfAbsent(subject, () => <String>[]);
    for (final c in chapters) {
      if (!list.contains(c)) list.add(c);
    }
    await save(data);
  }

  Future<void> removeChapter(String subject, String chapter) async {
    final data = await load();
    final list = data[subject];
    if (list == null) return;
    list.removeWhere((item) => item == chapter);
    await save(data);
  }

  Future<List<String>> getChapters(String subject) async {
    final data = await load();
    return data[subject] ?? [];
  }

  Future<List<String>> getSubjects() async {
    final data = await load();
    return data.keys.toList()..sort();
  }

  Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }

  Future<Map<String, List<String>>> syncWithFirestore(String uid) async {
    final localData = await load();
    final prefs = await SharedPreferences.getInstance();
    final localUpdated = prefs.getInt('${_key}_last_updated') ?? 0;

    try {
      final doc = await AxonPaths.privateUserDoc(uid).get();
      if (!doc.exists || doc.data() == null) {
        return localData;
      }
      final data = doc.data()!;
      final firestoreUpdated = data['study_catalog_last_updated'] as int? ?? 0;

      if (localUpdated > firestoreUpdated && localUpdated > 0) {
        // Local is newer, push to firestore
        await saveToFirestore(uid);
        return localData;
      } else if (firestoreUpdated > localUpdated && firestoreUpdated > 0) {
        // Firestore is newer, pull from firestore completely
        final merged = <String, List<String>>{};
        
        void mergeCatalogMap(Map<dynamic, dynamic> rawMap) {
          for (final entry in rawMap.entries) {
            final subject = normalizeSubject(entry.key.toString());
            final chapters = (entry.value as List?)
                    ?.map((e) => e.toString())
                    .where((c) => c.isNotEmpty)
                    .toList() ??
                const <String>[];
            final existing = merged[subject] ?? const <String>[];
            final combined = {...existing, ...chapters}.toList()..sort();
            merged[subject] = combined;
          }
        }

        final studyCatalogData = data['study_catalog'];
        if (studyCatalogData is Map) {
          mergeCatalogMap(studyCatalogData);
        }

        final firestoreSubjects = data['subjects'];
        if (firestoreSubjects is Map) {
          mergeCatalogMap(firestoreSubjects);
        } else if (firestoreSubjects is List) {
          final selectedSubjects = normalizeSubjects(firestoreSubjects).toSet();
          merged.removeWhere((subject, _) => !selectedSubjects.contains(subject));
          for (final subject in selectedSubjects) {
            merged.putIfAbsent(subject, () => <String>[]);
          }
        }

        // Save locally without bumping timestamp past firestore's
        await prefs.setString(_key, jsonEncode(merged));
        await prefs.setInt('${_key}_last_updated', firestoreUpdated);
        return merged;
      }

      // Fallback: timestamps are equal or missing, do the union merge
      final merged = Map<String, List<String>>.from(localData);

      void mergeCatalogMap(Map<dynamic, dynamic> rawMap) {
        for (final entry in rawMap.entries) {
          final subject = normalizeSubject(entry.key.toString());
          final chapters = (entry.value as List?)
                  ?.map((e) => e.toString())
                  .where((c) => c.isNotEmpty)
                  .toList() ??
              const <String>[];
          final existing = merged[subject] ?? const <String>[];
          final combined = {...existing, ...chapters}.toList()..sort();
          merged[subject] = combined;
        }
      }

      final studyCatalogData = data['study_catalog'];
      if (studyCatalogData is Map) {
        mergeCatalogMap(studyCatalogData);
      }

      final firestoreSubjects = data['subjects'];
      if (firestoreSubjects is Map) {
        mergeCatalogMap(firestoreSubjects);
      } else if (firestoreSubjects is List) {
        final selectedSubjects = normalizeSubjects(firestoreSubjects).toSet();
        merged.removeWhere((subject, _) => !selectedSubjects.contains(subject));
        for (final subject in selectedSubjects) {
          merged.putIfAbsent(subject, () => <String>[]);
        }
      }

      await save(merged);
      await saveToFirestore(uid); // Push the union merge result to Firestore
      return merged;
    } catch (_) {
      return localData;
    }
  }

  Future<void> saveToFirestore(String uid) async {
    final data = await load();
    final prefs = await SharedPreferences.getInstance();
    final localUpdated = prefs.getInt('${_key}_last_updated') ?? DateTime.now().millisecondsSinceEpoch;
    try {
      await AxonFirestore.instance
          .collection(AxonCollections.usersPrivate)
          .doc(uid)
          .set({
            'study_catalog': data,
            'study_catalog_last_updated': localUpdated,
          }, SetOptions(merge: true));
    } catch (_) {}
  }

  Future<Map<String, List<String>>> hydrateWithCurriculum({
    required String board,
    required Iterable<String> subjects,
  }) async {
    final result = <String, List<String>>{};
    for (final rawSubject in subjects) {
      final subject = normalizeSubject(rawSubject);
      final chapters = await CurriculumCatalogService.instance.chapterTitles(
        board: board,
        subject: subject,
      );
      result[subject] = chapters;
    }
    return result;
  }

  static const String _resourcesKey = 'scraped_resources';
  static const String _aariPath =
      'lib/scrapper/aari_project/aari/output/batch_upload.json';

  Future<Map<String, Map<String, List<String>>>> loadScrapedResources() async {
    return StudyCatalog()._loadScrapedResourcesInternal();
  }

  Future<Map<String, Map<String, List<String>>>>
      _loadScrapedResourcesInternal() async {
    final prefs = await SharedPreferences.getInstance();
    final cached = prefs.getString(_resourcesKey);

    if (cached != null) {
      try {
        final decoded = jsonDecode(cached);
        return Map<String, Map<String, List<String>>>.from(
          (decoded as Map).map(
            (key, qualMap) => MapEntry(
              key.toString(),
              Map<String, List<String>>.from(
                (qualMap as Map).map(
                  (k, v) => MapEntry(k.toString(), (v as List).cast<String>()),
                ),
              ),
            ),
          ),
        );
      } catch (_) {}
    }

    try {
      final file = File(_aariPath);
      if (await file.exists()) {
        final content = await file.readAsString();
        final data = jsonDecode(content);
        final resources = data['resources'] as List;

        final Map<String, Map<String, Set<String>>> subjectChapters = {};

        for (final resource in resources) {
          final subject = resource['subject_name'] as String?;
          final qualification = resource['qualification'] as String?;
          final resourceType = resource['resource_type'] as String?;
          final year = resource['year'] as int?;
          final session = resource['session'] as String?;

          if (subject == null) continue;

          final normalized = normalizeSubject(subject);
          if (!subjectChapters.containsKey(normalized)) {
            subjectChapters[normalized] = {};
          }

          final qualKey = qualification ?? 'IGCSE';
          if (!subjectChapters[normalized]!.containsKey(qualKey)) {
            subjectChapters[normalized]![qualKey] = {};
          }

          if (resourceType != null && year != null) {
            final chapter = '$resourceType - $year ${session ?? ""}'.trim();
            if (chapter.length > 3) {
              subjectChapters[normalized]![qualKey]!.add(chapter);
            }
          }
        }

        final result = subjectChapters.map(
          (key, qualMap) => MapEntry(
            key,
            qualMap.map((k, v) => MapEntry(k, v.toList()..sort())),
          ),
        );

        await prefs.setString(_resourcesKey, jsonEncode(result));
        return result;
      }
    } catch (_) {}

    return const {};
  }

  Future<Map<String, List<String>>> loadSyllabusChapters(String subject,
      {String? board, String? uid}) async {
    if (uid != null) {
      final userChapters =
          await CurriculumCatalogService.instance.getUserChapters(uid);
      final normalized = StudyCatalog.normalizeSubject(subject);
      if (userChapters.containsKey(normalized)) {
        return {normalized: userChapters[normalized]!};
      }
    }

    final normalizedBoard = board?.trim().isNotEmpty == true ? board! : 'IGCSE';
    return CurriculumCatalogService.instance.chapterTree(
      board: normalizedBoard,
      subject: StudyCatalog.normalizeSubject(subject),
    );
  }
}

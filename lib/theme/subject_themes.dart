import 'package:flutter/material.dart';

class SubjectTheme {
  final Color primary;
  final Color secondary;
  final IconData icon;
  final String symbol;

  const SubjectTheme({
    required this.primary,
    required this.secondary,
    required this.icon,
    required this.symbol,
  });

  Color get primaryColor => primary;
  Color get secondaryColor => secondary;
}

const Map<String, List<String>> _subjectAliases = {
  'Mathematics': [
    'mathematics',
    'math',
    'maths',
    'pure mathematics',
    'pure math',
    'p1',
    'p2',
    'p3',
    'p4',
    's1',
    's2',
    'm1',
    'm2',
    '9709',
    '0580',
    '0607',
    '4024',
    '4029',
  ],
  'Further Mathematics': [
    'further mathematics',
    'further maths',
    'further math',
    'additional mathematics',
    'additional math',
    '9231',
    '0606',
    'fm',
    'fp1',
    'fp2',
    'fp3',
  ],
  'Physics': [
    'physics',
    'phys',
    '9702',
    '0625',
    '5054',
    '9202',
  ],
  'Chemistry': [
    'chemistry',
    'chem',
    '9701',
    '0620',
    '5070',
    '9201',
  ],
  'Biology': [
    'biology',
    'bio',
    '9700',
    '0610',
    '5090',
    '9700',
  ],
  'English': [
    'english',
    'english language',
    'language',
    'literature',
    'english literature',
    'english lang',
    '1123',
    '0500',
    '0475',
    '9093',
  ],
  'Computer Science': [
    'computer science',
    'computer',
    'computing',
    'cs',
    'comp sci',
    'compsci',
    '9618',
    '0478',
    '2210',
    '9471',
    '9608',
  ],
  'Information Technology': [
    'information technology',
    'ict',
    'information and communication technology',
    '9618',
    '0413',
  ],
  'Economics': [
    'economics',
    'econ',
    'economy',
    '9708',
    '0455',
    '2281',
    '9221',
  ],
  'Business': [
    'business',
    'business studies',
    '9609',
    '0450',
    '0451',
  ],
  'Accounting': [
    'accounting',
    'accounts',
    '9706',
    '0452',
    '9212',
  ],
  'History': [
    'history',
    'hist',
    '9389',
    '0470',
    '9013',
  ],
  'Geography': [
    'geography',
    'geog',
    '9696',
    '0460',
    '9226',
  ],
  'Chinese': [
    'chinese',
    'mandarin',
    '9626',
    '9238',
    '0549',
  ],
  'French': [
    'french',
    '9681',
    '8651',
    '0502',
  ],
  'Arabic': [
    'arabic',
    '9781',
    '0544',
  ],
  'Spanish': [
    'spanish',
    '9785',
    '8665',
  ],
  'German': [
    'german',
    '9705',
    '8683',
  ],
  'Global Perspectives': [
    'global perspectives',
    'gp',
    'gpt',
    '9014',
    '0467',
  ],
  'Design and Technology': [
    'design and technology',
    'dt',
    'd&t',
    '9682',
    '0445',
  ],
  'Media Studies': [
    'media studies',
    'media',
    '9715',
    '0486',
  ],
  'Music': [
    'music',
    '9703',
    '0480',
  ],
  'Art and Design': [
    'art and design',
    'art',
    'art & design',
    '9471',
    '0400',
    '9481',
  ],
  'Environmental Management': [
    'environmental management',
    'em',
    '9694',
  ],
  'Sociology': [
    'sociology',
    '9699',
    '0493',
  ],
  'Psychology': [
    'psychology',
    'psych',
    '9698',
    '0448',
  ],
  'Law': [
    'law',
    '9824',
    '0487',
  ],
  'Thinking Skills': [
    'thinking skills',
    'critical thinking',
    'alevel thinking skills',
    '9704',
    '9393',
  ],
};

String _normalizedSubjectKey(String input) {
  return input
      .toLowerCase()
      .replaceAll('&', ' and ')
      .replaceAll(RegExp(r'[^a-z0-9]+'), ' ')
      .trim();
}

String canonicalSubject(String input, {String fallback = 'General'}) {
  final normalized = _normalizedSubjectKey(input);
  if (normalized.isEmpty) return fallback;

  for (final entry in _subjectAliases.entries) {
    final canonical = _normalizedSubjectKey(entry.key);
    if (normalized == canonical) return entry.key;
    if (entry.value
        .any((alias) => normalized == _normalizedSubjectKey(alias))) {
      return entry.key;
    }
  }

  for (final entry in _subjectAliases.entries) {
    final canonical = _normalizedSubjectKey(entry.key);
    if (normalized.contains(canonical) || canonical.contains(normalized)) {
      return entry.key;
    }
    for (final alias in entry.value) {
      final normalizedAlias = _normalizedSubjectKey(alias);
      if (normalized.contains(normalizedAlias) ||
          normalizedAlias.contains(normalized)) {
        return entry.key;
      }
    }
  }

  return fallback;
}

Map<String, List<String>> get subjectAliases =>
    Map<String, List<String>>.unmodifiable(_subjectAliases);

String subjectSvgAsset(String subject) {
  switch (canonicalSubject(subject)) {
    case 'Mathematics':
    case 'Further Mathematics':
      return 'assets/subjects/mathematics.svg';
    case 'Physics':
      return 'assets/subjects/physics.svg';
    case 'Chemistry':
      return 'assets/subjects/chemistry.svg';
    case 'Biology':
      return 'assets/subjects/biology.svg';
    case 'Computer Science':
    case 'Information Technology':
      return 'assets/subjects/computer_science.svg';
    default:
      return 'assets/subjects/general.svg';
  }
}

const List<String> subjectSvgAssets = [
  'assets/subjects/mathematics.svg',
  'assets/subjects/physics.svg',
  'assets/subjects/chemistry.svg',
  'assets/subjects/biology.svg',
  'assets/subjects/computer_science.svg',
  'assets/subjects/general.svg',
];

SubjectTheme themeFor(String subject) {
  // Refined Subject System - Similar saturation levels for premium look
  const mathBlue = Color(0xFF3A86FF);         // Brand Alignment
  const physPurple = Color(0xFF7C3AED);       // Deep Violet
  const chemTeal = Color(0xFF0D9488);         // Dark Teal
  const bioGreen = Color(0xFF059669);        // Deep Emerald
  const csOrange = Color(0xFFEA580C);        // Burnt Orange
  const histAmber = Color(0xFFB45309);        // Golden Brown
  const geoBlue = Color(0xFF0369A1);          // Ocean Blue
  const econRose = Color(0xFFBE123C);        // Crimson
  const psychIndigo = Color(0xFF6366F1);
  const engRed = Color(0xFFEF4444);
  const artPink = Color(0xFFEC4899);
  const langTeal = Color(0xFF0D9488);
  const defaultGray = Color(0xFF6B7280);
  switch (canonicalSubject(subject)) {
    case 'Mathematics':
    case 'Further Mathematics':
      return const SubjectTheme(
        primary: mathBlue,
        secondary: mathBlue,
        icon: Icons.functions_rounded,
        symbol: 'MATH',
      );
    case 'Physics':
      return const SubjectTheme(
        primary: physPurple,
        secondary: physPurple,
        icon: Icons.bolt_rounded,
        symbol: 'PHY',
      );
    case 'Chemistry':
      return const SubjectTheme(
        primary: chemTeal,
        secondary: chemTeal,
        icon: Icons.science_rounded,
        symbol: 'CHEM',
      );
    case 'Biology':
      return const SubjectTheme(
        primary: bioGreen,
        secondary: bioGreen,
        icon: Icons.biotech_rounded,
        symbol: 'BIO',
      );
    case 'Computer Science':
    case 'Information Technology':
      return const SubjectTheme(
        primary: csOrange,
        secondary: csOrange,
        icon: Icons.computer_rounded,
        symbol: 'CS',
      );
    case 'History':
      return const SubjectTheme(
        primary: histAmber,
        secondary: histAmber,
        icon: Icons.history_edu_rounded,
        symbol: 'HIST',
      );
    case 'Geography':
      return const SubjectTheme(
        primary: geoBlue,
        secondary: geoBlue,
        icon: Icons.public_rounded,
        symbol: 'GEO',
      );
    case 'Economics':
    case 'Business Studies':
      return const SubjectTheme(
        primary: econRose,
        secondary: econRose,
        icon: Icons.trending_up_rounded,
        symbol: 'ECON',
      );
    case 'Psychology':
      return const SubjectTheme(
        primary: psychIndigo,
        secondary: psychIndigo,
        icon: Icons.psychology_rounded,
        symbol: 'PSYCH',
      );
    case 'English':
    case 'English Language':
    case 'English Literature':
      return const SubjectTheme(
        primary: engRed,
        secondary: engRed,
        icon: Icons.menu_book_rounded,
        symbol: 'ENG',
      );
    case 'Art and Design':
    case 'Art':
      return const SubjectTheme(
        primary: artPink,
        secondary: artPink,
        icon: Icons.palette_rounded,
        symbol: 'ART',
      );
    case 'French':
    case 'Spanish':
    case 'Mandarin':
    case 'German':
    case 'Latin':
      return const SubjectTheme(
        primary: langTeal,
        secondary: langTeal,
        icon: Icons.translate_rounded,
        symbol: 'LANG',
      );
    case 'Music':
      return const SubjectTheme(
        primary: Color(0xFFA855F7),
        secondary: Color(0xFFA855F7),
        icon: Icons.music_note_rounded,
        symbol: 'MUSIC',
      );
    case 'Accounting':
      return const SubjectTheme(
        primary: Color(0xFF0EA5E9),
        secondary: Color(0xFF0EA5E9),
        icon: Icons.account_balance_rounded,
        symbol: 'ACC',
      );
    case 'Law':
      return const SubjectTheme(
        primary: Color(0xFF84CC16),
        secondary: Color(0xFF84CC16),
        icon: Icons.gavel_rounded,
        symbol: 'LAW',
      );
    default:
      return const SubjectTheme(
        primary: defaultGray,
        secondary: defaultGray,
        icon: Icons.subject_rounded,
        symbol: 'SUB',
      );
  }
}

List<Color> subjectGlow(String subject) {
  final theme = themeFor(subject);
  return [
    theme.primary.withValues(alpha: 0.24),
    theme.secondary.withValues(alpha: 0.08),
  ];
}

// lib/services/exam_zone_service.dart
import 'dart:convert';
import 'dart:io';

class ExamZoneService {
  static final ExamZoneService _instance = ExamZoneService._internal();
  factory ExamZoneService() => _instance;
  ExamZoneService._internal();

  // Cambridge International administrative zones
  // Zone 1: Mauritius, Seychelles
  // Zone 2: Middle East (except UAE, Saudi, etc.)
  // Zone 3: South Asia (India, Sri Lanka, etc.) - also March series
  // Zone 4: India & Romania (March series)
  // Zone 5: Southeast Asia (Malaysia, Singapore, etc.)
  // Zone 6: Americas, Africa, Pacific
  // Zone UK: United Kingdom
  static const Map<String, Map<String, dynamic>> zones = {
    'Pakistan': {
      'name': 'Pakistan',
      'region': 'Asia',
      'timezone': 'PKT (UTC+5)',
      'cambridgeZone': 'Zone 3',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'PK',
    },
    'India': {
      'name': 'India',
      'region': 'Asia',
      'timezone': 'IST (UTC+5:30)',
      'cambridgeZone': 'Zone 4',
      'datesheet': 'March/June/November',
      'series': ['March', 'June', 'November'],
      'code': 'IN',
    },
    'Bangladesh': {
      'name': 'Bangladesh',
      'region': 'Asia',
      'timezone': 'BST (UTC+6)',
      'cambridgeZone': 'Zone 3',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'BD',
    },
    'Sri Lanka': {
      'name': 'Sri Lanka',
      'region': 'Asia',
      'timezone': 'SLST (UTC+5:30)',
      'cambridgeZone': 'Zone 3',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'LK',
    },
    'Nepal': {
      'name': 'Nepal',
      'region': 'Asia',
      'timezone': 'NPT (UTC+5:45)',
      'cambridgeZone': 'Zone 3',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'NP',
    },
    'UAE': {
      'name': 'UAE',
      'region': 'Middle East',
      'timezone': 'GST (UTC+4)',
      'cambridgeZone': 'Zone 2',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'AE',
    },
    'Saudi Arabia': {
      'name': 'Saudi Arabia',
      'region': 'Middle East',
      'timezone': 'AST (UTC+3)',
      'cambridgeZone': 'Zone 2',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'SA',
    },
    'Qatar': {
      'name': 'Qatar',
      'region': 'Middle East',
      'timezone': 'AST (UTC+3)',
      'cambridgeZone': 'Zone 2',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'QA',
    },
    'Kuwait': {
      'name': 'Kuwait',
      'region': 'Middle East',
      'timezone': 'AST (UTC+3)',
      'cambridgeZone': 'Zone 2',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'KW',
    },
    'Oman': {
      'name': 'Oman',
      'region': 'Middle East',
      'timezone': 'GST (UTC+4)',
      'cambridgeZone': 'Zone 2',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'OM',
    },
    'Bahrain': {
      'name': 'Bahrain',
      'region': 'Middle East',
      'timezone': 'AST (UTC+3)',
      'cambridgeZone': 'Zone 2',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'BH',
    },
    'Malaysia': {
      'name': 'Malaysia',
      'region': 'Southeast Asia',
      'timezone': 'MYT (UTC+8)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'MY',
    },
    'Singapore': {
      'name': 'Singapore',
      'region': 'Southeast Asia',
      'timezone': 'SGT (UTC+8)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'SG',
    },
    'Indonesia': {
      'name': 'Indonesia',
      'region': 'Southeast Asia',
      'timezone': 'WIB (UTC+7)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'ID',
    },
    'Thailand': {
      'name': 'Thailand',
      'region': 'Southeast Asia',
      'timezone': 'ICT (UTC+7)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'TH',
    },
    'Philippines': {
      'name': 'Philippines',
      'region': 'Southeast Asia',
      'timezone': 'PHT (UTC+8)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'PH',
    },
    'Vietnam': {
      'name': 'Vietnam',
      'region': 'Southeast Asia',
      'timezone': 'ICT (UTC+7)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'VN',
    },
    'Myanmar': {
      'name': 'Myanmar',
      'region': 'Southeast Asia',
      'timezone': 'MMT (UTC+6:30)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'MM',
    },
    'Hong Kong': {
      'name': 'Hong Kong',
      'region': 'Asia',
      'timezone': 'HKT (UTC+8)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'HK',
    },
    'China': {
      'name': 'China',
      'region': 'Asia',
      'timezone': 'CST (UTC+8)',
      'cambridgeZone': 'Zone 5',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'CN',
    },
    'United Kingdom': {
      'name': 'United Kingdom',
      'region': 'Europe',
      'timezone': 'GMT/BST (UTC+0/UTC+1)',
      'cambridgeZone': 'Zone UK',
      'datesheet': 'June/November',
      'series': ['June', 'November'],
      'code': 'GB',
    },
    'Mauritius': {
      'name': 'Mauritius',
      'region': 'Africa',
      'timezone': 'MUT (UTC+4)',
      'cambridgeZone': 'Zone 1',
      'datesheet': 'June/November',
      'series': ['June', 'November'],
      'code': 'MU',
    },
    'Seychelles': {
      'name': 'Seychelles',
      'region': 'Africa',
      'timezone': 'SCT (UTC+4)',
      'cambridgeZone': 'Zone 1',
      'datesheet': 'June/November',
      'series': ['June', 'November'],
      'code': 'SC',
    },
    'South Africa': {
      'name': 'South Africa',
      'region': 'Africa',
      'timezone': 'SAST (UTC+2)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'ZA',
    },
    'Nigeria': {
      'name': 'Nigeria',
      'region': 'Africa',
      'timezone': 'WAT (UTC+1)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'NG',
    },
    'Kenya': {
      'name': 'Kenya',
      'region': 'Africa',
      'timezone': 'EAT (UTC+3)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'KE',
    },
    'Egypt': {
      'name': 'Egypt',
      'region': 'Africa',
      'timezone': 'EET (UTC+2)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'EG',
    },
    'USA': {
      'name': 'United States',
      'region': 'Americas',
      'timezone': 'EST/PDT (UTC-5/UTC-8)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['June', 'November'],
      'code': 'US',
    },
    'Canada': {
      'name': 'Canada',
      'region': 'Americas',
      'timezone': 'EST/PST (UTC-5/UTC-8)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['June', 'November'],
      'code': 'CA',
    },
    'Australia': {
      'name': 'Australia',
      'region': 'Oceania',
      'timezone': 'AEST (UTC+10)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'AU',
    },
    'New Zealand': {
      'name': 'New Zealand',
      'region': 'Oceania',
      'timezone': 'NZST (UTC+12)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'NZ',
    },
    'Romania': {
      'name': 'Romania',
      'region': 'Europe',
      'timezone': 'EET (UTC+2)',
      'cambridgeZone': 'Zone 4',
      'datesheet': 'March/June/November',
      'series': ['March', 'June', 'November'],
      'code': 'RO',
    },
    'Europe (Other)': {
      'name': 'Europe',
      'region': 'Europe',
      'timezone': 'CET (UTC+1)',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'EU',
    },
    'Other': {
      'name': 'Other',
      'region': 'Other',
      'timezone': 'UTC',
      'cambridgeZone': 'Zone 6',
      'datesheet': 'June/November',
      'series': ['March', 'June', 'November'],
      'code': 'OTHER',
    },
  };

  // Board-specific series availability — CAIE only
  static const Map<String, List<String>> boardSeries = {
    'caie_igcse': ['June', 'November'],
    'caie_as_level': ['March', 'June', 'November'],
    'caie_a_level': ['March', 'June', 'November'],
  };

  // Board-specific zone requirements — all CAIE levels use administrative zones
  static const Map<String, bool> boardUsesZones = {
    'caie_igcse': true,
    'caie_as_level': true,
    'caie_a_level': true,
  };

  static List<String> get countryList => zones.keys.toList();

  static Map<String, dynamic>? getZone(String country) => zones[country];

  static String getTimezone(String country) =>
      zones[country]?['timezone'] ?? 'UTC';

  static String getCambridgeZone(String country) =>
      zones[country]?['cambridgeZone'] ?? 'Zone 6';

  static List<String> getSeries(String country) =>
      zones[country]?['series'] ?? ['June', 'November'];

  static List<String> getSeriesForBoard(String board) {
    final canonical = board.toLowerCase().trim();
    final direct = boardSeries[canonical];
    if (direct != null) return direct;

    // All inputs map to CAIE series by level
    if (canonical.contains('as_level') || canonical.contains('a_level')) {
      return ['March', 'June', 'November'];
    }
    return ['June', 'November'];
  }

  static bool boardUsesAdministrativeZones(String board) {
    return true; // CAIE is the only board and uses zones
  }

  static String getCurrentSeries() {
    final now = DateTime.now();
    final month = now.month;
    if (month >= 1 && month <= 2) return 'March';
    if (month >= 3 && month <= 5) return 'June';
    return 'November';
  }

  static int getCurrentYear() => DateTime.now().year;
}

class ExamDateEvent {
  final String board;
  final String subject;
  final String component;
  final DateTime date;
  final String startTime;
  final String endTime;
  final String timezone;
  final String source;

  ExamDateEvent({
    required this.board,
    required this.subject,
    required this.component,
    required this.date,
    required this.startTime,
    required this.endTime,
    required this.timezone,
    required this.source,
  });

  String get formattedDate =>
      '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';

  int get daysUntil => date.difference(DateTime.now()).inDays;
}

class ExamDatesheetService {
  static final ExamDatesheetService _instance =
      ExamDatesheetService._internal();
  factory ExamDatesheetService() => _instance;
  ExamDatesheetService._internal();

  List<ExamDateEvent>? _cachedEvents;
  String? _cachedBoard;

  Future<List<ExamDateEvent>> loadDatesheet({
    required String board,
    int? year,
  }) async {
    final targetYear = year ?? ExamZoneService.getCurrentYear();

    if (_cachedEvents != null && _cachedBoard == board) {
      return _cachedEvents!;
    }

    try {
      final file = File('datesheet_cambridge_2026_June.json');
      if (await file.exists()) {
        final content = await file.readAsString();
        final data = jsonDecode(content);

        if (data is Map && data['events'] is List) {
          final events = <ExamDateEvent>[];

          for (final e in data['events']) {
            if (e is Map) {
              final dateStr = e['date']?.toString();
              if (dateStr != null && dateStr.isNotEmpty) {
                final dateParts = dateStr.split('-');
                if (dateParts.length == 3) {
                  final date = DateTime(
                    int.tryParse(dateParts[0]) ?? DateTime.now().year,
                    int.tryParse(dateParts[1]) ?? 1,
                    int.tryParse(dateParts[2]) ?? 1,
                  );

                  if (date.year == targetYear) {
                    events.add(ExamDateEvent(
                      board: e['board']?.toString() ?? '',
                      subject: e['subject']?.toString() ?? '',
                      component: e['component']?.toString() ?? '',
                      date: date,
                      startTime: e['start_time']?.toString() ?? '08:00',
                      endTime: e['end_time']?.toString() ?? '11:00',
                      timezone: e['timezone']?.toString() ?? 'UTC',
                      source: e['source']?.toString() ?? '',
                    ));
                  }
                }
              }
            }
          }

          events.sort((a, b) => a.date.compareTo(b.date));
          _cachedEvents = events;
          _cachedBoard = board;

          return events;
        }
      }
    } catch (e) {
      // Return empty list on error
    }

    return [];
  }

  Future<List<ExamDateEvent>> getUpcomingExams({
    int days = 90,
    String? subject,
  }) async {
    final events = await loadDatesheet(board: 'CAIE');
    final now = DateTime.now();
    final cutoff = now.add(Duration(days: days));

    return events.where((e) {
      if (e.date.isBefore(now) || e.date.isAfter(cutoff)) return false;
      if (subject != null &&
          !e.subject.toLowerCase().contains(subject.toLowerCase())) {
        return false;
      }
      return true;
    }).toList();
  }

  Future<List<ExamDateEvent>> getExamsForSubject(String subject) async {
    return getUpcomingExams(subject: subject);
  }

  Future<DateTime?> getNextExamDate(String? subject) async {
    final events = await getUpcomingExams(subject: subject);
    if (events.isEmpty) return null;
    return events.first.date;
  }

  Future<int> getDaysUntilNextExam(String? subject) async {
    final nextDate = await getNextExamDate(subject);
    if (nextDate == null) return -1;
    return nextDate.difference(DateTime.now()).inDays;
  }
}

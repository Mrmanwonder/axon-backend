// lib/services/past_papers_service.dart
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class PastPapersService {
  static final PastPapersService _instance = PastPapersService._internal();
  factory PastPapersService() => _instance;
  PastPapersService._internal();

  static const Map<String, String> subjectNameToCode = {
    'Biology': '0610',
    'Chemistry': '0620',
    'Physics': '0625',
    'Mathematics': '0580',
    'Computer Science': '0478',
    'Economics': '0455',
    'Business Studies': '0450',
    'Geography': '0460',
    'History': '0416',
    'English': '0500',
    'English as a Second Language': '0476',
  };

  static const Map<String, List<String>> subjectCodes = {
    'Biology': ['0610', '5090', '9700'],
    'Chemistry': ['0620', '5070', '9701'],
    'Physics': ['0625', '5054', '9702'],
    'Mathematics': ['0580', '4024', '9709'],
    'Computer Science': ['0478', '0984', '9608'],
    'Information and Communication Technology': ['0417', '0983', '9618'],
    'Economics': ['0455', '9708'],
    'Business Studies': ['0450', '9707'],
    'Geography': ['0460', '9696'],
    'History': ['0416', '9477', '9697'],
    'English Language': ['0500', '0990'],
    'English as a Second Language': ['0476', '0993'],
    'French': ['0501', '0520', '9716'],
    'German': ['0505', '0525', '9717'],
    'Spanish': ['0502', '0530', '9719'],
    'Arabic': ['0508', '0527', '9680'],
    'Hindi': ['0549', '9687'],
    'English Literature': ['9695'],
    'Psychology': ['9698'],
  };

  static const Map<String, String> subjectCodeToName = {
    '0610': 'Biology',
    '5090': 'Biology',
    '9700': 'Biology',
    '0620': 'Chemistry',
    '5070': 'Chemistry',
    '9701': 'Chemistry',
    '0625': 'Physics',
    '5054': 'Physics',
    '9702': 'Physics',
    '0580': 'Mathematics',
    '4024': 'Mathematics',
    '9709': 'Mathematics',
    '0478': 'Computer Science',
    '0984': 'Computer Science',
    '9608': 'Computer Science',
    '0417': 'Information and Communication Technology',
    '0983': 'Information and Communication Technology',
    '9618': 'Information and Communication Technology',
    '0455': 'Economics',
    '9708': 'Economics',
    '0450': 'Business Studies',
    '9707': 'Business Studies',
    '0460': 'Geography',
    '9696': 'Geography',
    '0416': 'History',
    '9477': 'History',
    '9697': 'History',
    '0500': 'English Language',
    '0990': 'English Language',
    '0476': 'English as a Second Language',
    '0993': 'English as a Second Language',
    '0501': 'French',
    '0520': 'French',
    '9716': 'French',
    '0505': 'German',
    '0525': 'German',
    '9717': 'German',
    '0502': 'Spanish',
    '0530': 'Spanish',
    '9719': 'Spanish',
    '0508': 'Arabic',
    '0527': 'Arabic',
    '9680': 'Arabic',
    '0549': 'Hindi',
    '9687': 'Hindi',
    '9695': 'English Literature',
    '9698': 'Psychology',
  };

  String? getCodeForSubject(String subject) {
    final normalized = subject.trim().toLowerCase();
    for (final entry in subjectNameToCode.entries) {
      if (entry.key.toLowerCase() == normalized) {
        return entry.value;
      }
    }
    return null;
  }

  String getSubjectName(String code) {
    return subjectCodeToName[code] ?? '';
  }

  Future<String> get _pastPapersPath async {
    final appDir = await getApplicationDocumentsDirectory();
    return '${appDir.path}/../../PastPapers';
  }

  Future<List<int>> getAvailableYears(String subjectCode) async {
    // Always show years 2019-2026 regardless of offline storage
    // The actual download happens on-demand when user selects a paper
    return [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019];
  }

  Future<List<String>> getAvailableSeries(String subjectCode, int year) async {
    // Return available series - these will be downloaded on-demand
    return ['Summer', 'Winter'];
  }

  Future<List<String>> getPaperVariants(
      String subjectCode, int year, String series) async {
    final path = await _pastPapersPath;
    final qpPath = Directory('$path/$subjectCode/$year/$series/QP');
    if (!await qpPath.exists()) return [];

    final variants = <String>[];
    await for (final entity in qpPath.list()) {
      if (entity is Directory) {
        final name = entity.path.split(Platform.pathSeparator).last;
        variants.add(name);
      }
    }
    variants.sort();
    return variants;
  }

  Future<List<String>> getPaperTypes(
      String subjectCode, int year, String series, String variant) async {
    final path = await _pastPapersPath;
    final qpPath = Directory('$path/$subjectCode/$year/$series/QP/$variant');
    if (!await qpPath.exists()) return ['QP'];

    final types = <String>[];
    await for (final entity in qpPath.list()) {
      if (entity is File) {
        final name = entity.path.split(Platform.pathSeparator).last;
        if (name.endsWith('.pdf')) {
          types.add('QP');
          break;
        }
      }
    }

    final msPath = Directory('$path/$subjectCode/$year/$series/MS/$variant');
    if (await msPath.exists()) {
      await for (final entity in msPath.list()) {
        if (entity is File) {
          final name = entity.path.split(Platform.pathSeparator).last;
          if (name.endsWith('.pdf')) {
            types.add('MS');
            break;
          }
        }
      }
    }

    return types.isEmpty ? ['QP'] : types;
  }

  Future<String?> getPaperPath(String subjectCode, int year, String series,
      String variant, String type) async {
    final path = await _pastPapersPath;
    final filePath = '$path/$subjectCode/$year/$series/$type/$variant';

    final dir = Directory(filePath);
    if (!await dir.exists()) return null;

    String? foundFile;
    await for (final entity in dir.list()) {
      if (entity is File && entity.path.endsWith('.pdf')) {
        foundFile = entity.path;
        break;
      }
    }
    return foundFile;
  }

  String getSubjectNameFromCode(String code) {
    for (final entry in subjectNameToCode.entries) {
      if (entry.value == code) return entry.key;
    }
    return '';
  }

  List<String> getAllSubjectCodes(String subject) {
    return subjectCodes[subject] ?? [];
  }
}

class PastPaperYear {
  final int year;
  final List<String> series;

  PastPaperYear({required this.year, required this.series});
}

class PastPaperVariant {
  final String variant;
  final List<String> types;

  PastPaperVariant({required this.variant, required this.types});
}

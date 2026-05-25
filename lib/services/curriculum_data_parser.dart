import 'dart:io';
import 'package:path_provider/path_provider.dart';

class CurriculumDataParser {
  static Future<Map<String, dynamic>> parseAllCurriculumData() async {
    final directory = await getApplicationDocumentsDirectory();
    final curriculumDir = Directory('${directory.path}/../Curriculumn');
    
    final subjects = <Map<String, dynamic>>[];
    final chapters = <Map<String, dynamic>>[];
    final subchapters = <Map<String, dynamic>>[];
    final syllabi = <Map<String, dynamic>>[];
    
    // Parse subjects_master.csv
    final subjectsFile = File('${curriculumDir.path}/subjects_master.csv');
    if (await subjectsFile.exists()) {
      final content = await subjectsFile.readAsString();
      final lines = content.split('\n');
      for (int i = 1; i < lines.length; i++) {
        final line = lines[i].trim();
        if (line.isEmpty) continue;
        final parts = _parseCsvLine(line);
        if (parts.length >= 8) {
          subjects.add({
            'board': parts[0],
            'qualification': parts[1],
            'subject_code': parts[2],
            'subject_name': parts[3],
            'level': parts[4],
            'version': parts[5],
            'year': parts[6],
            'pdf_url': parts[7],
          });
          
          syllabi.add({
            'subject_code': parts[2],
            'board': parts[0],
            'syllabus_url': parts[7],
            'year': parts[6],
          });
        }
      }
    }
    
    // Parse syllabus_structure.csv
    final structureFile = File('${curriculumDir.path}/syllabus_structure.csv');
    if (await structureFile.exists()) {
      final content = await structureFile.readAsString();
      final lines = content.split('\n');
      for (int i = 1; i < lines.length; i++) {
        final line = lines[i].trim();
        if (line.isEmpty) continue;
        final parts = _parseCsvLine(line);
        if (parts.length >= 6) {
          final subjectCode = parts[0];
          final chapterNum = parts[2];
          final chapterTitle = parts[3];
          final subchapterNum = parts[4];
          final subchapterTitle = parts[5];
          
          // Add chapter if not exists
          final chapterExists = chapters.any((c) => 
            c['subject_code'] == subjectCode && c['chapter_number'] == chapterNum);
          if (!chapterExists) {
            chapters.add({
              'subject_code': subjectCode,
              'board': parts[1],
              'chapter_number': chapterNum,
              'chapter_title': chapterTitle,
            });
          }
          
          // Add subchapter
          subchapters.add({
            'subject_code': subjectCode,
            'board': parts[1],
            'chapter_number': chapterNum,
            'subchapter_number': subchapterNum,
            'subchapter_title': subchapterTitle,
          });
        }
      }
    }
    
    return {
      'subjects': subjects,
      'chapters': chapters,
      'subchapters': subchapters,
      'syllabi': syllabi,
    };
  }
  
  static List<String> _parseCsvLine(String line) {
    final result = <String>[];
    var current = StringBuffer();
    var inQuotes = false;
    
    for (int i = 0; i < line.length; i++) {
      final char = line[i];
      if (char == '"') {
        inQuotes = !inQuotes;
      } else if (char == ',' && !inQuotes) {
        result.add(current.toString().trim());
        current = StringBuffer();
      } else {
        current.write(char);
      }
    }
    result.add(current.toString().trim());
    return result;
  }
}
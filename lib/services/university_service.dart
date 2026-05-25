import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/university.dart';
import 'grok_service.dart';

class UniversityService {
  static final UniversityService _instance = UniversityService._internal();
  factory UniversityService() => _instance;
  UniversityService._internal();

  static const String _backendUrl = 'https://bhavu.up.railway.app';

  Future<List<University>> searchUniversities(String query) async {
    try {
      final token = await _getIdToken();
      if (token == null) return [];

      final response = await http.post(
        Uri.parse('$_backendUrl/universities/search'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token',
        },
        body: jsonEncode({'query': query, 'limit': 50}),
      ).timeout(const Duration(seconds: 15));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final results = data['results'] as List;
        return results.map((r) => _fromBackendJson(r as Map<String, dynamic>)).toList();
      }
    } catch (e) {
      debugPrint('University search backend failed: $e');
    }
    return [];
  }

  Future<List<University>> suggestUniversities({
    String? preferredDegree,
    String? preferredLocation,
  }) async {
    final query = preferredDegree ?? preferredLocation ?? '';
    return searchUniversities(query);
  }

  Future<void> addToPreferences(UserUniversityPreference pref) async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList('university_prefs') ?? [];
    list.add(jsonEncode({
      'universityId': pref.universityId,
      'degreeId': pref.degreeId,
      'category': pref.category.name,
      'addedAt': pref.addedAt.toIso8601String(),
    }));
    await prefs.setStringList('university_prefs', list);
  }

  Future<List<UserUniversityPreference>> getPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList('university_prefs') ?? [];
    return list.map((item) {
      final map = jsonDecode(item);
      return UserUniversityPreference(
        universityId: map['universityId'],
        degreeId: map['degreeId'],
        category: UniversityCategory.values
            .firstWhere((e) => e.name == map['category']),
        addedAt: DateTime.parse(map['addedAt']),
      );
    }).toList();
  }

  Future<UserAchievement> rateAchievement(String title, String description) async {
    final grok = GrokService();
    if (grok.isReady) {
      try {
        final prompt = '''
        Rate the importance of this academic/extracurricular achievement for a university application.
        Achievement: $title
        Description: $description
        Provide a JSON response with:
        "rating": (float between 0.0 and 1.0)
        "feedback": (short explanation of why)
        ''';
        final response = await grok.chat(prompt, systemPrompt: 'You are a university admissions expert. Output JSON only.');
        final cleaned = response.replaceAll('```json', '').replaceAll('```', '').trim();
        final data = jsonDecode(cleaned);
        return UserAchievement(
          id: DateTime.now().millisecondsSinceEpoch.toString(),
          title: title,
          description: description,
          date: DateTime.now(),
          importanceRating: (data['rating'] as num).toDouble(),
          feedback: data['feedback'],
        );
      } catch (_) {
        rethrow;
      }
    }
    throw Exception('Grok service is not ready');
  }

  Future<List<Degree>> getPrograms(String universityName, {String country = '', String domain = ''}) async {
    try {
      final token = await _getIdToken();
      if (token == null) return [];

      final response = await http.post(
        Uri.parse('$_backendUrl/universities/programs'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token',
        },
        body: jsonEncode({
          'university_name': universityName,
          'country': country,
          'domain': domain,
        }),
      ).timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final programs = data['programs'] as List;
        return programs.map((p) {
          final m = p as Map<String, dynamic>;
          final reqs = (m['grade_requirements'] as Map<String, dynamic>?)?.map(
            (k, v) => MapEntry(k, v.toString()),
          ) ?? {};
          return Degree(
            id: m['id'] ?? '',
            name: m['name'] ?? m['course_name'] ?? 'Unknown Program',
            duration: '${m['duration_years'] ?? 4} Years',
            syllabus: (m['core_modules'] as List?)?.join(', ') ?? '',
            gradeRequirements: reqs,
            requiredSubjects: [],
          );
        }).toList();
      }
    } catch (e) {
      debugPrint('Get programs failed: $e');
    }
    return [];
  }

  University? getUniversityById(String id) {
    return null;
  }

  University _fromBackendJson(Map<String, dynamic> json) {
    return University(
      id: json['id'] ?? '',
      name: json['name'] ?? 'Unknown',
      location: '${json['country'] ?? ''}${json['state_province'] != null && json['state_province'].toString().isNotEmpty ? ', ${json['state_province']}' : ''}',
      description: 'University in ${json['country'] ?? 'unknown'}',
      logoUrl: json['logo_url'] ?? '',
      degrees: [],
    );
  }

  Future<String?> _getIdToken() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) return null;
      return await user.getIdToken();
    } catch (_) {
      return null;
    }
  }
}
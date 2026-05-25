import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'board_exam_service.dart';

class SerperSearchService {
  SerperSearchService._();

  static final SerperSearchService instance = SerperSearchService._();

  final http.Client _client = http.Client();

  String get _backendUrl => BoardExamService.backendUrl;

  Future<List<SerperSearchResult>> _proxySearch(
      Map<String, dynamic> body) async {
    try {
      final token = await FirebaseAuth.instance.currentUser?.getIdToken();
      if (token == null) return [];

      final response = await _client.post(
        Uri.parse('$_backendUrl/api/search'),
        headers: {
          'Authorization': 'Bearer $token',
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final organic = data['organic'] as List? ?? [];
        return organic
            .map((item) => SerperSearchResult(
                  title: item['title'] ?? '',
                  link: item['link'] ?? '',
                  snippet: item['snippet'] ?? '',
                  date: item['date'] ?? '',
                ))
            .where((r) => r.link.endsWith('.pdf') || r.link.contains('pdf'))
            .toList();
      }

      debugPrint('Serper proxy failed: ${response.statusCode}');
      return [];
    } catch (e) {
      debugPrint('Serper proxy error: $e');
      return [];
    }
  }

  Future<List<SerperSearchResult>> searchPastPapers(String query,
      {int limit = 10}) async {
    return _proxySearch({
      'q': '$query past papers pdf',
      'num': limit,
      'filetype': 'pdf',
    });
  }

  Future<List<SerperSearchResult>> searchByUrl(String url) async {
    return _proxySearch({'q': 'site:$url'});
  }
}

class SerperSearchResult {
  final String title;
  final String link;
  final String snippet;
  final String date;

  SerperSearchResult({
    required this.title,
    required this.link,
    required this.snippet,
    required this.date,
  });

  bool get isPdf => link.endsWith('.pdf') || link.contains('.pdf?');
}

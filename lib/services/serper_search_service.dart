import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class SerperSearchService {
  SerperSearchService._();

  static final SerperSearchService instance = SerperSearchService._();

  // Serper API key - loaded from .env
  String get _apiKey => dotenv.env['SERPER_API_KEY'] ?? '';
  static const String _baseUrl = 'https://google.serper.dev/search';

  final http.Client _client = http.Client();

  Future<List<SerperSearchResult>> searchPastPapers(String query,
      {int limit = 10}) async {
    try {
      final response = await _client.post(
        Uri.parse(_baseUrl),
        headers: {
          'X-API-KEY': _apiKey,
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'q': '$query past papers pdf',
          'num': limit,
          'filetype': 'pdf',
        }),
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

      debugPrint('Serper search failed: ${response.statusCode}');
      return [];
    } catch (e) {
      debugPrint('Serper search error: $e');
      return [];
    }
  }

  Future<List<SerperSearchResult>> searchByUrl(String url) async {
    try {
      final response = await _client.post(
        Uri.parse('https://google.serper.dev/search'),
        headers: {
          'X-API-KEY': _apiKey,
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'q': 'site:$url',
        }),
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
            .toList();
      }
      return [];
    } catch (e) {
      debugPrint('Serper search error: $e');
      return [];
    }
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

import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'auth_service.dart';
import 'board_exam_service.dart';

class SupabaseProxyService {
  static final SupabaseProxyService _instance = SupabaseProxyService._();
  static SupabaseProxyService get instance => _instance;
  SupabaseProxyService._();

  String get _baseUrl => BoardExamService.backendUrl;

  Future<List<Map<String, dynamic>>> query(
    String table, {
    String method = 'select',
    Map<String, dynamic> params = const {},
  }) async {
    final token = await AuthService.instance.getIdToken();
    if (token == null) return [];

    try {
      final resp = await http
          .post(
            Uri.parse('$_baseUrl/supabase/query'),
            headers: {
              'Authorization': 'Bearer $token',
              'Content-Type': 'application/json',
            },
            body: jsonEncode({
              'table': table,
              'method': method,
              'params': params,
            }),
          )
          .timeout(const Duration(seconds: 30));

      if (resp.statusCode == 200) {
        final decoded = jsonDecode(resp.body);
        if (decoded is List) {
          return decoded.cast<Map<String, dynamic>>();
        }
        if (decoded is Map) {
          return [decoded.cast<String, dynamic>()];
        }
        return [];
      }
      debugPrint('SupabaseProxy: $table $method failed (${resp.statusCode}): ${resp.body}');
      return [];
    } catch (e) {
      debugPrint('SupabaseProxy: $table $method error: $e');
      return [];
    }
  }

  Future<void> mutate(
    String table, {
    required String method,
    Map<String, dynamic> body = const {},
    Map<String, dynamic> params = const {},
  }) async {
    final token = await AuthService.instance.getIdToken();
    if (token == null) return;

    try {
      final resp = await http
          .post(
            Uri.parse('$_baseUrl/supabase/query'),
            headers: {
              'Authorization': 'Bearer $token',
              'Content-Type': 'application/json',
            },
            body: jsonEncode({
              'table': table,
              'method': method,
              'params': {'body': body, ...params},
            }),
          )
          .timeout(const Duration(seconds: 30));

      if (resp.statusCode != 200) {
        debugPrint('SupabaseProxy: $table $method failed (${resp.statusCode}): ${resp.body}');
      }
    } catch (e) {
      debugPrint('SupabaseProxy: $table $method error: $e');
    }
  }
}

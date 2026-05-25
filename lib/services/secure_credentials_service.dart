import 'dart:async';
import 'dart:convert';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'board_exam_service.dart';
import 'package:http/http.dart' as http;

class ApiCredentials {
  final String? supabaseUrl;
  final String? supabaseAnonKey;
  final String? geminiApiKey;
  final String? serperApiKey;
  final String? cloudinaryCloud;
  final String? cloudinaryPreset;
  final String? grokApiKey;
  final String? vercelApiKey;
  final String? openrouterApiKey;
  final String? deepseekApiKey;
  final String? deepgramApiKey;
  final String? googleDrivePrivateKey;

  const ApiCredentials({
    this.supabaseUrl,
    this.supabaseAnonKey,
    this.geminiApiKey,
    this.serperApiKey,
    this.cloudinaryCloud,
    this.cloudinaryPreset,
    this.grokApiKey,
    this.vercelApiKey,
    this.openrouterApiKey,
    this.deepseekApiKey,
    this.deepgramApiKey,
    this.googleDrivePrivateKey,
  });

  bool get hasAnyKey => [
        serperApiKey,
        cloudinaryCloud,
        cloudinaryPreset,
        grokApiKey,
        vercelApiKey,
        openrouterApiKey,
        deepseekApiKey,
        deepgramApiKey,
        googleDrivePrivateKey,
      ].any((k) => k != null && k.isNotEmpty);

  String? get effectiveSerperKey => serperApiKey;
  String? get effectiveCloudinaryCloud => cloudinaryCloud;
  String? get effectiveCloudinaryPreset => cloudinaryPreset;
  String? get effectiveGrokKey => grokApiKey;
  String? get effectiveVercelKey => vercelApiKey;
  String? get effectiveOpenrouterKey => openrouterApiKey;
  String? get effectiveDeepseekKey => deepseekApiKey;
  String? get effectiveDeepgramKey => deepgramApiKey;
  String? get effectiveGoogleDriveKey => googleDrivePrivateKey;
  // Legacy Supabase/Gemini getters — always return null (keys moved to backend)
  String? get effectiveSupabaseUrl => null;
  String? get effectiveSupabaseAnonKey => null;
  String? get effectiveGeminiKey => null;
}

class SecureCredentialsService {
  static final SecureCredentialsService _instance =
      SecureCredentialsService._internal();
  factory SecureCredentialsService() => _instance;
  SecureCredentialsService._internal();

  static const String _cacheKey = 'axon_credentials_cache';

  // In-memory cache fetched from backend
  ApiCredentials? _cached;

  /// Backward-compatible initialization (now a no-op — keys fetched lazily).
  Future<void> initialize() async {}

  Future<ApiCredentials> getAllCredentials() async {
    if (_cached != null) return _cached!;

    // Try backend first
    try {
      final token = await FirebaseAuth.instance.currentUser?.getIdToken();
      if (token != null) {
        final backendUrl = BoardExamService.backendUrl;
        final resp = await http
            .post(
              Uri.parse('$backendUrl/api/credentials'),
              headers: {
                'Authorization': 'Bearer $token',
                'Content-Type': 'application/json',
              },
            )
            .timeout(const Duration(seconds: 10));

        if (resp.statusCode == 200) {
          final data = Map<String, dynamic>.from(jsonDecode(resp.body));
          final creds = ApiCredentials(
            serperApiKey: data['serper_api_key'] as String?,
            cloudinaryCloud: data['cloudinary_cloud_name'] as String?,
            cloudinaryPreset: data['cloudinary_upload_preset'] as String?,
            grokApiKey: data['grok_api_key'] as String?,
            vercelApiKey: data['vercel_api_key'] as String?,
            openrouterApiKey: data['openrouter_api_key'] as String?,
            deepseekApiKey: data['deepseek_api_key'] as String?,
            deepgramApiKey: data['deepgram_api_key'] as String?,
            googleDrivePrivateKey: data['google_drive_private_key'] as String?,
          );
          if (creds.hasAnyKey) {
            _cached = creds;
            _cacheLocally(data);
            return creds;
          }
        }
      }
    } catch (e) {
      debugPrint('SecureCredentials: Backend fetch failed, using cache: $e');
    }

    // Fall back to local cache
    _cached = await _loadFromCache();
    if (_cached != null) return _cached!;

    // Empty — no credentials available
    return const ApiCredentials();
  }

  Future<ApiCredentials?> _loadFromCache() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_cacheKey);
      if (raw == null || raw.isEmpty) return null;
      final data = Map<String, dynamic>.from(jsonDecode(raw));
      return ApiCredentials(
        serperApiKey: data['serper_api_key'] as String?,
        cloudinaryCloud: data['cloudinary_cloud_name'] as String?,
        cloudinaryPreset: data['cloudinary_upload_preset'] as String?,
        grokApiKey: data['grok_api_key'] as String?,
        vercelApiKey: data['vercel_api_key'] as String?,
        openrouterApiKey: data['openrouter_api_key'] as String?,
        deepseekApiKey: data['deepseek_api_key'] as String?,
        deepgramApiKey: data['deepgram_api_key'] as String?,
        googleDrivePrivateKey: data['google_drive_private_key'] as String?,
      );
    } catch (_) {
      return null;
    }
  }

  void _cacheLocally(Map<String, dynamic> data) {
    try {
      SharedPreferences.getInstance().then((prefs) {
        prefs.setString(_cacheKey, jsonEncode(data));
      });
    } catch (_) {}
  }

  void clearCache() {
    _cached = null;
    SharedPreferences.getInstance().then((prefs) {
      prefs.remove(_cacheKey);
    });
  }

  /// Legacy: no-op since keys now come from backend
  Future<void> storeAllCredentials(ApiCredentials _) async {}
  Future<void> clearAllCredentials() async {}
  bool isValidApiKey(String? key) => key != null && key.isNotEmpty;
}

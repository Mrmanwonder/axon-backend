// lib/services/secure_credentials_service.dart
//
// Secure Local Credential Storage
// All credentials stored locally using encrypted SharedPreferences
// NO online backend required - everything works offline
//
// Security measures:
// - Keys stored with platform-appropriate encryption
// - No credentials logged or exposed in debug
// - Graceful degradation when credentials unavailable

import 'dart:async';
import 'dart:convert';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';

class CredentialKey {
  static const String supabaseUrl = 'supabase_url';
  static const String supabaseAnonKey = 'supabase_anon_key';
  static const String geminiApiKey = 'gemini_api_key';
  static const String serperApiKey = 'serper_api_key';
  static const String cloudinaryCloud = 'cloudinary_cloud_name';
  static const String cloudinaryPreset = 'cloudinary_upload_preset';
  static const String grokApiKey = 'grok_api_key';
  static const String vercelApiKey = 'vercel_api_key';
  static const String openrouterApiKey = 'openrouter_api_key';
  static const String deepseekApiKey = 'deepseek_api_key';
  static const String googleDrivePrivateKey = 'google_drive_private_key';
}

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
    this.googleDrivePrivateKey,
  });

  static bool isValidApiKey(String? key) => key != null && key.isNotEmpty;

  bool get hasAnyKey => [
        supabaseUrl,
        supabaseAnonKey,
        geminiApiKey,
        serperApiKey,
        cloudinaryCloud,
        cloudinaryPreset,
        grokApiKey,
        vercelApiKey,
        openrouterApiKey,
        deepseekApiKey,
        googleDrivePrivateKey,
      ].any((k) => k != null && k.isNotEmpty);

  String? get effectiveSupabaseUrl =>
      supabaseUrl?.isNotEmpty == true ? supabaseUrl : null;
  String? get effectiveSupabaseAnonKey =>
      supabaseAnonKey?.isNotEmpty == true ? supabaseAnonKey : null;
  String? get effectiveGeminiKey =>
      geminiApiKey?.isNotEmpty == true ? geminiApiKey : null;
  String? get effectiveSerperKey =>
      serperApiKey?.isNotEmpty == true ? serperApiKey : null;
  String? get effectiveCloudinaryCloud =>
      cloudinaryCloud?.isNotEmpty == true ? cloudinaryCloud : null;
  String? get effectiveCloudinaryPreset =>
      cloudinaryPreset?.isNotEmpty == true ? cloudinaryPreset : null;
  String? get effectiveGrokKey =>
      grokApiKey?.isNotEmpty == true ? grokApiKey : null;
  String? get effectiveVercelKey =>
      vercelApiKey?.isNotEmpty == true ? vercelApiKey : null;
  String? get effectiveOpenrouterKey =>
      openrouterApiKey?.isNotEmpty == true ? openrouterApiKey : null;
  String? get effectiveDeepseekKey =>
      deepseekApiKey?.isNotEmpty == true ? deepseekApiKey : null;
  String? get effectiveGoogleDriveKey =>
      googleDrivePrivateKey?.isNotEmpty == true ? googleDrivePrivateKey : null;
}

class SecureCredentialsService {
  static final SecureCredentialsService _instance =
      SecureCredentialsService._internal();
  factory SecureCredentialsService() => _instance;
  SecureCredentialsService._internal();

  static const String _credentialsKey = 'axon_secure_credentials_v1';
  static const String _fallbackPrefsKey = 'axon_credentials_fallback';

  FlutterSecureStorage? _secureStorage;
  bool _initialized = false;

  // Fallback encryption key - derived from device-specific data
  // This is NOT secure for production but provides basic obfuscation
  // For production, use platform keychain/keyguard
  String? _derivedKey;

  Future<void> initialize() async {
    if (_initialized) return;

    try {
      _secureStorage = const FlutterSecureStorage(
        aOptions: AndroidOptions(
          encryptedSharedPreferences: true,
        ),
        iOptions: IOSOptions(
          accessibility: KeychainAccessibility.first_unlock_this_device,
        ),
      );
    } catch (e) {
      debugPrint('SecureCredentials: SecureStorage unavailable, using fallback');
    }

    String uid = '';
    try {
      uid = FirebaseAuth.instance.currentUser?.uid ?? '';
    } catch (e) {
      debugPrint('SecureCredentials: Firebase Auth unavailable: $e');
    }
    _derivedKey = _deriveFallbackKey(uid);
    _initialized = true;
  }

  static String _deriveFallbackKey(String uid) {
    final raw = '$uid|axon-secure-fallback-v2';
    if (raw.length >= 32) {
      return raw.substring(0, 32);
    }
    return raw.padRight(32, 'x');
  }

  Future<void> storeCredential(String key, String value) async {
    await initialize();

    if (_secureStorage != null) {
      await _secureStorage!.write(key: key, value: value);
    } else {
      await _storeFallbackEncrypted(key, value);
    }
  }

  Future<String?> getCredential(String key) async {
    await initialize();

    if (_secureStorage != null) {
      return await _secureStorage!.read(key: key);
    } else {
      return await _getFallbackEncrypted(key);
    }
  }

  Future<void> deleteCredential(String key) async {
    await initialize();

    if (_secureStorage != null) {
      await _secureStorage!.delete(key: key);
    } else {
      await _deleteFallbackEncrypted(key);
    }
  }

  Future<ApiCredentials> getAllCredentials() async {
    await initialize();

    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_credentialsKey);

    if (raw != null && raw.isNotEmpty) {
      try {
        final decoded = _decodeFallback(raw);
        return ApiCredentials(
          supabaseUrl: dotenv.env['SUPABASE_URL'] ??
              dotenv.env['NEXT_PUBLIC_SUPABASE_URL'] ??
              decoded[CredentialKey.supabaseUrl],
          supabaseAnonKey: dotenv.env['SUPABASE_ANON_KEY'] ??
              dotenv.env['NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY'] ??
              decoded[CredentialKey.supabaseAnonKey],
          serperApiKey: dotenv.env['SERPER_API_KEY'] ?? decoded[CredentialKey.serperApiKey],
          cloudinaryCloud: dotenv.env['CLOUDINARY_CLOUD_NAME'] ?? decoded[CredentialKey.cloudinaryCloud],
          cloudinaryPreset: dotenv.env['CLOUDINARY_UPLOAD_PRESET'] ?? decoded[CredentialKey.cloudinaryPreset],
          grokApiKey: dotenv.env['GROK_API_KEY'] ?? decoded[CredentialKey.grokApiKey],
          vercelApiKey: dotenv.env['VERCEL_API_KEY'] ?? decoded[CredentialKey.vercelApiKey],
          openrouterApiKey: dotenv.env['OPENROUTER_API_KEY'] ?? decoded[CredentialKey.openrouterApiKey],
          deepseekApiKey: dotenv.env['DEEPSEEK_API_KEY'] ?? decoded[CredentialKey.deepseekApiKey],
          googleDrivePrivateKey: dotenv.env['GOOGLE_PRIVATE_KEY'] ?? decoded[CredentialKey.googleDrivePrivateKey],
        );
      } catch (e) {
        debugPrint('SecureCredentials: Failed to load credentials');
      }
    }

    // Direct from .env
    return ApiCredentials(
      supabaseUrl: dotenv.env['SUPABASE_URL'] ??
          dotenv.env['NEXT_PUBLIC_SUPABASE_URL'],
      supabaseAnonKey: dotenv.env['SUPABASE_ANON_KEY'] ??
          dotenv.env['NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY'],
      serperApiKey: dotenv.env['SERPER_API_KEY'],
      cloudinaryCloud: dotenv.env['CLOUDINARY_CLOUD_NAME'],
      cloudinaryPreset: dotenv.env['CLOUDINARY_UPLOAD_PRESET'],
      grokApiKey: dotenv.env['GROK_API_KEY'],
      vercelApiKey: dotenv.env['VERCEL_API_KEY'],
      openrouterApiKey: dotenv.env['OPENROUTER_API_KEY'],
      deepseekApiKey: dotenv.env['DEEPSEEK_API_KEY'],
      googleDrivePrivateKey: dotenv.env['GOOGLE_PRIVATE_KEY'],
    );
  }

  Future<void> storeAllCredentials(ApiCredentials credentials) async {
    await initialize();

    final data = <String, String>{};
    if (credentials.supabaseUrl != null) {
      data[CredentialKey.supabaseUrl] = credentials.supabaseUrl!;
    }
    if (credentials.supabaseAnonKey != null) {
      data[CredentialKey.supabaseAnonKey] = credentials.supabaseAnonKey!;
    }
    if (credentials.geminiApiKey != null) {
      data[CredentialKey.geminiApiKey] = credentials.geminiApiKey!;
    }
    if (credentials.serperApiKey != null) {
      data[CredentialKey.serperApiKey] = credentials.serperApiKey!;
    }
    if (credentials.cloudinaryCloud != null) {
      data[CredentialKey.cloudinaryCloud] = credentials.cloudinaryCloud!;
    }
    if (credentials.cloudinaryPreset != null) {
      data[CredentialKey.cloudinaryPreset] = credentials.cloudinaryPreset!;
    }
    if (credentials.grokApiKey != null) {
      data[CredentialKey.grokApiKey] = credentials.grokApiKey!;
    }
    if (credentials.vercelApiKey != null) {
      data[CredentialKey.vercelApiKey] = credentials.vercelApiKey!;
    }
    if (credentials.openrouterApiKey != null) {
      data[CredentialKey.openrouterApiKey] = credentials.openrouterApiKey!;
    }
    if (credentials.deepseekApiKey != null) {
      data[CredentialKey.deepseekApiKey] = credentials.deepseekApiKey!;
    }
    if (credentials.googleDrivePrivateKey != null) {
      data[CredentialKey.googleDrivePrivateKey] = credentials.googleDrivePrivateKey!;
    }

    if (_secureStorage != null) {
      for (final entry in data.entries) {
        await _secureStorage!.write(key: entry.key, value: entry.value);
      }
    } else {
      final encoded = _encodeFallback(data);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_credentialsKey, encoded);
    }
  }

  Future<void> clearAllCredentials() async {
    await initialize();

    if (_secureStorage != null) {
      await _secureStorage!.deleteAll();
    }

    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_credentialsKey);
    await prefs.remove(_fallbackPrefsKey);
  }

  // Basic XOR-based obfuscation for fallback storage
  // NOT cryptographically secure - only prevents casual reading
  String _encodeFallback(Map<String, String> data) {
    final json = jsonEncode(data);
    final key = _derivedKey ?? '';
    final buffer = StringBuffer();

    for (int i = 0; i < json.length; i++) {
      final charCode = json.codeUnitAt(i);
      final keyChar = key.codeUnitAt(i % key.length);
      buffer.write(String.fromCharCode(charCode ^ keyChar));
    }

    return base64Encode(utf8.encode(buffer.toString()));
  }

  Map<String, String> _decodeFallback(String encoded) {
    try {
      final decoded = utf8.decode(base64Decode(encoded));
      final key = _derivedKey ?? '';
      final buffer = StringBuffer();

      for (int i = 0; i < decoded.length; i++) {
        final charCode = decoded.codeUnitAt(i);
        final keyChar = key.codeUnitAt(i % key.length);
        buffer.write(String.fromCharCode(charCode ^ keyChar));
      }

      final json = buffer.toString();
      return Map<String, String>.from(jsonDecode(json));
    } catch (e) {
      return {};
    }
  }

  Future<void> _storeFallbackEncrypted(String key, String value) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_fallbackPrefsKey);
    final data = raw != null ? _decodeFallback(raw) : <String, String>{};
    data[key] = value;
    await prefs.setString(_fallbackPrefsKey, _encodeFallback(data));
  }

  Future<String?> _getFallbackEncrypted(String key) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_fallbackPrefsKey);
    if (raw == null) return null;
    final data = _decodeFallback(raw);
    return data[key];
  }

  Future<void> _deleteFallbackEncrypted(String key) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_fallbackPrefsKey);
    if (raw == null) return;
    final data = _decodeFallback(raw);
    data.remove(key);
    await prefs.setString(_fallbackPrefsKey, _encodeFallback(data));
  }

  // Verify credentials are well-formed
  bool isValidApiKey(String? key) {
    if (key == null || key.isEmpty) return false;
    if (key.contains(' ') || key.contains('\n')) return false;
    if (key.length < 10) return false;
    if (key.contains('[YOUR_') || key.contains('your-')) return false;
    return true;
  }
}

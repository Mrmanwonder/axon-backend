// lib/services/google_drive_service.dart
// Google Drive addon for PDF uploads using Google Sign-In

import 'dart:convert';
import 'dart:io';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'package:google_sign_in/google_sign_in.dart';

class GoogleDriveService {
  static final GoogleDriveService _instance = GoogleDriveService._internal();
  factory GoogleDriveService() => _instance;
  GoogleDriveService._internal();

  static const String _folderIdKey = 'google_drive_folder_id';
  static const String _tokenKey = 'google_drive_token';
  static const _secureStorage = FlutterSecureStorage();

  final GoogleSignIn _googleSignIn = GoogleSignIn(
    scopes: [
      'https://www.googleapis.com/auth/drive.file',
      'https://www.googleapis.com/auth/drive.readonly',
    ],
  );

  String? _accessToken;

  bool get isAuthenticated => _accessToken != null;
  GoogleSignInAccount? get currentAccount => _googleSignIn.currentUser;

  Future<bool> signIn() async {
    try {
      final account = await _googleSignIn.signIn();
      if (account != null) {
        final auth = await account.authentication;
        _accessToken = auth.accessToken;

        await _secureStorage.write(key: _tokenKey, value: _accessToken!);

        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  Future<void> initFromStoredToken() async {
    final storedToken = await _secureStorage.read(key: _tokenKey);
    if (storedToken != null && storedToken.isNotEmpty) {
      _accessToken = storedToken;
    }
  }

  /// List all files and folders from Drive (root or specific folder)
  Future<List<GoogleDriveItem>> listItems({String? folderId}) async {
    if (_accessToken == null) return [];

    try {
      String query;
      if (folderId != null) {
        query = "'$folderId' in parents and trashed = false";
      } else {
        query = "'root' in parents and trashed = false";
      }
      final encodedQuery = Uri.encodeComponent(query);

      final response = await http.get(
        Uri.parse(
            'https://www.googleapis.com/drive/v3/files?q=$encodedQuery&orderBy=name&fields=files(id,name,mimeType,size,modifiedTime,parents)'),
        headers: {'Authorization': 'Bearer $_accessToken'},
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final files = data['files'] as List? ?? [];
        return files
            .map((f) => GoogleDriveItem(
                  id: f['id'] ?? '',
                  name: f['name'] ?? '',
                  mimeType: f['mimeType'] ?? '',
                  size: f['size'],
                  modifiedTime: f['modifiedTime'] != null
                      ? DateTime.tryParse(f['modifiedTime'])
                      : null,
                  isFolder:
                      f['mimeType'] == 'application/vnd.google-apps.folder',
                ))
            .toList();
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  /// Get file content (for downloading PDF files)
  Future<http.StreamedResponse?> downloadFile(String fileId) async {
    if (_accessToken == null) return null;

    try {
      final client = http.Client();
      final request = http.Request(
        'GET',
        Uri.parse(
            'https://www.googleapis.com/drive/v3/files/$fileId?alt=media'),
      );
      request.headers['Authorization'] = 'Bearer $_accessToken';

      return await client.send(request);
    } catch (e) {
      return null;
    }
  }

  /// Get web view link for a file
  Future<String?> getWebViewLink(String fileId) async {
    if (_accessToken == null) return null;

    try {
      final response = await http.get(
        Uri.parse(
            'https://www.googleapis.com/drive/v3/files/$fileId?fields=webViewLink'),
        headers: {'Authorization': 'Bearer $_accessToken'},
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['webViewLink'];
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  /// Check if user has Drive access
  Future<bool> hasAccess() async {
    if (_accessToken == null) {
      return await signIn();
    }
    return true;
  }

  Future<void> signOut() async {
    await _googleSignIn.signOut();
    _accessToken = null;

    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_folderIdKey);
  }

  Future<List<GoogleDriveItem>> searchFiles(String query) async {
    if (_accessToken == null) return [];

    try {
      final encodedQuery = Uri.encodeComponent(
        "name contains '$query' and mimeType='application/pdf' and trashed=false",
      );
      final response = await http.get(
        Uri.parse(
            'https://www.googleapis.com/drive/v3/files?q=$encodedQuery&orderBy=modifiedTime desc&fields=files(id,name,mimeType,size,modifiedTime)'),
        headers: {'Authorization': 'Bearer $_accessToken'},
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final files = data['files'] as List? ?? [];

        return files
            .map((f) => GoogleDriveItem(
                  id: f['id'] ?? '',
                  name: f['name'] ?? '',
                  mimeType: f['mimeType'] ?? '',
                  size: f['size'],
                  modifiedTime: f['modifiedTime'] != null
                      ? DateTime.tryParse(f['modifiedTime'])
                      : null,
                  isFolder: false,
                ))
            .toList();
      }
      return [];
    } catch (e) {
      return [];
    }
  }

  Future<String?> downloadAndSave(String fileId, String fileName) async {
    if (_accessToken == null) return null;

    try {
      final response = await http.get(
        Uri.parse(
            'https://www.googleapis.com/drive/v3/files/$fileId?alt=media'),
        headers: {'Authorization': 'Bearer $_accessToken'},
      );

      if (response.statusCode == 200) {
        final dir = Directory.systemTemp;
        final tempFile = File('${dir.path}/$fileName');
        await tempFile.writeAsBytes(response.bodyBytes);
        return tempFile.path;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
}

class GoogleDriveItem {
  final String id;
  final String name;
  final String mimeType;
  final String? size;
  final DateTime? modifiedTime;
  final bool isFolder;

  GoogleDriveItem({
    required this.id,
    required this.name,
    required this.mimeType,
    this.size,
    this.modifiedTime,
    this.isFolder = false,
  });

  bool get isPdf =>
      mimeType == 'application/pdf' || name.toLowerCase().endsWith('.pdf');
}

import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'caie_models.dart';
import 'api_client.dart';

class R2PaperService {
  static const String _baseUrl =
      'https://pub-2d1af1d72ef54aef84cbf8de76bd216c.r2.dev';

  static final R2PaperService _instance = R2PaperService._internal();
  factory R2PaperService() => _instance;
  R2PaperService._internal();

  final Dio _dio = ApiClient.sharedDio;

  String objectKeyFor(CaiePaper paper) {
    return objectKey(
      subjectCode: paper.subjectCode,
      session: paper.session,
      year: paper.year,
      variant: paper.variant,
      type: 'qp',
    );
  }

  String objectKey({
    required String subjectCode,
    required String session,
    required int year,
    required String variant,
    required String type,
  }) {
    final yearShort = (year % 100).toString().padLeft(2, '0');
    final sessionPath = _sessionToPath(session, yearShort);
    final sessionFile = _sessionToFile(session);
    return '$subjectCode/$sessionPath/$variant/${subjectCode}_$sessionFile$yearShort${_typeSuffix(type, variant)}';
  }

  String _sessionToPath(String session, String yearShort) {
    switch (session.toLowerCase()) {
      case 's':
      case 'summer':
        return 's$yearShort';
      case 'w':
      case 'winter':
      case 'o':
      case 'o_n':
        return 'o_n$yearShort';
      case 'm':
      case 'march':
        return 'm$yearShort';
      default:
        return '${session}_$yearShort';
    }
  }

  String _sessionToFile(String session) {
    switch (session.toLowerCase()) {
      case 's':
      case 'summer':
        return 's';
      case 'w':
      case 'winter':
      case 'o':
      case 'o_n':
        return 'w';
      case 'm':
      case 'march':
        return 'm';
      default:
        return session;
    }
  }

  String _typeSuffix(String type, String variant) {
    final t = type.toLowerCase();
    if (t == 'question paper' || t == 'qp') return '_qp_$variant.pdf';
    if (t == 'mark scheme' || t == 'ms') return '_ms_$variant.pdf';
    if (t == 'grade thresholds' || t == 'gt') return '_gt_$variant.pdf';
    if (t == 'examiner report' || t == 'er') return '_er_$variant.pdf';
    return '_${t}_$variant.pdf';
  }

  String downloadUrlFor(CaiePaper paper) {
    return '$_baseUrl/${objectKeyFor(paper)}';
  }

  Future<String> downloadPaper(
    CaiePaper paper, {
    void Function(int received, int total)? onProgress,
  }) async {
    final localPath = await _localPathFor(paper);
    final file = File(localPath);
    if (await file.exists()) return localPath;

    final url = downloadUrlFor(paper);
    debugPrint('[R2] Downloading: $url');

    try {
      await _dio.download(
        url,
        localPath,
        onReceiveProgress: onProgress,
        options: Options(
          responseType: ResponseType.bytes,
          receiveTimeout: const Duration(seconds: 60),
        ),
      );

      if (!await _isValidPdf(localPath)) {
        await file.delete();
        throw R2InvalidContentException(
          'Downloaded file is not a valid PDF (may be HTML)',
        );
      }

      debugPrint('[R2] Downloaded to: $localPath');
      return localPath;
    } catch (e) {
      if (e is R2InvalidContentException) rethrow;
      debugPrint('[R2] Download failed: $e');
      rethrow;
    }
  }

  Future<bool> _isValidPdf(String path) async {
    try {
      final file = File(path);
      if (!await file.exists()) return false;
      final firstBytes = await file.openRead(0, 5).first;
      if (firstBytes.length < 5) return false;
      return String.fromCharCodes(firstBytes) == '%PDF-';
    } catch (_) {
      return false;
    }
  }

  Future<String> _localPathFor(CaiePaper paper) async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/r2_papers/${paper.subjectCode}');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return '${dir.path}/${paper.id}.pdf';
  }
}

class R2InvalidContentException implements Exception {
  final String message;
  const R2InvalidContentException(this.message);
  @override
  String toString() => message;
}

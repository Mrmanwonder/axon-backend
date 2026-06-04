import 'dart:typed_data';

import '../models/models.dart';
import 'document_vision_runtime.dart';

typedef VisionCapabilitiesProvider = Future<VisionRuntimeCapabilities> Function();
typedef PythonEngineRunner = Future<Map<String, dynamic>?> Function(String filePath);
typedef MobileFallbackExtractor = Future<List<PdfQuestion>> Function(String filePath);
typedef QuestionEnricher = Future<List<PdfQuestion>> Function(String filePath, List<PdfQuestion> questions);
typedef AsyncJobExtractor = Future<List<PdfQuestion>> Function(String filePath);

class PdfService {
  PdfService({
    VisionCapabilitiesProvider? capabilitiesProvider,
    PythonEngineRunner? pythonEngineRunner,
    MobileFallbackExtractor? mobileFallbackExtractor,
    QuestionEnricher? questionEnricher,
    AsyncJobExtractor? asyncJobExtractor,
  });

  Map<String, dynamic> get lastVisionDiagnostics => {};

  Future<String> loadOrExtractText(String filePath) async => '';
  Future<List<String>> extractChapters(String filePath) async => [];
  Future<String> extractText(String filePath) async => '';
  Future<List<PdfQuestion>> extractQuestionsV2(String filePath) async => [];
}

import '../models/models.dart';
import 'document_vision_runtime.dart';

typedef VisionCapabilitiesProvider = Future<VisionRuntimeCapabilities>
    Function();
typedef PythonEngineRunner = Future<Map<String, dynamic>?> Function(
    String filePath);
typedef AsyncJobExtractor = Future<List<PdfQuestion>> Function(String filePath);
typedef MobileFallbackExtractor = Future<List<PdfQuestion>> Function(
    String filePath);
typedef QuestionEnricher = Future<List<PdfQuestion>> Function(
  String filePath,
  List<PdfQuestion> questions,
);

class PdfService {
  final VisionCapabilitiesProvider capabilitiesProvider;
  final PythonEngineRunner? pythonEngineRunner;
  final AsyncJobExtractor? asyncJobExtractor;
  final MobileFallbackExtractor mobileFallbackExtractor;
  final QuestionEnricher questionEnricher;
  Map<String, dynamic> lastVisionDiagnostics = const {};

  PdfService({
    VisionCapabilitiesProvider? capabilitiesProvider,
    this.pythonEngineRunner,
    this.asyncJobExtractor,
    MobileFallbackExtractor? mobileFallbackExtractor,
    QuestionEnricher? questionEnricher,
  })  : capabilitiesProvider = capabilitiesProvider ??
            (() async => const VisionRuntimeCapabilities(
                  supportsPythonOffline: false,
                  supportsMobileFallback: true,
                  embeddedPython: false,
                  platform: 'unknown',
                )),
        mobileFallbackExtractor =
            mobileFallbackExtractor ?? ((_) async => const []),
        questionEnricher =
            questionEnricher ?? ((_, questions) async => questions);

  Future<List<PdfQuestion>> loadCachedQuestions(String filePath) async => [];
  Future<List<PdfQuestion>> extractQuestions(String filePath) =>
      extractQuestionsV2(filePath);
  Future<void> cacheQuestions(
      String filePath, List<PdfQuestion> questions) async {}
  Future<String> loadOrExtractText(String filePath) async => '';
  Future<void> clearCachedQuestions(String filePath) async {}
  Future<Map<String, dynamic>?> autoFindMarkingScheme(String filePath) async =>
      null;
  Future<String> extractText(String path) async => '';
  Future<List<String>> extractChapters(String filePath) async => const [];

  Future<List<PdfQuestion>> extractQuestionsV2(String filePath) async {
    final capabilities = await capabilitiesProvider();

    if (capabilities.supportsPythonOffline && pythonEngineRunner != null) {
      final envelope = await pythonEngineRunner!(filePath);
      final rawQuestions = envelope?['questions'];
      if (rawQuestions is List && rawQuestions.isNotEmpty) {
        lastVisionDiagnostics = Map<String, dynamic>.from(
            envelope?['diagnostics'] as Map? ?? const {});
        final questions = rawQuestions
            .whereType<Map>()
            .map(
                (item) => PdfQuestion.fromJson(Map<String, dynamic>.from(item)))
            .toList();
        return questionEnricher(filePath, questions);
      }
    }

    if (!capabilities.supportsPythonOffline && asyncJobExtractor != null) {
      final questions = await asyncJobExtractor!(filePath);
      if (questions.isNotEmpty) {
        lastVisionDiagnostics = const {'engine': 'backend-async-job'};
        return questionEnricher(filePath, questions);
      }
    }

    final fallback = await mobileFallbackExtractor(filePath);
    lastVisionDiagnostics = const {'engine': 'mobile-fallback'};
    return questionEnricher(filePath, fallback);
  }
}

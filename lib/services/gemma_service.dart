class GemmaService {
  GemmaService._();

  static final GemmaService instance = GemmaService._();

  bool get isReady => false;

  Future<bool> initialize() async => false;

  Future<String> chat({
    required String message,
    String? context,
    int? maxNewTokens,
  }) async {
    return '';
  }
}

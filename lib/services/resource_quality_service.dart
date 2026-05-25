class ResourceQualityService {
  static final ResourceQualityService _instance =
      ResourceQualityService._internal();
  factory ResourceQualityService() => _instance;
  ResourceQualityService._internal();

  Future<Map<String, double>> getQualityScores() async {
    return {};
  }

  Map<String, double> get qualityScores => {};

  Future<void> recordOpen({
    required String resourceId,
    required String source,
  }) async {}

  Future<void> recordRating({
    required String resourceId,
    required String source,
    required double rating,
  }) async {}

  Future<void> recordUsefulness({
    required String resourceId,
    required String source,
    required bool wasUseful,
  }) async {}
}

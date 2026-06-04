import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/caie_models.dart';
import '../services/caie_offline_store.dart';
import '../services/caie_search_index.dart';
import '../services/caie_topic_mapping.dart';
import '../services/caie_topic_engine.dart';
import '../services/caie_recommendation_engine.dart';
import '../services/caie_command_palette.dart';

final caieOfflineStoreProvider = Provider<CaieOfflineStore>((ref) {
  final store = CaieOfflineStore();
  return store;
});

final caieInitProvider = FutureProvider<void>((ref) async {
  await ref.read(caieOfflineStoreProvider).init();
});

final caieSearchIndexProvider = Provider<CaieSearchIndex>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  return CaieSearchIndex(store);
});

final caieTopicMappingProvider = Provider<CaieTopicMapping>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  return CaieTopicMapping(store);
});

final caieTopicEngineProvider = Provider<CaieTopicEngine>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  final mapping = ref.read(caieTopicMappingProvider);
  return CaieTopicEngine(store, mapping);
});

final caieRecommendationEngineProvider = Provider<CaieRecommendationEngine>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  final engine = ref.read(caieTopicEngineProvider);
  return CaieRecommendationEngine(store, engine);
});

final caieCommandPaletteProvider = Provider<CaieCommandPaletteService>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  final index = ref.read(caieSearchIndexProvider);
  return CaieCommandPaletteService(store, index);
});

final caieAllPapersProvider = Provider<List<CaiePaper>>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  return store.getAllPapers();
});

final caiePapersBySubjectProvider =
    Provider.family<List<CaiePaper>, String>((ref, subjectCode) {
  final store = ref.read(caieOfflineStoreProvider);
  return store.getPapersBySubject(subjectCode);
});

final caieRecentPapersProvider = Provider<List<CaiePaper>>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  return store.getRecentPapers();
});

final caieBookmarkedPapersProvider = Provider<List<CaiePaper>>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  return store.getBookmarkedPapers();
});

final caieDownloadedPapersProvider = Provider<List<CaiePaper>>((ref) {
  final store = ref.read(caieOfflineStoreProvider);
  return store.getDownloadedPapers();
});

final caieTopicPerformanceProvider =
    Provider.family<List<CaieTopicPerformance>, String>((ref, subjectCode) {
  final engine = ref.read(caieTopicEngineProvider);
  return engine.getTopicPerformance(subjectCode);
});

final caieWeakTopicsProvider =
    Provider.family<List<CaieTopicPerformance>, String>((ref, subjectCode) {
  final engine = ref.read(caieTopicEngineProvider);
  return engine.getWeakTopics(subjectCode);
});

final caieRecommendationsProvider = Provider<List<CaieRecommendation>>((ref) {
  final engine = ref.read(caieRecommendationEngineProvider);
  return engine.getRecommendations();
});

final caieSearchProvider =
    Provider.family<List<Map<String, dynamic>>, String>((ref, query) {
  final palette = ref.read(caieCommandPaletteProvider);
  return palette.search(query);
});

final caiePaperByIdProvider = Provider.family<CaiePaper?, String>((ref, id) {
  final store = ref.read(caieOfflineStoreProvider);
  return store.getPaper(id);
});

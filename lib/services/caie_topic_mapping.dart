import 'caie_models.dart';
import 'caie_offline_store.dart';

class CaieTopicMapping {
  final CaieOfflineStore _store;

  CaieTopicMapping(this._store);

  List<CaieTopic> getTopicsForSubject(String subjectCode) =>
      kCaieTopicMap[subjectCode] ?? [];

  List<CaiePaper> getPapersForTopic(String topicId) {
    return _store.getAllPapers().where((p) => p.topics.contains(topicId)).toList();
  }

  List<CaiePaper> getPapersForSubjectAndTopics(
      String subjectCode, List<String> topicIds) {
    return _store
        .getPapersBySubject(subjectCode)
        .where((p) => p.topics.any((t) => topicIds.contains(t)))
        .toList();
  }

  List<String> suggestTopics(String subjectCode, String query) {
    final topics = getTopicsForSubject(subjectCode);
    final lower = query.toLowerCase();
    return topics
        .where((t) =>
            t.name.toLowerCase().contains(lower) ||
            t.id.toLowerCase().contains(lower))
        .map((t) => t.id)
        .toList();
  }
}

import 'caie_models.dart';
import 'caie_offline_store.dart';
import 'caie_topic_mapping.dart';

class CaieTopicEngine {
  final CaieOfflineStore _store;
  final CaieTopicMapping _mapping;

  CaieTopicEngine(this._store, this._mapping);

  List<CaieTopicPerformance> getTopicPerformance(String subjectCode) {
    final records = _store.getQuestionRecords();
    final subjectRecords = records.where((r) {
      final paper = _store.getPaper(r.paperId);
      return paper?.subjectCode == subjectCode;
    }).toList();

    if (subjectRecords.isEmpty) return [];

    final byTopic = <String, List<CaieQuestionRecord>>{};
    for (final r in subjectRecords) {
      byTopic.putIfAbsent(r.topicId, () => []).add(r);
    }

    final topics = _mapping.getTopicsForSubject(subjectCode);
    final result = <CaieTopicPerformance>[];
    for (final topic in topics) {
      final records = byTopic[topic.id];
      if (records == null || records.isEmpty) continue;
      final correct = records.where((r) => r.correct).length;
      final accuracy = correct / records.length;
      result.add(CaieTopicPerformance(
        topicId: topic.id,
        topicName: topic.name,
        subjectCode: subjectCode,
        attempted: records.length,
        correct: correct,
        accuracy: accuracy,
      ));
    }

    result.sort((a, b) => a.accuracy.compareTo(b.accuracy));
    return result;
  }

  CaieTopicPerformance? getWeakestTopic(String subjectCode) {
    final perf = getTopicPerformance(subjectCode);
    if (perf.isEmpty) return null;
    return perf.first;
  }

  List<CaieTopicPerformance> getWeakTopics(String subjectCode,
      {double threshold = 0.5}) {
    return getTopicPerformance(subjectCode)
        .where((t) => t.accuracy < threshold)
        .toList();
  }

  Map<String, double> getTopicAccuracyMap(String subjectCode) {
    final perf = getTopicPerformance(subjectCode);
    return {for (final p in perf) p.topicId: p.accuracy};
  }
}

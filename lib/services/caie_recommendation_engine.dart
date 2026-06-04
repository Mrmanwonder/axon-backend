import 'caie_models.dart';
import 'caie_offline_store.dart';
import 'caie_topic_engine.dart';

class CaieRecommendationEngine {
  final CaieOfflineStore _store;
  final CaieTopicEngine _topicEngine;

  CaieRecommendationEngine(this._store, this._topicEngine);

  List<CaieRecommendation> getRecommendations({
    int limit = 10,
    List<String>? subjectCodes,
  }) {
    final results = <CaieRecommendation>[];
    final usedPaperIds = <String>{};
    final weakTopicsBySubject = <String, List<CaieTopicPerformance>>{};

    final subjects = subjectCodes ??
        _store
            .getAllPapers()
            .map((p) => p.subjectCode)
            .toSet()
            .toList();

    for (final code in subjects) {
      final weak = _topicEngine.getWeakTopics(code);
      if (weak.isNotEmpty) {
        weakTopicsBySubject[code] = weak;
      }
    }

    for (final entry in weakTopicsBySubject.entries) {
      for (final weakTopic in entry.value) {
        final papers = _store
            .getPapersBySubject(entry.key)
            .where((p) =>
                p.topics.contains(weakTopic.topicId) && !p.solved)
            .toList();

        papers.sort((a, b) {
          if (a.solved != b.solved) return a.solved ? 1 : -1;
          if (a.difficulty != b.difficulty) {
            return (a.difficulty - weakTopic.accuracy).abs()
                .compareTo((b.difficulty - weakTopic.accuracy).abs());
          }
          return (a.lastOpened?.millisecondsSinceEpoch ?? 0)
              .compareTo(b.lastOpened?.millisecondsSinceEpoch ?? 0);
        });

        for (final paper in papers.take(2)) {
          if (usedPaperIds.contains(paper.id)) continue;
          usedPaperIds.add(paper.id);
          results.add(CaieRecommendation(
            paperId: paper.id,
            displayName: paper.displayName,
            reason:
                'Weak in "${weakTopic.topicName}" (${(weakTopic.accuracy * 100).toStringAsFixed(0)}% accuracy)',
            score: (1.0 - weakTopic.accuracy) + paper.difficulty,
          ));
        }
      }
    }

    final recentPapers = _store.getRecentPapers();
    if (recentPapers.isNotEmpty) {
      final latest = recentPapers.first;
      final sameSubjectPapers = _store
          .getPapersBySubject(latest.subjectCode)
          .where((p) => !p.solved && !usedPaperIds.contains(p.id))
          .toList();

      sameSubjectPapers.sort((a, b) => a.year.compareTo(b.year));
      for (final paper in sameSubjectPapers.take(2)) {
        if (usedPaperIds.contains(paper.id)) continue;
        usedPaperIds.add(paper.id);
        results.add(CaieRecommendation(
          paperId: paper.id,
          displayName: paper.displayName,
          reason: 'Continue with ${latest.subjectCode} — try ${paper.displayName}',
          score: 0.5 + paper.difficulty * 0.3,
        ));
      }
    }

    results.sort((a, b) => b.score.compareTo(a.score));
    return results.take(limit).toList();
  }

  List<CaieRecommendation> getRecommendedForSubject(String subjectCode,
      {int limit = 5}) {
    return getRecommendations(limit: limit, subjectCodes: [subjectCode]);
  }
}

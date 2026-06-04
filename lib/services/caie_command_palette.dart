import 'caie_offline_store.dart';
import 'caie_search_index.dart';

class _Suggestion {
  final String label;
  final String subtitle;
  final String? paperId;
  final double score;
  final String type;

  const _Suggestion({
    required this.label,
    required this.subtitle,
    this.paperId,
    required this.score,
    required this.type,
  });
}

class CaieCommandPaletteService {
  final CaieOfflineStore _store;
  final CaieSearchIndex _searchIndex;

  CaieCommandPaletteService(this._store, this._searchIndex);

  List<_Suggestion> _shortcuts = [];
  bool _shortcutsLoaded = false;

  List<String> get builtinCommands => const [
        '/recent',
        '/bookmarks',
        '/downloaded',
        '/weak',
        '/recommendations',
        '/unsolved',
      ];

  void loadBuiltinShortcuts() {
    if (_shortcutsLoaded) return;
    _shortcuts = [
      _Suggestion(
          label: '/recent',
          subtitle: 'Browse recently opened papers',
          score: 1.0,
          type: 'command'),
      _Suggestion(
          label: '/bookmarks',
          subtitle: 'Browse bookmarked papers',
          score: 1.0,
          type: 'command'),
      _Suggestion(
          label: '/downloaded',
          subtitle: 'Browse downloaded papers',
          score: 1.0,
          type: 'command'),
      _Suggestion(
          label: '/weak',
          subtitle: 'View weak topics analysis',
          score: 1.0,
          type: 'command'),
      _Suggestion(
          label: '/recommendations',
          subtitle: 'View recommended papers',
          score: 1.0,
          type: 'command'),
      _Suggestion(
          label: '/unsolved',
          subtitle: 'Browse unsolved papers',
          score: 1.0,
          type: 'command'),
    ];
    _shortcutsLoaded = true;
  }

  List<Map<String, dynamic>> search(String query, {int limit = 8}) {
    if (query.isEmpty) return [];
    loadBuiltinShortcuts();
    final results = <Map<String, dynamic>>[];
    final lower = query.toLowerCase();

    for (final cmd in _shortcuts) {
      if (cmd.label.toLowerCase().contains(lower)) {
        results.add({
          'type': cmd.type,
          'label': cmd.label,
          'subtitle': cmd.subtitle,
          'paper_id': cmd.paperId,
        });
      }
    }

    final tokens = lower.split(RegExp(r'\s+'));
    final hasFilters =
        tokens.any((t) => ['recent', 'downloaded', 'bookmarked', 'unsolved', 'solved'].contains(t));

    for (final token in tokens) {
      if (token == 'recent') {
        for (final paper in _store.getRecentPapers().take(5)) {
          results.add({
            'type': 'paper',
            'label': paper.displayName,
            'subtitle': paper.subject,
            'paper_id': paper.id,
          });
        }
      }
      if (token == 'downloaded') {
        for (final paper in _store.getDownloadedPapers().take(5)) {
          results.add({
            'type': 'paper',
            'label': paper.displayName,
            'subtitle': 'Downloaded — ${paper.subject}',
            'paper_id': paper.id,
          });
        }
      }
      if (token == 'bookmarked') {
        for (final paper in _store.getBookmarkedPapers().take(5)) {
          results.add({
            'type': 'paper',
            'label': paper.displayName,
            'subtitle': 'Bookmarked — ${paper.subject}',
            'paper_id': paper.id,
          });
        }
      }
      if (token == 'unsolved') {
        for (final paper in _store.getAllPapers().where((p) => !p.solved).take(5)) {
          results.add({
            'type': 'paper',
            'label': paper.displayName,
            'subtitle': 'Unsolved — ${paper.subject}',
            'paper_id': paper.id,
          });
        }
      }
    }

    if (!hasFilters) {
      final searchResults = _searchIndex.search(query);
      for (final paper in searchResults) {
        results.add({
          'type': 'paper',
          'label': paper.displayName,
          'subtitle': '${paper.subject} — ${paper.topics.take(3).join(', ')}',
          'paper_id': paper.id,
        });
      }
    }

    results.sort((a, b) {
      final aScore = _resultScore(a, lower);
      final bScore = _resultScore(b, lower);
      return bScore.compareTo(aScore);
    });

    final seen = <String>{};
    return results
        .where((r) {
          final key = '${r['type']}:${r['paper_id'] ?? r['label']}';
          return seen.add(key);
        })
        .take(limit)
        .toList();
  }

  double _resultScore(Map<String, dynamic> result, String query) {
    double score = 0;
    if (result['type'] == 'command') score += 2.0;
    if ((result['label'] as String).toLowerCase().startsWith(query)) score += 1.0;
    return score;
  }
}

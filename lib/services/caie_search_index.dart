import 'caie_models.dart';
import 'caie_offline_store.dart';

class _IndexEntry {
  final String paperId;
  final double relevance;

  const _IndexEntry(this.paperId, this.relevance);
}

class CaieSearchIndex {
  final CaieOfflineStore _store;

  CaieSearchIndex(this._store);

  final Map<String, List<_IndexEntry>> _termIndex = {};
  bool _built = false;

  void build() {
    _termIndex.clear();
    for (final paper in _store.getAllPapers()) {
      _indexPaper(paper);
    }
    _built = true;
  }

  void indexPaper(CaiePaper paper) {
    _termIndex.clear();
    for (final p in _store.getAllPapers()) {
      _indexPaper(p);
    }
  }

  void _indexPaper(CaiePaper paper) {
    final terms = _tokenize(paper.subjectCode);
    for (final t in terms) {
      _termIndex.putIfAbsent(t, () => []).add(_IndexEntry(paper.id, 1.0));
    }
    final nameTerms = _tokenize(paper.subject);
    for (final t in nameTerms) {
      _termIndex.putIfAbsent(t, () => []).add(_IndexEntry(paper.id, 0.9));
    }
    _termIndex.putIfAbsent(paper.year.toString(), () => []).add(_IndexEntry(paper.id, 0.8));
    _termIndex.putIfAbsent(paper.session.toLowerCase(), () => []).add(_IndexEntry(paper.id, 0.7));
    _termIndex.putIfAbsent(paper.variant.toLowerCase(), () => []).add(_IndexEntry(paper.id, 0.7));
    for (final topic in paper.topics) {
      for (final t in _tokenize(topic)) {
        _termIndex.putIfAbsent(t, () => []).add(_IndexEntry(paper.id, 0.6));
      }
    }
  }

  List<String> _tokenize(String text) {
    return text
        .toLowerCase()
        .split(RegExp(r'[\s,./\-_()]+'))
        .where((t) => t.length >= 2)
        .toSet()
        .toList();
  }

  int _editDistance(String a, String b) {
    final m = a.length, n = b.length;
    final dp = List.generate(m + 1, (_) => List.filled(n + 1, 0));
    for (int i = 0; i <= m; i++) { dp[i][0] = i; }
    for (int j = 0; j <= n; j++) { dp[0][j] = j; }
    for (int i = 1; i <= m; i++) {
      for (int j = 1; j <= n; j++) {
        if (a[i - 1] == b[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = 1 + [dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]].reduce((x, y) => x < y ? x : y);
        }
      }
    }
    return dp[m][n];
  }

  List<CaiePaper> search(String query) {
    if (!_built) build();
    final tokens = _tokenize(query);
    if (tokens.isEmpty) return [];

    final scores = <String, double>{};
    final exactCodes = _cacheByCode();

    for (final token in tokens) {
      if (_termIndex.containsKey(token)) {
        for (final entry in _termIndex[token]!) {
          scores[entry.paperId] = (scores[entry.paperId] ?? 0) + entry.relevance;
        }
      }
      for (final code in exactCodes.keys) {
        if (code.startsWith(token) || token.startsWith(code)) {
          for (final paper in exactCodes[code]!) {
            scores[paper.id] = (scores[paper.id] ?? 0) + 0.5;
          }
        }
        if (_editDistance(code, token) <= 1) {
          for (final paper in exactCodes[code]!) {
            scores[paper.id] = (scores[paper.id] ?? 0) + 0.3;
          }
        }
      }
    }

    final boostRecent = _store.getRecentPapers();
    for (int i = 0; i < boostRecent.length; i++) {
      final id = boostRecent[i].id;
      scores[id] = (scores[id] ?? 0) + (1.0 - i * 0.02);
    }

    final sorted = scores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return sorted
        .where((e) => e.value >= 0.5)
        .map((e) => _store.getPaper(e.key))
        .whereType<CaiePaper>()
        .toList();
  }

  Map<String, List<CaiePaper>> _cacheByCode() {
    final map = <String, List<CaiePaper>>{};
    for (final paper in _store.getAllPapers()) {
      map.putIfAbsent(paper.subjectCode, () => []).add(paper);
    }
    return map;
  }

  List<CaiePaper> searchByCodePrefix(String prefix) {
    final lower = prefix.toLowerCase();
    return _store
        .getAllPapers()
        .where((p) =>
            p.subjectCode.toLowerCase().startsWith(lower) ||
            p.subject.toLowerCase().contains(lower))
        .toList();
  }
}

import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'caie_models.dart';

class CaieOfflineStore {
  CaieOfflineStore._();
  static final CaieOfflineStore _instance = CaieOfflineStore._();
  factory CaieOfflineStore() => _instance;

  static const _papersKey = 'caie_papers';
  static const _recentsKey = 'caie_recent_ids';
  static const _bookmarksKey = 'caie_bookmark_ids';
  static const _progressKey = 'caie_question_records';
  static const _maxRecents = 50;

  SharedPreferences? _prefs;
  Map<String, CaiePaper> _cache = {};

  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
    _loadCache();
  }

  void _loadCache() {
    final raw = _prefs!.getString(_papersKey);
    if (raw != null) {
      final list = jsonDecode(raw) as List<dynamic>;
      _cache = {
        for (final item in list)
          (item as Map<String, dynamic>)['id'] as String: CaiePaper.fromMap(item),
      };
    }
  }

  void _saveCache() {
    _prefs!.setString(
      _papersKey,
      jsonEncode(_cache.values.map((p) => p.toMap()).toList()),
    );
  }

  void putPaper(CaiePaper paper) {
    _cache[paper.id] = paper;
    _saveCache();
  }

  void putPapers(List<CaiePaper> papers) {
    for (final p in papers) {
      _cache[p.id] = p;
    }
    _saveCache();
  }

  CaiePaper? getPaper(String id) => _cache[id];

  List<CaiePaper> getAllPapers() => _cache.values.toList();

  List<CaiePaper> getPapersBySubject(String subjectCode) =>
      _cache.values.where((p) => p.subjectCode == subjectCode).toList();

  void markDownloaded(String paperId, {bool downloaded = true}) {
    final p = _cache[paperId];
    if (p != null) {
      _cache[paperId] = p.copyWith(downloaded: downloaded);
      _saveCache();
    }
  }

  List<String> _loadStringList(String key) =>
      _prefs!.getStringList(key) ?? [];

  void _saveStringList(String key, List<String> list) =>
      _prefs!.setStringList(key, list);

  List<CaiePaper> getRecentPapers() {
    final ids = _loadStringList(_recentsKey);
    return ids.map((id) => _cache[id]).whereType<CaiePaper>().toList();
  }

  void recordOpen(String paperId) {
    final ids = _loadStringList(_recentsKey);
    ids.remove(paperId);
    ids.insert(0, paperId);
    if (ids.length > _maxRecents) ids.removeLast();
    _saveStringList(_recentsKey, ids);

    final p = _cache[paperId];
    if (p != null) {
      _cache[paperId] = p.copyWith(
        lastOpened: DateTime.now(),
        openCount: p.openCount + 1,
      );
      _saveCache();
    }
  }

  List<CaiePaper> getBookmarkedPapers() {
    final ids = _loadStringList(_bookmarksKey);
    return ids.map((id) => _cache[id]).whereType<CaiePaper>().toList();
  }

  void toggleBookmark(String paperId) {
    final ids = _loadStringList(_bookmarksKey);
    if (ids.contains(paperId)) {
      ids.remove(paperId);
    } else {
      ids.add(paperId);
    }
    _saveStringList(_bookmarksKey, ids);

    final p = _cache[paperId];
    if (p != null) {
      _cache[paperId] = p.copyWith(bookmarked: ids.contains(paperId));
      _saveCache();
    }
  }

  bool isBookmarked(String paperId) =>
      _loadStringList(_bookmarksKey).contains(paperId);

  List<CaiePaper> getDownloadedPapers() {
    return _cache.values.where((p) => p.downloaded).toList();
  }

  Future<void> saveQuestionRecord(CaieQuestionRecord record) async {
    final list = _loadQuestionRecords();
    list.add(record);
    _prefs!.setString(
        _progressKey, jsonEncode(list.map((r) => r.toMap()).toList()));
  }

  List<CaieQuestionRecord> getQuestionRecords({String? paperId}) {
    final records = _loadQuestionRecords();
    if (paperId != null) {
      return records.where((r) => r.paperId == paperId).toList();
    }
    return records;
  }

  List<CaieQuestionRecord> _loadQuestionRecords() {
    final raw = _prefs!.getString(_progressKey);
    if (raw == null) return [];
    final list = jsonDecode(raw) as List<dynamic>;
    return list
        .map((item) => CaieQuestionRecord.fromMap(item as Map<String, dynamic>))
        .toList();
  }
}

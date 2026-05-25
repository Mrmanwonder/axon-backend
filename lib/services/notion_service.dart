// lib/services/notion_service.dart
import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// ─── Models ─────────────────────────────────────────────────────────────────

enum NotionSyncStatus { idle, syncing, error }

class NotionEvent {
  final String localId;
  final String? notionPageId;
  final String title;
  final String description;
  final String subject;
  final String topic;
  final DateTime startTime;
  final DateTime endTime;
  final int durationMinutes;
  final bool isCompleted;
  final String source;
  final DateTime updatedAt;
  final bool isDeleted;

  const NotionEvent({
    required this.localId,
    this.notionPageId,
    required this.title,
    required this.description,
    required this.subject,
    required this.topic,
    required this.startTime,
    required this.endTime,
    required this.durationMinutes,
    this.isCompleted = false,
    required this.source,
    required this.updatedAt,
    this.isDeleted = false,
  });

  NotionEvent copyWith({
    String? notionPageId,
    String? title,
    String? description,
    String? subject,
    String? topic,
    DateTime? startTime,
    DateTime? endTime,
    int? durationMinutes,
    bool? isCompleted,
    String? source,
    DateTime? updatedAt,
    bool? isDeleted,
  }) {
    return NotionEvent(
      localId: localId,
      notionPageId: notionPageId ?? this.notionPageId,
      title: title ?? this.title,
      description: description ?? this.description,
      subject: subject ?? this.subject,
      topic: topic ?? this.topic,
      startTime: startTime ?? this.startTime,
      endTime: endTime ?? this.endTime,
      durationMinutes: durationMinutes ?? this.durationMinutes,
      isCompleted: isCompleted ?? this.isCompleted,
      source: source ?? this.source,
      updatedAt: updatedAt ?? this.updatedAt,
      isDeleted: isDeleted ?? this.isDeleted,
    );
  }

  Map<String, dynamic> toJson() => {
        'localId': localId,
        'notionPageId': notionPageId,
        'title': title,
        'description': description,
        'subject': subject,
        'topic': topic,
        'startTime': startTime.toIso8601String(),
        'endTime': endTime.toIso8601String(),
        'durationMinutes': durationMinutes,
        'isCompleted': isCompleted,
        'source': source,
        'updatedAt': updatedAt.toIso8601String(),
        'isDeleted': isDeleted,
      };

  factory NotionEvent.fromJson(Map<String, dynamic> map) => NotionEvent(
        localId: map['localId'] as String,
        notionPageId: map['notionPageId'] as String?,
        title: map['title'] as String? ?? '',
        description: map['description'] as String? ?? '',
        subject: map['subject'] as String? ?? '',
        topic: map['topic'] as String? ?? '',
        startTime: DateTime.parse(map['startTime'] as String),
        endTime: DateTime.parse(map['endTime'] as String),
        durationMinutes: map['durationMinutes'] as int? ?? 60,
        isCompleted: map['isCompleted'] as bool? ?? false,
        source: map['source'] as String? ?? 'app',
        updatedAt: DateTime.parse(map['updatedAt'] as String),
        isDeleted: map['isDeleted'] as bool? ?? false,
      );

  factory NotionEvent.fromNotionPage(Map<String, dynamic> page) {
    final props = page['properties'] as Map<String, dynamic>? ?? {};

    String title = _extractText(props['Name'] ?? props['Title'] ?? props['name']);
    String localId = _extractText(props['Axon ID'] ?? props['axon_id']);
    if (localId.isEmpty) localId = 'notion_${page['id']}';

    String description = _extractText(props['Description'] ?? props['description']);
    String subject = _extractText(props['Subject'] ?? props['subject']);
    String topic = _extractText(props['Topic'] ?? props['topic']);

    final dateBlock = props['Date'] ?? props['date'];
    DateTime startTime = DateTime.now();
    DateTime endTime = DateTime.now().add(const Duration(hours: 1));
    if (dateBlock is Map && dateBlock['date'] is Map) {
      final d = dateBlock['date'] as Map<String, dynamic>;
      startTime = DateTime.tryParse(d['start'] as String? ?? '') ?? DateTime.now();
      endTime = DateTime.tryParse(d['end'] as String? ?? '') ??
          startTime.add(const Duration(hours: 1));
    }

    final durationVal = props['Duration']?['number'] as num?;
    final duration = durationVal?.toInt() ?? endTime.difference(startTime).inMinutes;
    final checkboxVal = props['Completed']?['checkbox'] as bool? ?? false;

    final lastEdited = page['last_edited_time'] as String?;
    final updatedAt = lastEdited != null
        ? DateTime.tryParse(lastEdited) ?? DateTime.now()
        : DateTime.now();

    return NotionEvent(
      localId: localId,
      notionPageId: page['id'] as String?,
      title: title.isNotEmpty ? title : '(Untitled)',
      description: description,
      subject: subject,
      topic: topic,
      startTime: startTime,
      endTime: endTime,
      durationMinutes: duration,
      isCompleted: checkboxVal,
      source: 'notion',
      updatedAt: updatedAt,
    );
  }

  Map<String, dynamic> toNotionProperties() => {
        'Name': {
          'title': [
            {'text': {'content': title}}
          ],
        },
        'Date': {
          'date': {
            'start': startTime.toIso8601String(),
            'end': endTime.toIso8601String(),
          },
        },
        'Subject': {
          'rich_text': [
            {'text': {'content': subject}}
          ],
        },
        'Topic': {
          'rich_text': [
            {'text': {'content': topic}}
          ],
        },
        'Description': {
          'rich_text': [
            {'text': {'content': description}}
          ],
        },
        'Duration': {'number': durationMinutes},
        'Completed': {'checkbox': isCompleted},
        'Axon ID': {
          'rich_text': [
            {'text': {'content': localId}}
          ],
        },
      };

  static String _extractText(dynamic prop) {
    if (prop == null) return '';
    final list = prop['title'] ?? prop['rich_text'];
    if (list is List && list.isNotEmpty) {
      return (list.first['plain_text'] ??
              list.first['text']?['content'] ??
              '')
          .toString();
    }
    return '';
  }
}

class NotionSyncResult {
  final int pulled;
  final int pushed;
  final int updated;
  final int deleted;
  final List<String> errors;
  final DateTime syncedAt;

  const NotionSyncResult({
    required this.pulled,
    required this.pushed,
    required this.updated,
    required this.deleted,
    required this.errors,
    required this.syncedAt,
  });
}

// ─── Service ─────────────────────────────────────────────────────────────────

class NotionService {
  static final NotionService _instance = NotionService._internal();
  factory NotionService() => _instance;
  NotionService._internal();

  static const String _baseUrl = 'https://api.notion.com/v1';
  static const String _notionVersion = '2022-06-28';

  String? _integrationToken;
  String? _databaseId;
  bool _isConnected = false;
  SharedPreferences? _prefs;

  final Map<String, NotionEvent> _localEvents = {};
  final _streamController = StreamController<List<NotionEvent>>.broadcast();
  Timer? _pollTimer;
  NotionSyncStatus _syncStatus = NotionSyncStatus.idle;

  bool get isConnected => _isConnected;
  NotionSyncStatus get syncStatus => _syncStatus;
  Stream<List<NotionEvent>> get eventStream => _streamController.stream;
  List<NotionEvent> get events => _localEvents.values
      .where((e) => !e.isDeleted)
      .toList()
    ..sort((a, b) => a.startTime.compareTo(b.startTime));

  // ── Init & Auth ────────────────────────────────────────────────────────────

  Future<bool> initialize() async {
    _prefs ??= await SharedPreferences.getInstance();
    _integrationToken = _prefs!.getString('notion_integration_token');
    _databaseId = _prefs!.getString('notion_database_id');
    _isConnected = _integrationToken != null && _databaseId != null;
    await _loadLocalEvents();
    return _isConnected;
  }

  Future<bool> connect(String integrationToken, String databaseId) async {
    _prefs ??= await SharedPreferences.getInstance();
    _integrationToken = integrationToken;
    _databaseId = databaseId;
    await _prefs!.setString('notion_integration_token', integrationToken);
    await _prefs!.setString('notion_database_id', databaseId);
    _isConnected = await validateConnection();
    return _isConnected;
  }

  Future<void> disconnect() async {
    stopRealtimeSync();
    _integrationToken = null;
    _databaseId = null;
    _isConnected = false;
    _prefs ??= await SharedPreferences.getInstance();
    await _prefs!.remove('notion_integration_token');
    await _prefs!.remove('notion_database_id');
    await _prefs!.remove('notion_local_events');
    await _prefs!.remove('notion_last_sync');
  }

  Future<bool> validateConnection() async {
    try {
      await _queryDatabase(pageSize: 1);
      return true;
    } catch (_) {
      return false;
    }
  }

  // ── Real-time Sync ─────────────────────────────────────────────────────────

  void startRealtimeSync({int intervalSeconds = 60}) {
    _pollTimer?.cancel();
    syncNow();
    _pollTimer =
        Timer.periodic(Duration(seconds: intervalSeconds), (_) => syncNow());
  }

  void stopRealtimeSync() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  Future<NotionSyncResult> syncNow() => _runBidirectionalSync();

  // ── Local Event Mutations ──────────────────────────────────────────────────

  Future<NotionEvent> upsertEvent(NotionEvent event) async {
    _localEvents[event.localId] = event;
    await _saveLocalEvents();
    _emitStream();
    if (!_isConnected) return event;

    try {
      final existingPageId =
          event.notionPageId ?? await _findPageByAxonId(event.localId);
      if (existingPageId != null) {
        await _patchNotionPage(existingPageId, event.toNotionProperties());
        final synced = event.copyWith(
            notionPageId: existingPageId,
            source: 'synced',
            updatedAt: DateTime.now());
        _localEvents[event.localId] = synced;
      } else {
        final created =
            await _createNotionPage(event.toNotionProperties());
        final synced = event.copyWith(
            notionPageId: created['id'] as String?,
            source: 'synced',
            updatedAt: DateTime.now());
        _localEvents[event.localId] = synced;
      }
      await _saveLocalEvents();
      _emitStream();
    } catch (e) {
      debugPrint('[Notion] upsertEvent failed: $e');
    }
    return _localEvents[event.localId]!;
  }

  Future<void> removeEvent(String localId) async {
    final event = _localEvents[localId];
    if (event == null) return;

    _localEvents[localId] =
        event.copyWith(isDeleted: true, updatedAt: DateTime.now());
    await _saveLocalEvents();
    _emitStream();

    if (!_isConnected || event.notionPageId == null) return;
    try {
      await _archiveNotionPage(event.notionPageId!);
      _localEvents.remove(localId);
      await _saveLocalEvents();
    } catch (e) {
      debugPrint('[Notion] removeEvent failed: $e');
    }
  }

  // ── Core Bidirectional Sync ────────────────────────────────────────────────

  Future<NotionSyncResult> _runBidirectionalSync() async {
    if (!_isConnected || _syncStatus == NotionSyncStatus.syncing) {
      return NotionSyncResult(
          pulled: 0,
          pushed: 0,
          updated: 0,
          deleted: 0,
          errors: [],
          syncedAt: DateTime.now());
    }

    _syncStatus = NotionSyncStatus.syncing;
    int pulled = 0, pushed = 0, updated = 0, deleted = 0;
    final errors = <String>[];

    try {
      final lastSync = _prefs!.getInt('notion_last_sync');
      final lastSyncDt = lastSync != null
          ? DateTime.fromMillisecondsSinceEpoch(lastSync)
          : null;

      // ── Step 1: Pull changed pages from Notion → App ───────────────────────
      final notionPages = await _fetchChangedPages(since: lastSyncDt);

      for (final page in notionPages) {
        try {
          final archived = page['archived'] as bool? ?? false;
          final incoming = NotionEvent.fromNotionPage(
              Map<String, dynamic>.from(page));
          final existing = _localEvents[incoming.localId];

          if (archived) {
            if (existing != null) {
              _localEvents[incoming.localId] =
                  existing.copyWith(isDeleted: true);
              deleted++;
            }
            continue;
          }

          if (existing == null) {
            _localEvents[incoming.localId] = incoming;
            pulled++;
          } else if (existing.source == 'app' &&
              existing.updatedAt.isAfter(incoming.updatedAt)) {
            // App version newer — push back
            await _patchNotionPage(
                incoming.notionPageId!, existing.toNotionProperties());
            _localEvents[existing.localId] = existing.copyWith(
                notionPageId: incoming.notionPageId, source: 'synced');
            pushed++;
          } else {
            _localEvents[incoming.localId] =
                incoming.copyWith(source: 'synced');
            updated++;
          }
        } catch (e) {
          errors.add('Pull error: $e');
        }
      }

      // ── Step 2: Push local-only events → Notion ───────────────────────────
      for (final event in _localEvents.values.toList()) {
        if (event.source == 'synced' || event.source == 'notion') continue;

        try {
          if (event.isDeleted && event.notionPageId != null) {
            await _archiveNotionPage(event.notionPageId!);
            _localEvents.remove(event.localId);
            deleted++;
          } else if (!event.isDeleted && event.notionPageId == null) {
            final existingId = await _findPageByAxonId(event.localId);
            if (existingId != null) {
              await _patchNotionPage(existingId, event.toNotionProperties());
              _localEvents[event.localId] = event.copyWith(
                  notionPageId: existingId, source: 'synced');
            } else {
              final created =
                  await _createNotionPage(event.toNotionProperties());
              _localEvents[event.localId] = event.copyWith(
                  notionPageId: created['id'] as String?,
                  source: 'synced');
              pushed++;
            }
          }
        } catch (e) {
          errors.add('Push error for ${event.localId}: $e');
        }
      }

      await _saveLocalEvents();
      await _prefs!.setInt(
          'notion_last_sync', DateTime.now().millisecondsSinceEpoch);
      _emitStream();
      _syncStatus = NotionSyncStatus.idle;

      return NotionSyncResult(
          pulled: pulled,
          pushed: pushed,
          updated: updated,
          deleted: deleted,
          errors: errors,
          syncedAt: DateTime.now());
    } catch (e) {
      _syncStatus = NotionSyncStatus.error;
      debugPrint('[Notion] Sync failed: $e');
      return NotionSyncResult(
          pulled: pulled,
          pushed: pushed,
          updated: updated,
          deleted: deleted,
          errors: [...errors, e.toString()],
          syncedAt: DateTime.now());
    }
  }

  // ── Notion API ─────────────────────────────────────────────────────────────

  Map<String, String> get _headers => {
        'Authorization': 'Bearer $_integrationToken',
        'Notion-Version': _notionVersion,
        'Content-Type': 'application/json',
      };

  /// Incremental fetch — filters by last_edited_time > [since] if provided
  Future<List<Map<String, dynamic>>> _fetchChangedPages(
      {DateTime? since}) async {
    final allPages = <Map<String, dynamic>>[];
    String? cursor;

    do {
      final body = <String, dynamic>{
        'page_size': 100,
        'sorts': [
          {'timestamp': 'last_edited_time', 'direction': 'descending'}
        ],
      };
      if (since != null) {
        body['filter'] = {
          'timestamp': 'last_edited_time',
          'last_edited_time': {'after': since.toIso8601String()},
        };
      }
      if (cursor != null) body['start_cursor'] = cursor;

      final response = await http.post(
        Uri.parse('$_baseUrl/databases/$_databaseId/query'),
        headers: _headers,
        body: jsonEncode(body),
      );
      if (response.statusCode != 200) {
        throw Exception(
            'Notion fetch failed: ${response.statusCode} ${response.body}');
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      allPages.addAll((data['results'] as List)
          .map((p) => Map<String, dynamic>.from(p as Map)));
      cursor = data['next_cursor'] as String?;
      if (!(data['has_more'] as bool? ?? false)) break;
    } while (cursor != null);

    return allPages;
  }

  Future<Map<String, dynamic>> _createNotionPage(
      Map<String, dynamic> properties) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/pages'),
      headers: _headers,
      body: jsonEncode({
        'parent': {'database_id': _databaseId},
        'properties': properties,
      }),
    );
    if (response.statusCode != 200) {
      throw Exception(
          'Notion create failed: ${response.statusCode} ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> _patchNotionPage(
      String pageId, Map<String, dynamic> properties) async {
    final response = await http.patch(
      Uri.parse('$_baseUrl/pages/$pageId'),
      headers: _headers,
      body: jsonEncode({'properties': properties}),
    );
    if (response.statusCode != 200) {
      throw Exception(
          'Notion patch failed: ${response.statusCode} ${response.body}');
    }
  }

  Future<void> _archiveNotionPage(String pageId) async {
    final response = await http.patch(
      Uri.parse('$_baseUrl/pages/$pageId'),
      headers: _headers,
      body: jsonEncode({'archived': true}),
    );
    if (response.statusCode != 200) {
      throw Exception(
          'Notion archive failed: ${response.statusCode} ${response.body}');
    }
  }

  Future<String?> _findPageByAxonId(String axonId) async {
    try {
      final results = await _queryDatabase(
        filter: {
          'property': 'Axon ID',
          'rich_text': {'equals': axonId},
        },
        pageSize: 1,
      );
      if (results.isNotEmpty) return results.first['id'] as String?;
    } catch (_) {}
    return null;
  }

  Future<List<Map<String, dynamic>>> _queryDatabase({
    Map<String, dynamic>? filter,
    int? pageSize,
    String? startCursor,
  }) async {
    final body = <String, dynamic>{};
    if (filter != null) body['filter'] = filter;
    if (pageSize != null) body['page_size'] = pageSize;
    if (startCursor != null) body['start_cursor'] = startCursor;

    final response = await http.post(
      Uri.parse('$_baseUrl/databases/$_databaseId/query'),
      headers: _headers,
      body: jsonEncode(body),
    );
    if (response.statusCode != 200) {
      final error = jsonDecode(response.body);
      throw Exception('Notion API error: ${error['message']}');
    }
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return (data['results'] as List)
        .map((p) => Map<String, dynamic>.from(p as Map))
        .toList();
  }

  // ── Local Persistence ──────────────────────────────────────────────────────

  Future<void> _saveLocalEvents() async {
    _prefs ??= await SharedPreferences.getInstance();
    final json =
        jsonEncode(_localEvents.values.map((e) => e.toJson()).toList());
    await _prefs!.setString('notion_local_events', json);
  }

  Future<void> _loadLocalEvents() async {
    _prefs ??= await SharedPreferences.getInstance();
    final raw = _prefs!.getString('notion_local_events');
    if (raw == null) return;
    try {
      final list = jsonDecode(raw) as List;
      for (final item in list) {
        final event =
            NotionEvent.fromJson(Map<String, dynamic>.from(item as Map));
        _localEvents[event.localId] = event;
      }
    } catch (e) {
      debugPrint('[Notion] Failed to load local events: $e');
    }
  }

  void _emitStream() {
    if (!_streamController.isClosed) _streamController.add(events);
  }

  List<NotionEvent> getEventsForDay(DateTime day) => events
      .where((e) =>
          e.startTime.year == day.year &&
          e.startTime.month == day.month &&
          e.startTime.day == day.day)
      .toList();

  DateTime? get lastSyncedAt {
    final ms = _prefs?.getInt('notion_last_sync');
    return ms != null ? DateTime.fromMillisecondsSinceEpoch(ms) : null;
  }

  Future<List<Map<String, dynamic>>> search(String query) async {
    if (_integrationToken == null) throw Exception('Not connected');
    final response = await http.post(
      Uri.parse('$_baseUrl/search'),
      headers: _headers,
      body: jsonEncode({'query': query, 'page_size': 10}),
    );
    if (response.statusCode != 200) throw Exception('Search failed');
    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return (data['results'] as List)
        .where((p) => p['object'] == 'page')
        .map((p) => Map<String, dynamic>.from(p as Map))
        .toList();
  }

  void dispose() {
    stopRealtimeSync();
    _streamController.close();
  }

  Future<void> syncStudyPlan(List<Map<String, dynamic>> tasks) async {
    debugPrint('NotionService: syncStudyPlan - ${tasks.length} tasks (stub)');
  }
}

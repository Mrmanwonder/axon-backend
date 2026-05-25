// lib/services/google_calendar_service.dart
import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// ─── Models ────────────────────────────────────────────────────────────────

enum SyncDirection { googleToApp, appToGoogle, both }
enum SyncStatus { idle, syncing, error }
enum ConflictResolution { googleWins, appWins, latestWins }

/// Unified calendar event that exists in both the app and Google Calendar
class CalendarEvent {
  final String localId;       // App-side UUID
  final String? googleId;     // Google Calendar event ID
  final String title;
  final String description;
  final DateTime startTime;
  final DateTime endTime;
  final String? location;
  final String source;        // 'app' | 'google' | 'synced'
  final DateTime updatedAt;
  final bool isDeleted;

  const CalendarEvent({
    required this.localId,
    this.googleId,
    required this.title,
    required this.description,
    required this.startTime,
    required this.endTime,
    this.location,
    required this.source,
    required this.updatedAt,
    this.isDeleted = false,
  });

  CalendarEvent copyWith({
    String? googleId,
    String? title,
    String? description,
    DateTime? startTime,
    DateTime? endTime,
    String? location,
    String? source,
    DateTime? updatedAt,
    bool? isDeleted,
  }) {
    return CalendarEvent(
      localId: localId,
      googleId: googleId ?? this.googleId,
      title: title ?? this.title,
      description: description ?? this.description,
      startTime: startTime ?? this.startTime,
      endTime: endTime ?? this.endTime,
      location: location ?? this.location,
      source: source ?? this.source,
      updatedAt: updatedAt ?? this.updatedAt,
      isDeleted: isDeleted ?? this.isDeleted,
    );
  }

  Map<String, dynamic> toJson() => {
        'localId': localId,
        'googleId': googleId,
        'title': title,
        'description': description,
        'startTime': startTime.toIso8601String(),
        'endTime': endTime.toIso8601String(),
        'location': location,
        'source': source,
        'updatedAt': updatedAt.toIso8601String(),
        'isDeleted': isDeleted,
      };

  factory CalendarEvent.fromJson(Map<String, dynamic> map) => CalendarEvent(
        localId: map['localId'] as String,
        googleId: map['googleId'] as String?,
        title: map['title'] as String,
        description: map['description'] as String? ?? '',
        startTime: DateTime.parse(map['startTime'] as String),
        endTime: DateTime.parse(map['endTime'] as String),
        location: map['location'] as String?,
        source: map['source'] as String? ?? 'app',
        updatedAt: DateTime.parse(map['updatedAt'] as String),
        isDeleted: map['isDeleted'] as bool? ?? false,
      );

  factory CalendarEvent.fromGoogleEvent(Map<String, dynamic> g) {
    final start = _parseGoogleDateTime(g['start'] as Map<String, dynamic>?);
    final end = _parseGoogleDateTime(g['end'] as Map<String, dynamic>?);
    final updated = g['updated'] != null
        ? DateTime.tryParse(g['updated'] as String) ?? DateTime.now()
        : DateTime.now();
    final desc = g['description'] as String? ?? '';
    final localId = _extractLocalId(desc) ?? 'gcal_${g['id']}';

    return CalendarEvent(
      localId: localId,
      googleId: g['id'] as String?,
      title: g['summary'] as String? ?? '(No title)',
      description: desc.replaceAll(RegExp(r'\[AxonLocalId:[^\]]+\]\s*'), '').trim(),
      startTime: start,
      endTime: end,
      location: g['location'] as String?,
      source: 'google',
      updatedAt: updated,
    );
  }

  static DateTime _parseGoogleDateTime(Map<String, dynamic>? block) {
    if (block == null) return DateTime.now();
    // All-day events use 'date', timed events use 'dateTime'
    final raw = block['dateTime'] as String? ?? block['date'] as String? ?? '';
    return DateTime.tryParse(raw) ?? DateTime.now();
  }

  static String? _extractLocalId(String description) {
    final match = RegExp(r'\[AxonLocalId:([^\]]+)\]').firstMatch(description);
    return match?.group(1);
  }

  /// Build the description we embed in Google Calendar to allow round-trip identification
  String get googleDescription {
    final base = description.isNotEmpty ? description : '';
    return '$base\n\n[AxonLocalId:$localId]'.trim();
  }
}

class SyncResult {
  final int pulled;       // Google → App
  final int pushed;       // App → Google
  final int conflicts;
  final int deleted;
  final List<String> errors;
  final DateTime syncedAt;

  const SyncResult({
    required this.pulled,
    required this.pushed,
    required this.conflicts,
    required this.deleted,
    required this.errors,
    required this.syncedAt,
  });
}

// ─── Service ───────────────────────────────────────────────────────────────

class GoogleCalendarService {
  static final GoogleCalendarService _instance =
      GoogleCalendarService._internal();
  factory GoogleCalendarService() => _instance;
  GoogleCalendarService._internal();

  // ── Auth ──────────────────────────────────────────────────────────────────

  final GoogleSignIn _googleSignIn = GoogleSignIn(
    scopes: [
      'https://www.googleapis.com/auth/calendar',
      'https://www.googleapis.com/auth/calendar.events',
    ],
  );

  GoogleSignInAccount? _currentUser;
  Map<String, String>? _authHeaders;
  bool _isConnected = false;
  String? _axonCalendarId;
  SharedPreferences? _prefs;

  // ── Sync state ────────────────────────────────────────────────────────────

  /// In-memory event store (localId → CalendarEvent)
  final Map<String, CalendarEvent> _localEvents = {};

  /// Google's incremental sync token — avoids re-fetching all events
  String? _syncToken;

  /// Real-time stream controller
  final _syncStreamController =
      StreamController<List<CalendarEvent>>.broadcast();

  /// Polling timer for real-time pull
  Timer? _pollTimer;
  SyncStatus _syncStatus = SyncStatus.idle;

  // ── Public API ────────────────────────────────────────────────────────────

  bool get isConnected => _isConnected;
  String? get currentUserEmail => _currentUser?.email;
  SyncStatus get syncStatus => _syncStatus;
  List<CalendarEvent> get events => _localEvents.values
      .where((e) => !e.isDeleted)
      .toList()
    ..sort((a, b) => a.startTime.compareTo(b.startTime));

  /// Stream of all current events — emits whenever sync runs
  Stream<List<CalendarEvent>> get eventStream => _syncStreamController.stream;

  // ── Initialization & Auth ─────────────────────────────────────────────────

  Future<bool> initialize() async {
    _prefs ??= await SharedPreferences.getInstance();
    _axonCalendarId = _prefs!.getString('google_axon_calendar_id');
    _syncToken = _prefs!.getString('google_sync_token');
    await _loadLocalEvents();

    final accountName = _prefs!.getString('google_account_name');
    if (accountName == null) return false;

    try {
      await _googleSignIn.signInSilently();
      _currentUser = _googleSignIn.currentUser;
      if (_currentUser != null) {
        _authHeaders = await _currentUser!.authHeaders;
        _isConnected = true;
        return true;
      }
    } catch (e) {
      debugPrint('[GCal] Silent sign-in failed: $e');
      await signOut();
    }
    return false;
  }

  Future<void> signIn() async {
    try {
      _currentUser = await _googleSignIn.signIn();
      if (_currentUser == null) throw Exception('Sign-in cancelled');
      _authHeaders = await _currentUser!.authHeaders;
      _isConnected = true;
      _prefs ??= await SharedPreferences.getInstance();
      await _prefs!.setString('google_account_name', _currentUser!.email);
      _axonCalendarId = null;
      _syncToken = null; // Force full sync on new sign-in
    } catch (e) {
      _isConnected = false;
      rethrow;
    }
  }

  Future<void> signOut() async {
    stopRealtimeSync();
    await _googleSignIn.signOut();
    _currentUser = null;
    _authHeaders = null;
    _isConnected = false;
    _axonCalendarId = null;
    _syncToken = null;
    _prefs ??= await SharedPreferences.getInstance();
    await _prefs!.remove('google_account_name');
    await _prefs!.remove('google_axon_calendar_id');
    await _prefs!.remove('google_sync_token');
    await _prefs!.remove('local_events');
  }

  // ── Real-time Sync ────────────────────────────────────────────────────────

  /// Starts bidirectional real-time sync.
  /// - Runs a full sync immediately
  /// - Then polls Google Calendar every [intervalSeconds] for changes
  /// - Any local event mutations also push immediately via [upsertEvent] / [deleteEvent]
  void startRealtimeSync({
    int intervalSeconds = 60,
    ConflictResolution conflictResolution = ConflictResolution.latestWins,
  }) {
    _pollTimer?.cancel();
    // Run immediately
    _runBidirectionalSync(conflictResolution: conflictResolution);
    // Then poll
    _pollTimer = Timer.periodic(Duration(seconds: intervalSeconds), (_) {
      _runBidirectionalSync(conflictResolution: conflictResolution);
    });
  }

  void stopRealtimeSync() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  /// Manually trigger a full bidirectional sync
  Future<SyncResult> syncNow({
    ConflictResolution conflictResolution = ConflictResolution.latestWins,
  }) async {
    return _runBidirectionalSync(conflictResolution: conflictResolution);
  }

  // ── Local Event Mutations (push immediately to Google) ────────────────────

  /// Add or update an event from the app side — pushes to Google immediately
  Future<CalendarEvent> upsertEvent(CalendarEvent event) async {
    // Update local store
    _localEvents[event.localId] = event;
    await _saveLocalEvents();
    _emitStream();

    if (!_isConnected) return event;

    try {
      final calendarId = await _getOrCreateAxonCalendar();
      if (event.googleId != null) {
        // PATCH — partial update, preserves fields we don't send
        final updated = await _patchGoogleEvent(calendarId, event);
        final synced = event.copyWith(googleId: updated['id'] as String?, source: 'synced');
        _localEvents[event.localId] = synced;
        await _saveLocalEvents();
        _emitStream();
        return synced;
      } else {
        // Create new
        final created = await _createGoogleEvent(calendarId, event);
        final synced = event.copyWith(googleId: created['id'] as String?, source: 'synced');
        _localEvents[event.localId] = synced;
        await _saveLocalEvents();
        _emitStream();
        return synced;
      }
    } catch (e) {
      debugPrint('[GCal] upsertEvent failed: $e');
      return event; // Kept locally, will retry on next sync
    }
  }

  /// Delete an event from the app side — removes from Google immediately
  Future<void> removeEvent(String localId) async {
    final event = _localEvents[localId];
    if (event == null) return;

    // Soft-delete locally
    _localEvents[localId] = event.copyWith(isDeleted: true, updatedAt: DateTime.now());
    await _saveLocalEvents();
    _emitStream();

    if (!_isConnected || event.googleId == null) return;

    try {
      final calendarId = await _getOrCreateAxonCalendar();
      await _deleteGoogleEvent(calendarId, event.googleId!);
      _localEvents.remove(localId);
      await _saveLocalEvents();
    } catch (e) {
      debugPrint('[GCal] removeEvent failed: $e');
    }
  }

  // ── Core Bidirectional Sync ───────────────────────────────────────────────

  Future<SyncResult> _runBidirectionalSync({
    required ConflictResolution conflictResolution,
  }) async {
    if (!_isConnected || _syncStatus == SyncStatus.syncing) {
      return SyncResult(
          pulled: 0, pushed: 0, conflicts: 0, deleted: 0,
          errors: [], syncedAt: DateTime.now());
    }

    _syncStatus = SyncStatus.syncing;
    int pulled = 0, pushed = 0, conflicts = 0, deleted = 0;
    final errors = <String>[];

    try {
      await _refreshAuthHeaders();
      final calendarId = await _getOrCreateAxonCalendar();

      // ── Step 1: Pull changes from Google → App ────────────────────────────
      final googleEvents = await _fetchGoogleChanges(calendarId);

      for (final ge in googleEvents) {
        if (ge['status'] == 'cancelled') {
          // Google deleted this event
          final localId = CalendarEvent._extractLocalId(
              ge['description'] as String? ?? '') ??
              'gcal_${ge['id']}';
          if (_localEvents.containsKey(localId)) {
            _localEvents[localId] =
                _localEvents[localId]!.copyWith(isDeleted: true);
            deleted++;
          }
          continue;
        }

        final incoming = CalendarEvent.fromGoogleEvent(
            Map<String, dynamic>.from(ge));
        final existing = _localEvents[incoming.localId];

        if (existing == null) {
          // New event from Google — add to app
          _localEvents[incoming.localId] = incoming;
          pulled++;
        } else if (existing.googleId == null) {
          // Local event found its Google counterpart — link them
          _localEvents[incoming.localId] =
              existing.copyWith(googleId: incoming.googleId, source: 'synced');
        } else {
          // Conflict: both sides may have changed
          final resolved = _resolveConflict(existing, incoming, conflictResolution);
          if (resolved != existing) {
            _localEvents[incoming.localId] = resolved;
            conflicts++;
            if (conflictResolution == ConflictResolution.appWins) {
              // Push app version back to Google
              await _patchGoogleEvent(calendarId, existing);
            }
          } else {
            pulled++;
          }
        }
      }

      // ── Step 2: Push local-only / dirty events → Google ──────────────────
      for (final event in _localEvents.values.toList()) {
        if (event.source == 'google') continue; // Already from Google
        if (event.googleId != null && event.source == 'synced') continue; // Already in sync

        try {
          if (event.isDeleted && event.googleId != null) {
            await _deleteGoogleEvent(calendarId, event.googleId!);
            _localEvents.remove(event.localId);
            deleted++;
          } else if (!event.isDeleted && event.googleId == null) {
            final created = await _createGoogleEvent(calendarId, event);
            _localEvents[event.localId] = event.copyWith(
                googleId: created['id'] as String?, source: 'synced');
            pushed++;
          }
        } catch (e) {
          errors.add('Push failed for ${event.localId}: $e');
        }
      }

      // Persist updated sync token and local store
      await _saveLocalEvents();
      if (_syncToken != null) {
        await _prefs!.setString('google_sync_token', _syncToken!);
      }
      await _prefs!.setInt(
          'google_calendar_last_sync', DateTime.now().millisecondsSinceEpoch);

      _emitStream();
      _syncStatus = SyncStatus.idle;

      return SyncResult(
        pulled: pulled,
        pushed: pushed,
        conflicts: conflicts,
        deleted: deleted,
        errors: errors,
        syncedAt: DateTime.now(),
      );
    } catch (e) {
      _syncStatus = SyncStatus.error;
      debugPrint('[GCal] Sync failed: $e');
      return SyncResult(
          pulled: pulled, pushed: pushed, conflicts: conflicts,
          deleted: deleted, errors: [...errors, e.toString()],
          syncedAt: DateTime.now());
    }
  }

  // ── Incremental Fetch (Sync Token) ────────────────────────────────────────

  /// Uses Google's syncToken for incremental fetches — only gets *changed* events
  /// Falls back to full range fetch on the first call or if the token is invalid
  Future<List<Map<String, dynamic>>> _fetchGoogleChanges(
      String calendarId) async {
    final headers = {..._authHeaders!, 'Accept': 'application/json'};
    final baseUri = Uri.parse(
        'https://www.googleapis.com/calendar/v3/calendars/'
        '${Uri.encodeComponent(calendarId)}/events');

    Uri uri;
    if (_syncToken != null) {
      // Incremental sync — only get events changed since last sync
      uri = baseUri.replace(queryParameters: {
        'syncToken': _syncToken!,
        'singleEvents': 'true',
      });
    } else {
      // Full sync — get all events in a 2-year window
      final now = DateTime.now();
      uri = baseUri.replace(queryParameters: {
        'timeMin': now.subtract(const Duration(days: 30)).toUtc().toIso8601String(),
        'timeMax': now.add(const Duration(days: 365)).toUtc().toIso8601String(),
        'singleEvents': 'true',
        'orderBy': 'updated',
        'maxResults': '500',
      });
    }

    final allEvents = <Map<String, dynamic>>[];
    String? pageToken;

    do {
      final pagedUri = pageToken != null
          ? uri.replace(
              queryParameters: {...uri.queryParameters, 'pageToken': pageToken})
          : uri;

      final response = await http.get(pagedUri, headers: headers);

      // 410 Gone = sync token expired, do full sync
      if (response.statusCode == 410) {
        _syncToken = null;
        await _prefs!.remove('google_sync_token');
        return _fetchGoogleChanges(calendarId);
      }

      if (response.statusCode != 200) {
        throw Exception(
            'Failed to fetch Google events: ${response.statusCode}');
      }

      final data = jsonDecode(response.body) as Map<String, dynamic>;
      allEvents.addAll(
          (data['items'] as List? ?? []).map((e) => Map<String, dynamic>.from(e as Map)));

      // Save new sync token for next incremental call
      if (data['nextSyncToken'] != null) {
        _syncToken = data['nextSyncToken'] as String;
      }

      pageToken = data['nextPageToken'] as String?;
    } while (pageToken != null);

    return allEvents;
  }

  // ── Conflict Resolution ───────────────────────────────────────────────────

  CalendarEvent _resolveConflict(
    CalendarEvent local,
    CalendarEvent remote,
    ConflictResolution resolution,
  ) {
    switch (resolution) {
      case ConflictResolution.appWins:
        return local.copyWith(source: 'synced');
      case ConflictResolution.googleWins:
        return remote.copyWith(source: 'synced');
      case ConflictResolution.latestWins:
        return local.updatedAt.isAfter(remote.updatedAt)
            ? local.copyWith(source: 'synced')
            : remote.copyWith(source: 'synced');
    }
  }

  // ── Google API Helpers ────────────────────────────────────────────────────

  Future<Map<String, dynamic>> _createGoogleEvent(
      String calendarId, CalendarEvent event) async {
    final body = {
      'summary': event.title,
      'description': event.googleDescription,
      'start': {'dateTime': event.startTime.toUtc().toIso8601String(), 'timeZone': 'UTC'},
      'end': {'dateTime': event.endTime.toUtc().toIso8601String(), 'timeZone': 'UTC'},
      if (event.location != null) 'location': event.location,
    };

    final response = await http.post(
      Uri.parse('https://www.googleapis.com/calendar/v3/calendars/'
          '${Uri.encodeComponent(calendarId)}/events'),
      headers: {..._authHeaders!, 'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );

    if (response.statusCode != 200 && response.statusCode != 201) {
      throw Exception('Create failed: ${response.statusCode} ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  /// PATCH instead of PUT — only sends changed fields, preserves the rest
  Future<Map<String, dynamic>> _patchGoogleEvent(
      String calendarId, CalendarEvent event) async {
    if (event.googleId == null) return _createGoogleEvent(calendarId, event);

    final body = {
      'summary': event.title,
      'description': event.googleDescription,
      'start': {'dateTime': event.startTime.toUtc().toIso8601String(), 'timeZone': 'UTC'},
      'end': {'dateTime': event.endTime.toUtc().toIso8601String(), 'timeZone': 'UTC'},
      if (event.location != null) 'location': event.location,
    };

    final response = await http.patch(
      Uri.parse('https://www.googleapis.com/calendar/v3/calendars/'
          '${Uri.encodeComponent(calendarId)}/events/'
          '${Uri.encodeComponent(event.googleId!)}'),
      headers: {..._authHeaders!, 'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );

    if (response.statusCode != 200) {
      throw Exception('Patch failed: ${response.statusCode} ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<void> _deleteGoogleEvent(String calendarId, String eventId) async {
    final response = await http.delete(
      Uri.parse('https://www.googleapis.com/calendar/v3/calendars/'
          '${Uri.encodeComponent(calendarId)}/events/'
          '${Uri.encodeComponent(eventId)}'),
      headers: _authHeaders!,
    );
    if (response.statusCode != 204 && response.statusCode != 200) {
      throw Exception('Delete failed: ${response.statusCode}');
    }
  }

  // ── Auth Token Refresh ────────────────────────────────────────────────────

  Future<void> _refreshAuthHeaders() async {
    try {
      // Force token refresh if needed
      final account = await _googleSignIn.signInSilently();
      if (account != null) {
        _currentUser = account;
        _authHeaders = await account.authHeaders;
      }
    } catch (e) {
      debugPrint('[GCal] Token refresh failed: $e');
    }
  }

  // ── Local Persistence ─────────────────────────────────────────────────────

  Future<void> _saveLocalEvents() async {
    _prefs ??= await SharedPreferences.getInstance();
    final json = jsonEncode(
        _localEvents.values.map((e) => e.toJson()).toList());
    await _prefs!.setString('local_events', json);
  }

  Future<void> _loadLocalEvents() async {
    _prefs ??= await SharedPreferences.getInstance();
    final raw = _prefs!.getString('local_events');
    if (raw == null) return;
    try {
      final list = jsonDecode(raw) as List;
      for (final item in list) {
        final event = CalendarEvent.fromJson(Map<String, dynamic>.from(item as Map));
        _localEvents[event.localId] = event;
      }
    } catch (e) {
      debugPrint('[GCal] Failed to load local events: $e');
    }
  }

  void _emitStream() {
    if (!_syncStreamController.isClosed) {
      _syncStreamController.add(events);
    }
  }

  // ── Calendar Management ───────────────────────────────────────────────────

  Future<String> _getOrCreateAxonCalendar() async {
    if (_axonCalendarId != null) return _axonCalendarId!;

    final response = await http.get(
      Uri.parse(
          'https://www.googleapis.com/calendar/v3/users/me/calendarList'),
      headers: {..._authHeaders!, 'Accept': 'application/json'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      for (final cal in (data['items'] as List? ?? [])) {
        final summary = cal['summary'] as String? ?? '';
        if (summary == 'Axon Study Plan') {
          _axonCalendarId = cal['id'] as String;
          await _prefs!.setString('google_axon_calendar_id', _axonCalendarId!);
          return _axonCalendarId!;
        }
      }
    }

    // Create new Axon calendar
    final res = await http.post(
      Uri.parse('https://www.googleapis.com/calendar/v3/calendars'),
      headers: {..._authHeaders!, 'Content-Type': 'application/json'},
      body: jsonEncode({
        'summary': 'Axon Study Plan',
        'description': 'Bidirectionally synced with Axon app',
        'timeZone': 'UTC',
      }),
    );

    if (res.statusCode != 200) throw Exception('Could not create calendar');
    final data = jsonDecode(res.body) as Map<String, dynamic>;
    _axonCalendarId = data['id'] as String;
    await _prefs!.setString('google_axon_calendar_id', _axonCalendarId!);
    return _axonCalendarId!;
  }

  // ── Convenience Getters ───────────────────────────────────────────────────

  List<CalendarEvent> getEventsForDay(DateTime day) {
    return events.where((e) =>
        e.startTime.year == day.year &&
        e.startTime.month == day.month &&
        e.startTime.day == day.day).toList();
  }

  DateTime? get lastSyncedAt {
    final ms = _prefs?.getInt('google_calendar_last_sync');
    return ms != null ? DateTime.fromMillisecondsSinceEpoch(ms) : null;
  }

  void dispose() {
    stopRealtimeSync();
    _syncStreamController.close();
  }

  Future<SyncResult> syncStudyPlan(List<Map<String, dynamic>> plans) async {
    debugPrint('GoogleCalendar: syncStudyPlan not implemented');
    return SyncResult(
      pulled: 0,
      pushed: 0,
      conflicts: 0,
      deleted: 0,
      errors: [],
      syncedAt: DateTime.now(),
    );
  }
}
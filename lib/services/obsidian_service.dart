// lib/services/obsidian_service.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ─── Models ─────────────────────────────────────────────────────────────────

enum ObsidianSyncStatus { idle, syncing, error }

class ObsidianNote {
  final String localId;       // App UUID — stored in frontmatter as axon_id
  final String? filePath;     // Absolute path to the .md file in vault
  final String title;
  final String description;
  final String subject;
  final String topic;
  final DateTime startTime;
  final DateTime endTime;
  final int durationMinutes;
  final List<String> notes;   // Bullet points under ## Notes
  final List<String> tasks;   // Checkboxes under ## Tasks (unchecked)
  final List<String> completedTasks;
  final String source;        // 'app' | 'obsidian' | 'synced'
  final DateTime updatedAt;
  final bool isDeleted;

  const ObsidianNote({
    required this.localId,
    this.filePath,
    required this.title,
    required this.description,
    required this.subject,
    required this.topic,
    required this.startTime,
    required this.endTime,
    required this.durationMinutes,
    this.notes = const [],
    this.tasks = const [],
    this.completedTasks = const [],
    required this.source,
    required this.updatedAt,
    this.isDeleted = false,
  });

  ObsidianNote copyWith({
    String? filePath,
    String? title,
    String? description,
    String? subject,
    String? topic,
    DateTime? startTime,
    DateTime? endTime,
    int? durationMinutes,
    List<String>? notes,
    List<String>? tasks,
    List<String>? completedTasks,
    String? source,
    DateTime? updatedAt,
    bool? isDeleted,
  }) {
    return ObsidianNote(
      localId: localId,
      filePath: filePath ?? this.filePath,
      title: title ?? this.title,
      description: description ?? this.description,
      subject: subject ?? this.subject,
      topic: topic ?? this.topic,
      startTime: startTime ?? this.startTime,
      endTime: endTime ?? this.endTime,
      durationMinutes: durationMinutes ?? this.durationMinutes,
      notes: notes ?? this.notes,
      tasks: tasks ?? this.tasks,
      completedTasks: completedTasks ?? this.completedTasks,
      source: source ?? this.source,
      updatedAt: updatedAt ?? this.updatedAt,
      isDeleted: isDeleted ?? this.isDeleted,
    );
  }

  Map<String, dynamic> toJson() => {
        'localId': localId,
        'filePath': filePath,
        'title': title,
        'description': description,
        'subject': subject,
        'topic': topic,
        'startTime': startTime.toIso8601String(),
        'endTime': endTime.toIso8601String(),
        'durationMinutes': durationMinutes,
        'notes': notes,
        'tasks': tasks,
        'completedTasks': completedTasks,
        'source': source,
        'updatedAt': updatedAt.toIso8601String(),
        'isDeleted': isDeleted,
      };

  factory ObsidianNote.fromJson(Map<String, dynamic> map) => ObsidianNote(
        localId: map['localId'] as String,
        filePath: map['filePath'] as String?,
        title: map['title'] as String? ?? '',
        description: map['description'] as String? ?? '',
        subject: map['subject'] as String? ?? '',
        topic: map['topic'] as String? ?? '',
        startTime: DateTime.parse(map['startTime'] as String),
        endTime: DateTime.parse(map['endTime'] as String),
        durationMinutes: map['durationMinutes'] as int? ?? 60,
        notes: List<String>.from(map['notes'] as List? ?? []),
        tasks: List<String>.from(map['tasks'] as List? ?? []),
        completedTasks:
            List<String>.from(map['completedTasks'] as List? ?? []),
        source: map['source'] as String? ?? 'app',
        updatedAt: DateTime.parse(map['updatedAt'] as String),
        isDeleted: map['isDeleted'] as bool? ?? false,
      );

  /// Render this note as a complete Obsidian Markdown file
  String toMarkdown() {
    final dateStr = '${startTime.year}-'
        '${startTime.month.toString().padLeft(2, '0')}-'
        '${startTime.day.toString().padLeft(2, '0')}';
    final startStr =
        '${startTime.hour.toString().padLeft(2, '0')}:${startTime.minute.toString().padLeft(2, '0')}';
    final endStr =
        '${endTime.hour.toString().padLeft(2, '0')}:${endTime.minute.toString().padLeft(2, '0')}';

    final sb = StringBuffer()
      ..writeln('---')
      ..writeln('axon_id: $localId')
      ..writeln('subject: $subject')
      ..writeln('topic: $topic')
      ..writeln('created: ${startTime.toIso8601String()}')
      ..writeln('updated: ${updatedAt.toIso8601String()}')
      ..writeln('tags: [axon/study-session]')
      ..writeln('---')
      ..writeln()
      ..writeln('# $title')
      ..writeln()
      ..writeln('**Date:** $dateStr')
      ..writeln('**Time:** $startStr - $endStr')
      ..writeln('**Duration:** $durationMinutes minutes')
      ..writeln()
      ..writeln('## Description')
      ..writeln(description.isNotEmpty ? description : '_No description_')
      ..writeln()
      ..writeln('## Notes');

    if (notes.isEmpty) {
      sb.writeln('- ');
    } else {
      for (final note in notes) {
        sb.writeln('- $note');
      }
    }

    sb.writeln();
    sb.writeln('## Tasks');

    if (tasks.isEmpty && completedTasks.isEmpty) {
      sb.writeln('- [ ] ');
    } else {
      for (final task in tasks) {
        sb.writeln('- [ ] $task');
      }
      for (final task in completedTasks) {
        sb.writeln('- [x] $task');
      }
    }

    return sb.toString();
  }

  /// Parse an Obsidian Markdown file into an ObsidianNote
  static ObsidianNote fromMarkdown(String content, String filePath) {
    final frontmatter = _parseFrontmatter(content);

    final localId =
        frontmatter['axon_id']?.toString() ?? 'obsidian_${filePath.hashCode}';
    final title = _extractH1(content);
    final subject = frontmatter['subject']?.toString() ?? '';
    final topic = frontmatter['topic']?.toString() ?? '';
    final description = _extractSection(content, 'Description');
    final notes = _extractBullets(content, 'Notes');
    final tasks = _extractCheckboxes(content, checked: false);
    final completedTasks = _extractCheckboxes(content, checked: true);

    final createdRaw = frontmatter['created']?.toString();
    final updatedRaw = frontmatter['updated']?.toString();
    final startTime =
        createdRaw != null ? DateTime.tryParse(createdRaw) ?? DateTime.now() : DateTime.now();
    final updatedAt =
        updatedRaw != null ? DateTime.tryParse(updatedRaw) ?? DateTime.now() : DateTime.now();
    final endTime = startTime.add(const Duration(hours: 1));

    // Parse duration from "**Duration:** X minutes"
    final durationMatch =
        RegExp(r'\*\*Duration:\*\*\s*(\d+)').firstMatch(content);
    final duration =
        durationMatch != null ? int.tryParse(durationMatch.group(1)!) ?? 60 : 60;

    return ObsidianNote(
      localId: localId,
      filePath: filePath,
      title: title.isNotEmpty ? title : 'Study Session',
      description: description,
      subject: subject,
      topic: topic,
      startTime: startTime,
      endTime: endTime,
      durationMinutes: duration,
      notes: notes,
      tasks: tasks,
      completedTasks: completedTasks,
      source: 'obsidian',
      updatedAt: updatedAt,
    );
  }

  static Map<String, String> _parseFrontmatter(String content) {
    final result = <String, String>{};
    if (!content.startsWith('---')) return result;
    final parts = content.split('---');
    if (parts.length < 2) return result;
    for (final line in parts[1].split('\n')) {
      final idx = line.indexOf(':');
      if (idx < 0) continue;
      final key = line.substring(0, idx).trim();
      final value = line.substring(idx + 1).trim();
      result[key] = value;
    }
    return result;
  }

  static String _extractH1(String content) {
    for (final line in content.split('\n')) {
      if (line.startsWith('# ') && !line.startsWith('## ')) {
        return line.substring(2).trim();
      }
    }
    return '';
  }

  static String _extractSection(String content, String heading) {
    final lines = content.split('\n');
    final sb = StringBuffer();
    bool inSection = false;
    for (final line in lines) {
      if (line.toLowerCase() == '## ${heading.toLowerCase()}') {
        inSection = true;
        continue;
      }
      if (inSection) {
        if (line.startsWith('## ')) break;
        if (line.trim().isNotEmpty &&
            !line.startsWith('_') &&
            !line.startsWith('*')) {
          sb.writeln(line.trim());
        }
      }
    }
    return sb.toString().trim();
  }

  static List<String> _extractBullets(String content, String heading) {
    final lines = content.split('\n');
    final result = <String>[];
    bool inSection = false;
    for (final line in lines) {
      if (line.toLowerCase() == '## ${heading.toLowerCase()}') {
        inSection = true;
        continue;
      }
      if (inSection) {
        if (line.startsWith('## ')) break;
        final trimmed = line.trim();
        if (trimmed.startsWith('- ') &&
            !trimmed.startsWith('- [ ]') &&
            !trimmed.startsWith('- [x]')) {
          final text = trimmed.substring(2).trim();
          if (text.isNotEmpty) result.add(text);
        }
      }
    }
    return result;
  }

  static List<String> _extractCheckboxes(String content,
      {required bool checked}) {
    final pattern = checked ? RegExp(r'^- \[x\] (.+)') : RegExp(r'^- \[ \] (.+)');
    final result = <String>[];
    for (final line in content.split('\n')) {
      final match = pattern.firstMatch(line.trim());
      if (match != null) result.add(match.group(1)!.trim());
    }
    return result;
  }
}

class ObsidianSyncResult {
  final int pulled;
  final int pushed;
  final int updated;
  final int deleted;
  final List<String> errors;
  final DateTime syncedAt;

  const ObsidianSyncResult({
    required this.pulled,
    required this.pushed,
    required this.updated,
    required this.deleted,
    required this.errors,
    required this.syncedAt,
  });
}

// ─── Service ─────────────────────────────────────────────────────────────────

class ObsidianService {
  static final ObsidianService _instance = ObsidianService._internal();
  factory ObsidianService() => _instance;
  ObsidianService._internal();

  String? _vaultPath;
  bool _isConnected = false;
  SharedPreferences? _prefs;

  final Map<String, ObsidianNote> _localNotes = {};
  final _streamController = StreamController<List<ObsidianNote>>.broadcast();

  // File system watcher for real-time detection of external edits
  StreamSubscription<FileSystemEvent>? _watcherSub;
  Timer? _debounceTimer;
  ObsidianSyncStatus _syncStatus = ObsidianSyncStatus.idle;

  bool get isConnected => _isConnected;
  String? get vaultPath => _vaultPath;
  ObsidianSyncStatus get syncStatus => _syncStatus;
  Stream<List<ObsidianNote>> get noteStream => _streamController.stream;

  String get _axonDirPath => '$_vaultPath/Axon/Study Plans';

  List<ObsidianNote> get notes => _localNotes.values
      .where((n) => !n.isDeleted)
      .toList()
    ..sort((a, b) => a.startTime.compareTo(b.startTime));

  // ── Init & Connect ─────────────────────────────────────────────────────────

  Future<bool> initialize() async {
    _prefs ??= await SharedPreferences.getInstance();
    _vaultPath = _prefs!.getString('obsidian_vault_path');
    if (_vaultPath != null) {
      _isConnected = await Directory(_vaultPath!).exists();
    }
    await _loadLocalNotes();
    return _isConnected;
  }

  Future<bool> connect(String vaultPath) async {
    final dir = Directory(vaultPath);
    if (!await dir.exists()) throw Exception('Vault directory does not exist');
    _vaultPath = vaultPath;
    _isConnected = true;
    _prefs ??= await SharedPreferences.getInstance();
    await _prefs!.setString('obsidian_vault_path', vaultPath);
    // Ensure Axon folder exists
    await Directory(_axonDirPath).create(recursive: true);
    return true;
  }

  Future<void> disconnect() async {
    stopRealtimeSync();
    _vaultPath = null;
    _isConnected = false;
    _prefs ??= await SharedPreferences.getInstance();
    await _prefs!.remove('obsidian_vault_path');
    await _prefs!.remove('obsidian_local_notes');
    await _prefs!.remove('obsidian_last_sync');
  }

  // ── Real-time Sync ─────────────────────────────────────────────────────────

  /// Starts bidirectional real-time sync.
  /// - Uses [FileSystemWatcher] to detect vault edits immediately (no polling lag)
  /// - Debounces rapid file saves (e.g., editor autosave) to avoid redundant reads
  /// - Any app-side mutations write files immediately via [upsertNote]
  Future<void> startRealtimeSync() async {
    if (!_isConnected || _vaultPath == null) return;

    // Initial full sync first
    await syncNow();

    // Watch the Axon folder for any file changes
    final axonDir = Directory(_axonDirPath);
    if (!await axonDir.exists()) await axonDir.create(recursive: true);

    _watcherSub?.cancel();
    _watcherSub = axonDir
        .watch(events: FileSystemEvent.all, recursive: false)
        .listen(_onFileSystemEvent);
  }

  void stopRealtimeSync() {
    _watcherSub?.cancel();
    _watcherSub = null;
    _debounceTimer?.cancel();
    _debounceTimer = null;
  }

  void _onFileSystemEvent(FileSystemEvent event) {
    // Debounce: wait 500ms after the last event before syncing
    // This prevents 10 reads when an editor saves a file in bursts
    _debounceTimer?.cancel();
    _debounceTimer = Timer(const Duration(milliseconds: 500), () {
      if (event is FileSystemDeleteEvent) {
        _handleFileDeletion(event.path);
      } else {
        _handleFileChange(event.path);
      }
    });
  }

  Future<ObsidianSyncResult> syncNow() => _runBidirectionalSync();

  // ── Local Note Mutations (write files immediately) ─────────────────────────

  /// Add or update a note from the app side — writes to vault immediately
  Future<ObsidianNote> upsertNote(ObsidianNote note) async {
    _localNotes[note.localId] = note;
    await _saveLocalNotes();
    _emitStream();

    if (!_isConnected) return note;

    try {
      final path = note.filePath ?? _buildFilePath(note);
      await Directory(_axonDirPath).create(recursive: true);
      await File(path).writeAsString(note.toMarkdown());
      final updated = note.copyWith(
          filePath: path, source: 'synced', updatedAt: DateTime.now());
      _localNotes[note.localId] = updated;
      await _saveLocalNotes();
      await _rebuildIndex();
      _emitStream();
      return updated;
    } catch (e) {
      debugPrint('[Obsidian] upsertNote failed: $e');
      return _localNotes[note.localId]!;
    }
  }

  /// Remove a note — deletes the .md file from vault
  Future<void> removeNote(String localId) async {
    final note = _localNotes[localId];
    if (note == null) return;

    _localNotes[localId] =
        note.copyWith(isDeleted: true, updatedAt: DateTime.now());
    await _saveLocalNotes();
    _emitStream();

    if (!_isConnected || note.filePath == null) return;

    try {
      final file = File(note.filePath!);
      if (await file.exists()) await file.delete();
      _localNotes.remove(localId);
      await _saveLocalNotes();
      await _rebuildIndex();
    } catch (e) {
      debugPrint('[Obsidian] removeNote failed: $e');
    }
  }

  // ── Core Bidirectional Sync ────────────────────────────────────────────────

  Future<ObsidianSyncResult> _runBidirectionalSync() async {
    if (!_isConnected || _syncStatus == ObsidianSyncStatus.syncing) {
      return ObsidianSyncResult(
          pulled: 0, pushed: 0, updated: 0, deleted: 0,
          errors: [], syncedAt: DateTime.now());
    }

    _syncStatus = ObsidianSyncStatus.syncing;
    int pulled = 0, pushed = 0, updated = 0, deleted = 0;
    final errors = <String>[];

    try {
      final axonDir = Directory(_axonDirPath);
      if (!await axonDir.exists()) await axonDir.create(recursive: true);

      final lastSync = _prefs!.getInt('obsidian_last_sync');
      final lastSyncDt = lastSync != null
          ? DateTime.fromMillisecondsSinceEpoch(lastSync)
          : null;

      // Track which localIds we've seen in vault files
      final seenLocalIds = <String>{};

      // ── Step 1: Pull changes from vault files → App ────────────────────────
      await for (final entity in axonDir.list(recursive: false)) {
        if (entity is! File || !entity.path.endsWith('.md')) continue;
        if (entity.path.endsWith('_index.md')) continue;

        try {
          final fileStat = await entity.stat();
          final fileModified = fileStat.modified;

          // Only process files modified since last sync (incremental)
          if (lastSyncDt != null && fileModified.isBefore(lastSyncDt)) {
            // Still track seen IDs to detect deletions
            final existingNote = _localNotes.values.firstWhere(
              (n) => n.filePath == entity.path,
              orElse: () => ObsidianNote(
                localId: '',
                title: '', description: '', subject: '', topic: '',
                startTime: DateTime.now(), endTime: DateTime.now(),
                durationMinutes: 0, source: '', updatedAt: DateTime.now(),
              ),
            );
            if (existingNote.localId.isNotEmpty) {
              seenLocalIds.add(existingNote.localId);
            }
            continue;
          }

          final content = await entity.readAsString();
          final incoming =
              ObsidianNote.fromMarkdown(content, entity.path);
          seenLocalIds.add(incoming.localId);

          final existing = _localNotes[incoming.localId];

          if (existing == null) {
            _localNotes[incoming.localId] = incoming;
            pulled++;
          } else if (existing.source == 'app' &&
              existing.updatedAt.isAfter(incoming.updatedAt)) {
            // App has a newer version — overwrite vault file
            await entity.writeAsString(existing.toMarkdown());
            _localNotes[existing.localId] =
                existing.copyWith(filePath: entity.path, source: 'synced');
            pushed++;
          } else {
            // Vault is newer — pull changes into app
            final merged = incoming.copyWith(source: 'synced');
            _localNotes[incoming.localId] = merged;
            updated++;
          }
        } catch (e) {
          errors.add('Error reading ${entity.path}: $e');
        }
      }

      // ── Detect vault deletions ─────────────────────────────────────────────
      for (final note in _localNotes.values.toList()) {
        if (note.isDeleted || note.filePath == null) continue;
        if (!seenLocalIds.contains(note.localId) &&
            note.source == 'synced' &&
            lastSyncDt != null) {
          // File existed before, now gone → deleted externally
          _localNotes[note.localId] =
              note.copyWith(isDeleted: true, updatedAt: DateTime.now());
          deleted++;
        }
      }

      // ── Step 2: Push local-only notes → vault ─────────────────────────────
      for (final note in _localNotes.values.toList()) {
        if (note.source == 'synced' || note.source == 'obsidian') continue;
        if (note.isDeleted) continue;

        try {
          final path = note.filePath ?? _buildFilePath(note);
          await File(path).writeAsString(note.toMarkdown());
          _localNotes[note.localId] = note.copyWith(
              filePath: path, source: 'synced', updatedAt: DateTime.now());
          pushed++;
        } catch (e) {
          errors.add('Push error for ${note.localId}: $e');
        }
      }

      await _rebuildIndex();
      await _saveLocalNotes();
      await _prefs!.setInt(
          'obsidian_last_sync', DateTime.now().millisecondsSinceEpoch);
      _emitStream();
      _syncStatus = ObsidianSyncStatus.idle;

      return ObsidianSyncResult(
          pulled: pulled, pushed: pushed, updated: updated,
          deleted: deleted, errors: errors, syncedAt: DateTime.now());
    } catch (e) {
      _syncStatus = ObsidianSyncStatus.error;
      debugPrint('[Obsidian] Sync failed: $e');
      return ObsidianSyncResult(
          pulled: pulled, pushed: pushed, updated: updated,
          deleted: deleted, errors: [...errors, e.toString()],
          syncedAt: DateTime.now());
    }
  }

  // ── File System Event Handlers ─────────────────────────────────────────────

  Future<void> _handleFileChange(String filePath) async {
    if (!filePath.endsWith('.md') || filePath.endsWith('_index.md')) return;

    try {
      final file = File(filePath);
      if (!await file.exists()) return;
      final content = await file.readAsString();
      final incoming = ObsidianNote.fromMarkdown(content, filePath);
      final existing = _localNotes[incoming.localId];

      if (existing == null || incoming.updatedAt.isAfter(existing.updatedAt)) {
        _localNotes[incoming.localId] = incoming.copyWith(source: 'synced');
        await _saveLocalNotes();
        _emitStream();
      }
    } catch (e) {
      debugPrint('[Obsidian] Error handling file change: $e');
    }
  }

  Future<void> _handleFileDeletion(String filePath) async {
    try {
      // Find the note that owned this file
      final entry = _localNotes.entries.firstWhere(
        (e) => e.value.filePath == filePath,
        orElse: () => MapEntry('', _localNotes.values.first),
      );
      if (entry.key.isNotEmpty) {
        _localNotes[entry.key] = entry.value.copyWith(
            isDeleted: true, updatedAt: DateTime.now());
        await _saveLocalNotes();
        _emitStream();
      }
    } catch (_) {}
  }

  // ── File Path Builder ──────────────────────────────────────────────────────

  String _buildFilePath(ObsidianNote note) {
    final dateStr = '${note.startTime.year}-'
        '${note.startTime.month.toString().padLeft(2, '0')}-'
        '${note.startTime.day.toString().padLeft(2, '0')}';
    final safeTitle = note.title
        .replaceAll(RegExp(r'[^\w\s-]'), '')
        .replaceAll(' ', '-')
        .toLowerCase();
    return '$_axonDirPath/$dateStr-$safeTitle.md';
  }

  // ── Index File Rebuild ─────────────────────────────────────────────────────

  Future<void> _rebuildIndex() async {
    if (!_isConnected) return;
    try {
      final sb = StringBuffer()
        ..writeln('# Axon Study Plans')
        ..writeln('> Auto-generated by Axon App — do not edit manually')
        ..writeln()
        ..writeln('| Date | Subject | Topic | Duration | Completed |')
        ..writeln('|------|---------|-------|----------|-----------|');

      for (final note in notes) {
        final dateStr = '${note.startTime.year}-'
            '${note.startTime.month.toString().padLeft(2, '0')}-'
            '${note.startTime.day.toString().padLeft(2, '0')}';
        final filename =
            note.filePath?.split('/').last ?? '${note.localId}.md';
        final completed =
            note.completedTasks.length + note.tasks.length == 0
                ? '-'
                : '${note.completedTasks.length}/${note.completedTasks.length + note.tasks.length}';
        sb.writeln(
            '| $dateStr | ${note.subject} | [[${filename.replaceAll('.md', '')}\\|${note.title}]] | ${note.durationMinutes}m | $completed |');
      }

      await File('$_axonDirPath/_index.md').writeAsString(sb.toString());
    } catch (e) {
      debugPrint('[Obsidian] Index rebuild failed: $e');
    }
  }

  // ── Local Persistence ──────────────────────────────────────────────────────

  Future<void> _saveLocalNotes() async {
    _prefs ??= await SharedPreferences.getInstance();
    final json =
        jsonEncode(_localNotes.values.map((n) => n.toJson()).toList());
    await _prefs!.setString('obsidian_local_notes', json);
  }

  Future<void> _loadLocalNotes() async {
    _prefs ??= await SharedPreferences.getInstance();
    final raw = _prefs!.getString('obsidian_local_notes');
    if (raw == null) return;
    try {
      final list = jsonDecode(raw) as List;
      for (final item in list) {
        final note =
            ObsidianNote.fromJson(Map<String, dynamic>.from(item as Map));
        _localNotes[note.localId] = note;
      }
    } catch (e) {
      debugPrint('[Obsidian] Failed to load local notes: $e');
    }
  }

  void _emitStream() {
    if (!_streamController.isClosed) _streamController.add(notes);
  }

  // ── Convenience ────────────────────────────────────────────────────────────

  List<ObsidianNote> getNotesForDay(DateTime day) => notes
      .where((n) =>
          n.startTime.year == day.year &&
          n.startTime.month == day.month &&
          n.startTime.day == day.day)
      .toList();

  Future<List<String>> getVaultFolders() async {
    if (_vaultPath == null) return [];
    final vaultDir = Directory(_vaultPath!);
    if (!await vaultDir.exists()) return [];
    final folders = <String>[];
    await for (final entity in vaultDir.list(followLinks: false)) {
      if (entity is Directory) folders.add(entity.path.split('/').last);
    }
    return folders;
  }

  DateTime? get lastSyncedAt {
    final ms = _prefs?.getInt('obsidian_last_sync');
    return ms != null ? DateTime.fromMillisecondsSinceEpoch(ms) : null;
  }

  void dispose() {
    stopRealtimeSync();
    _streamController.close();
  }

  Future<void> syncStudyPlan(List<Map<String, dynamic>> tasks) async {
    debugPrint('ObsidianService: syncStudyPlan - ${tasks.length} tasks (stub)');
  }
}

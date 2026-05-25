import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class LocalDatabaseService {
  static final LocalDatabaseService _instance =
      LocalDatabaseService._internal();
  static Database? _database;

  factory LocalDatabaseService() => _instance;

  LocalDatabaseService._internal();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'axon_local.db');

    return await openDatabase(
      path,
      version: 1,
      onCreate: _onCreate,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE user_interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        topic_id TEXT,
        interaction_type TEXT NOT NULL,
        content_id TEXT,
        duration_seconds INTEGER DEFAULT 0,
        score REAL,
        timestamp INTEGER NOT NULL,
        metadata TEXT
      )
    ''');

    await db.execute('''
      CREATE TABLE mock_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        subject_code TEXT NOT NULL,
        paper_code TEXT,
        chapter_id TEXT,
        total_marks REAL NOT NULL,
        obtained_marks REAL NOT NULL,
        percentage REAL NOT NULL,
        attempt_date INTEGER NOT NULL,
        time_taken_seconds INTEGER DEFAULT 0,
        source TEXT
      )
    ''');

    await db.execute('''
      CREATE TABLE chapter_progress (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        subject_code TEXT NOT NULL,
        chapter_id TEXT NOT NULL,
        subchapter_id TEXT,
        completion_percentage REAL DEFAULT 0,
        questions_attempted INTEGER DEFAULT 0,
        questions_correct INTEGER DEFAULT 0,
        time_spent_seconds INTEGER DEFAULT 0,
        last_accessed INTEGER,
        is_mastered INTEGER DEFAULT 0
      )
    ''');

    await db.execute('''
      CREATE TABLE study_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        subject_code TEXT,
        start_time INTEGER NOT NULL,
        end_time INTEGER,
        duration_seconds INTEGER DEFAULT 0,
        session_type TEXT,
        topics_covered INTEGER DEFAULT 0,
        questions_attempted INTEGER DEFAULT 0
      )
    ''');

    await db.execute('''
      CREATE TABLE growth_projections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        subject_code TEXT NOT NULL,
        projection_date INTEGER NOT NULL,
        predicted_completion REAL NOT NULL,
        current_velocity REAL NOT NULL,
        required_velocity REAL NOT NULL,
        days_remaining INTEGER NOT NULL,
        trajectory TEXT
      )
    ''');

    await db.execute('''
      CREATE TABLE sync_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        table_name TEXT NOT NULL,
        operation TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        retry_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending'
      )
    ''');

    await db.execute('''
      CREATE TABLE user_exam_dates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        subject_code TEXT NOT NULL,
        subject_name TEXT,
        exam_code TEXT NOT NULL,
        component TEXT,
        exam_date INTEGER NOT NULL,
        exam_time TEXT,
        duration TEXT,
        exam_type TEXT,
        inserted_at INTEGER NOT NULL,
        UNIQUE(user_id, subject_code, exam_code)
      )
    ''');

    await db.execute(
        'CREATE INDEX idx_interactions_user ON user_interactions(user_id)');
    await db.execute(
        'CREATE INDEX idx_interactions_topic ON user_interactions(topic_id)');
    await db.execute('CREATE INDEX idx_scores_user ON mock_scores(user_id)');
    await db.execute(
        'CREATE INDEX idx_progress_user ON chapter_progress(user_id, subject_code)');
    await db
        .execute('CREATE INDEX idx_sessions_user ON study_sessions(user_id)');
    await db.execute('CREATE INDEX idx_sync_status ON sync_queue(status)');
  }

  Future<int> insertInteraction(Map<String, dynamic> interaction) async {
    final db = await database;
    return await db.insert('user_interactions', interaction);
  }

  Future<List<Map<String, dynamic>>> getUserInteractions(String userId,
      {String? topicId, int? limit}) async {
    final db = await database;
    String where = 'user_id = ?';
    List<dynamic> args = [userId];

    if (topicId != null) {
      where += ' AND topic_id = ?';
      args.add(topicId);
    }

    return await db.query(
      'user_interactions',
      where: where,
      whereArgs: args,
      orderBy: 'timestamp DESC',
      limit: limit,
    );
  }

  Future<int> insertMockScore(Map<String, dynamic> score) async {
    final db = await database;
    return await db.insert('mock_scores', score);
  }

  Future<List<Map<String, dynamic>>> getUserMockScores(String userId,
      {String? subjectCode, int? limit}) async {
    final db = await database;
    String where = 'user_id = ?';
    List<dynamic> args = [userId];

    if (subjectCode != null) {
      where += ' AND subject_code = ?';
      args.add(subjectCode);
    }

    return await db.query(
      'mock_scores',
      where: where,
      whereArgs: args,
      orderBy: 'attempt_date DESC',
      limit: limit,
    );
  }

  Future<Map<String, dynamic>?> getLatestMockScore(
      String userId, String subjectCode) async {
    final db = await database;
    final results = await db.query(
      'mock_scores',
      where: 'user_id = ? AND subject_code = ?',
      whereArgs: [userId, subjectCode],
      orderBy: 'attempt_date DESC',
      limit: 1,
    );
    return results.isNotEmpty ? results.first : null;
  }

  Future<int> insertOrUpdateChapterProgress(
      Map<String, dynamic> progress) async {
    final db = await database;
    final existing = await db.query(
      'chapter_progress',
      where:
          'user_id = ? AND subject_code = ? AND chapter_id = ? AND subchapter_id = ?',
      whereArgs: [
        progress['user_id'],
        progress['subject_code'],
        progress['chapter_id'],
        progress['subchapter_id'],
      ],
    );

    if (existing.isNotEmpty) {
      return await db.update(
        'chapter_progress',
        progress,
        where: 'id = ?',
        whereArgs: [existing.first['id']],
      );
    } else {
      return await db.insert('chapter_progress', progress);
    }
  }

  Future<List<Map<String, dynamic>>> getUserChapterProgress(
      String userId, String subjectCode) async {
    final db = await database;
    return await db.query(
      'chapter_progress',
      where: 'user_id = ? AND subject_code = ?',
      whereArgs: [userId, subjectCode],
      orderBy: 'last_accessed DESC',
    );
  }

  Future<double> getOverallProgress(String userId, String subjectCode) async {
    final db = await database;
    final result = await db.rawQuery(
      'SELECT AVG(completion_percentage) as avg_progress FROM chapter_progress WHERE user_id = ? AND subject_code = ?',
      [userId, subjectCode],
    );
    return (result.first['avg_progress'] as num?)?.toDouble() ?? 0.0;
  }

  Future<int> insertStudySession(Map<String, dynamic> session) async {
    final db = await database;
    return await db.insert('study_sessions', session);
  }

  Future<List<Map<String, dynamic>>> getUserStudySessions(String userId,
      {int? days}) async {
    final db = await database;
    String where = 'user_id = ?';
    List<dynamic> args = [userId];

    if (days != null) {
      final since =
          DateTime.now().subtract(Duration(days: days)).millisecondsSinceEpoch;
      where += ' AND start_time > ?';
      args.add(since);
    }

    return await db.query(
      'study_sessions',
      where: where,
      whereArgs: args,
      orderBy: 'start_time DESC',
    );
  }

  Future<int> getTotalStudyTimeToday(String userId) async {
    final db = await database;
    final todayStart = DateTime.now()
        .copyWith(hour: 0, minute: 0, second: 0, millisecond: 0)
        .millisecondsSinceEpoch;

    final result = await db.rawQuery(
      'SELECT SUM(duration_seconds) as total FROM study_sessions WHERE user_id = ? AND start_time >= ?',
      [userId, todayStart],
    );
    return (result.first['total'] as int?) ?? 0;
  }

  Future<int> insertGrowthProjection(Map<String, dynamic> projection) async {
    final db = await database;

    await db.delete(
      'growth_projections',
      where: 'user_id = ? AND subject_code = ?',
      whereArgs: [projection['user_id'], projection['subject_code']],
    );

    return await db.insert('growth_projections', projection);
  }

  Future<Map<String, dynamic>?> getLatestGrowthProjection(
      String userId, String subjectCode) async {
    final db = await database;
    final results = await db.query(
      'growth_projections',
      where: 'user_id = ? AND subject_code = ?',
      whereArgs: [userId, subjectCode],
      orderBy: 'projection_date DESC',
      limit: 1,
    );
    return results.isNotEmpty ? results.first : null;
  }

  Future<int> addToSyncQueue(Map<String, dynamic> item) async {
    final db = await database;
    return await db.insert('sync_queue', item);
  }

  Future<List<Map<String, dynamic>>> getPendingSyncItems(
      {int limit = 50}) async {
    final db = await database;
    return await db.query(
      'sync_queue',
      where: 'status = ? AND retry_count < ?',
      whereArgs: ['pending', 3],
      orderBy: 'created_at ASC',
      limit: limit,
    );
  }

  Future<int> updateSyncItemStatus(int id, String status) async {
    final db = await database;
    return await db.update(
      'sync_queue',
      {'status': status, 'retry_count': status == 'failed' ? 1 : 0},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<int> incrementSyncRetry(int id) async {
    final db = await database;
    return await db.rawUpdate(
      'UPDATE sync_queue SET retry_count = retry_count + 1 WHERE id = ?',
      [id],
    );
  }

  Future<Map<String, dynamic>> getUserStudyAnalytics(String userId) async {
    final db = await database;

    final totalSessions = Sqflite.firstIntValue(
          await db.rawQuery(
              'SELECT COUNT(*) FROM study_sessions WHERE user_id = ?',
              [userId]),
        ) ??
        0;

    final totalTime = Sqflite.firstIntValue(
          await db.rawQuery(
              'SELECT SUM(duration_seconds) FROM study_sessions WHERE user_id = ?',
              [userId]),
        ) ??
        0;

    final totalQuestions = Sqflite.firstIntValue(
          await db.rawQuery(
              'SELECT SUM(questions_attempted) FROM study_sessions WHERE user_id = ?',
              [userId]),
        ) ??
        0;

    final avgScore = await db.rawQuery(
      'SELECT AVG(percentage) as avg FROM mock_scores WHERE user_id = ?',
      [userId],
    );

    final streakData = await db.rawQuery('''
      SELECT DATE(start_time/1000, 'unixepoch') as study_date, COUNT(*) as sessions
      FROM study_sessions
      WHERE user_id = ?
      GROUP BY DATE(start_time/1000, 'unixepoch')
      ORDER BY study_date DESC
      LIMIT 7
    ''', [userId]);

    return {
      'total_sessions': totalSessions,
      'total_study_time_seconds': totalTime,
      'total_questions_attempted': totalQuestions,
      'average_score': (avgScore.first['avg'] as num?)?.toDouble() ?? 0.0,
      'recent_streak': streakData.length,
    };
  }

  // ============ Exam Dates Methods ============

  Future<int> insertExamDates(
      List<Map<String, dynamic>> examDates, String userId) async {
    final db = await database;
    int inserted = 0;
    final now = DateTime.now().millisecondsSinceEpoch;

    for (final exam in examDates) {
      try {
        await db.insert(
          'user_exam_dates',
          {
            'user_id': userId,
            'subject_code': exam['subject_code'],
            'subject_name': exam['subject_name'],
            'exam_code': exam['exam_code'],
            'component': exam['component'],
            'exam_date': exam['exam_date'],
            'exam_time': exam['exam_time'],
            'duration': exam['duration'],
            'exam_type': exam['exam_type'],
            'inserted_at': now,
          },
          conflictAlgorithm: ConflictAlgorithm.replace,
        );
        inserted++;
      } catch (e) {
        // Skip duplicates
      }
    }
    return inserted;
  }

  Future<List<Map<String, dynamic>>> getUserExamDates(String userId,
      {String? subjectCode}) async {
    final db = await database;
    String where = 'user_id = ?';
    List<dynamic> args = [userId];

    if (subjectCode != null) {
      where += ' AND subject_code = ?';
      args.add(subjectCode);
    }

    return await db.query(
      'user_exam_dates',
      where: where,
      whereArgs: args,
      orderBy: 'exam_date ASC',
    );
  }

  Future<List<Map<String, dynamic>>> getUpcomingExamDates(String userId) async {
    final db = await database;
    final now = DateTime.now().millisecondsSinceEpoch;

    return await db.query(
      'user_exam_dates',
      where: 'user_id = ? AND exam_date > ?',
      whereArgs: [userId, now],
      orderBy: 'exam_date ASC',
    );
  }

  Future<int> deleteExamDatesForSubject(
      String userId, String subjectCode) async {
    final db = await database;
    return await db.delete(
      'user_exam_dates',
      where: 'user_id = ? AND subject_code = ?',
      whereArgs: [userId, subjectCode],
    );
  }

  Future<int> clearAllExamDates(String userId) async {
    final db = await database;
    return await db.delete(
      'user_exam_dates',
      where: 'user_id = ?',
      whereArgs: [userId],
    );
  }
}

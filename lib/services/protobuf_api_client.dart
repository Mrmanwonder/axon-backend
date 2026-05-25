import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'local_database_service.dart';
import 'axon_api_service.dart';

class ApiClient {
  final String baseUrl;
  final bool useProtobuf;

  ApiClient({required this.baseUrl, this.useProtobuf = true});

  Future<Map<String, dynamic>?> get(String endpoint,
      {Map<String, String>? params}) async {
    try {
      final uri =
          Uri.parse('$baseUrl$endpoint').replace(queryParameters: params);
      final response = await http.get(uri, headers: _headers);

      if (response.statusCode == 200) {
        return _decodeResponse(response.bodyBytes);
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  Future<Map<String, dynamic>?> post(String endpoint,
      {Map<String, dynamic>? data}) async {
    try {
      final uri = Uri.parse('$baseUrl$endpoint');
      final body = _encodeBody(data ?? {});
      final response = await http.post(uri, headers: _headers, body: body);

      if (response.statusCode == 200 || response.statusCode == 201) {
        return _decodeResponse(response.bodyBytes);
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  Map<String, String> get _headers {
    if (useProtobuf) {
      return {
        'Content-Type': 'application/octet-stream',
        'Accept': 'application/octet-stream',
        'X-Client-Version': '1.0.0',
      };
    }
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Client-Version': '1.0.0',
    };
  }

  Uint8List _encodeBody(Map<String, dynamic> data) {
    if (useProtobuf) {
      return ProtoCodec.encode(data);
    }
    return Uint8List.fromList(utf8.encode(jsonEncode(data)));
  }

  Map<String, dynamic>? _decodeResponse(Uint8List bytes) {
    if (useProtobuf) {
      if (bytes.isEmpty) return null;
      return ProtoCodec.decode(bytes);
    }
    return jsonDecode(utf8.decode(bytes));
  }

  Future<void> queueOffline(
      String endpoint, Map<String, dynamic> data, String tableName) async {
    await LocalDatabaseService().addToSyncQueue({
      'user_id': data['user_id'] ?? '',
      'table_name': tableName,
      'operation': 'sync',
      'payload': jsonEncode(data),
      'created_at': DateTime.now().millisecondsSinceEpoch,
      'status': 'pending',
    });
  }
}

class LeaderboardApiService {
  final ApiClient _client;

  LeaderboardApiService({String baseUrl = 'https://api.axon-study.com'})
      : _client = ApiClient(baseUrl: baseUrl, useProtobuf: true);

  Future<List<Map<String, dynamic>>> getLeaderboard(
      {String? subjectCode, int limit = 50}) async {
    final data = await _client.get('/leaderboard', params: {
      if (subjectCode != null) 'subject': subjectCode,
      'limit': limit.toString(),
    });

    if (data == null || data['leaderboard'] == null) return [];
    return (data['leaderboard'] as List).cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>?> getUserRank(String odentId,
      {String? subjectCode}) async {
    return await _client.get('/leaderboard/rank', params: {
      'user_id': odentId,
      if (subjectCode != null) 'subject': subjectCode,
    });
  }

  Future<void> submitScore(String odentId, double score,
      {String? subjectCode, String? paperCode}) async {
    await _client.post('/leaderboard/score', data: {
      'user_id': odentId,
      'score': score,
      if (subjectCode != null) 'subject_code': subjectCode,
      if (paperCode != null) 'paper_code': paperCode,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });
  }
}

class StudyPulseApiService {
  final ApiClient _client;

  StudyPulseApiService({String baseUrl = 'https://api.axon-study.com'})
      : _client = ApiClient(baseUrl: baseUrl, useProtobuf: true);

  Future<void> pulse(String odentId, String eventType,
      {Map<String, dynamic>? metadata}) async {
    try {
      await _client.post('/pulse', data: {
        'user_id': odentId,
        'event_type': eventType,
        'metadata': metadata ?? {},
        'timestamp': DateTime.now().millisecondsSinceEpoch,
      });
    } catch (e) {
      await _client.queueOffline(
          '/pulse',
          {
            'user_id': odentId,
            'event_type': eventType,
            'metadata': metadata ?? {},
            'timestamp': DateTime.now().millisecondsSinceEpoch,
          },
          'study_pulse');
    }
  }

  Future<Map<String, dynamic>?> getInsights(String odentId) async {
    return await _client.get('/pulse/insights', params: {'user_id': odentId});
  }
}

class GrowthProjectionApiService {
  final ApiClient _client;

  GrowthProjectionApiService({String baseUrl = 'https://api.axon-study.com'})
      : _client = ApiClient(baseUrl: baseUrl, useProtobuf: true);

  Future<Map<String, dynamic>?> getDailyProjection(
      String odentId, String subjectCode) async {
    return await _client.get('/growth/projection', params: {
      'user_id': odentId,
      'subject_code': subjectCode,
    });
  }

  Future<List<Map<String, dynamic>>> getWeeklyTrajectory(
      String odentId, String subjectCode) async {
    final data = await _client.get('/growth/trajectory', params: {
      'user_id': odentId,
      'subject_code': subjectCode,
      'range': '7days',
    });

    if (data == null || data['trajectory'] == null) return [];
    return (data['trajectory'] as List).cast<Map<String, dynamic>>();
  }

  Future<void> updateVelocity(
      String odentId, String subjectCode, double velocity) async {
    await _client.post('/growth/velocity', data: {
      'user_id': odentId,
      'subject_code': subjectCode,
      'velocity': velocity,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });
  }
}

class ExamZoneApiService {
  final ApiClient _client;

  ExamZoneApiService({String baseUrl = 'https://api.axon-study.com'})
      : _client = ApiClient(baseUrl: baseUrl, useProtobuf: true);

  Future<List<Map<String, dynamic>>> getUpcomingExams(String odentId,
      {String? subjectCode}) async {
    final data = await _client.get('/exams/upcoming', params: {
      'user_id': odentId,
      if (subjectCode != null) 'subject': subjectCode,
    });

    if (data == null || data['exams'] == null) return [];
    return (data['exams'] as List).cast<Map<String, dynamic>>();
  }

  Future<Map<String, dynamic>?> getExamZone(
      String odentId, String examId) async {
    return await _client.get('/exams/zone', params: {
      'user_id': odentId,
      'exam_id': examId,
    });
  }

  Future<void> registerExam(
      String odentId, String subjectCode, DateTime examDate) async {
    await _client.post('/exams/register', data: {
      'user_id': odentId,
      'subject_code': subjectCode,
      'exam_date': examDate.millisecondsSinceEpoch,
    });
  }
}

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'local_database_service.dart';

class ProtoCodec {
  static Uint8List encode(Map<String, dynamic> data) {
    final buffer = BytesBuilder();

    void writeStringField(int fieldNumber, String value) {
      final bytes = Uint8List.fromList(utf8.encode(value));
      final tag = (fieldNumber << 3) | 2;
      buffer.add(_encodeVarint(tag));
      buffer.add(_encodeVarint(bytes.length));
      buffer.add(bytes);
    }

    void writeInt32Field(int fieldNumber, int value) {
      final tag = (fieldNumber << 3) | 0;
      buffer.add(_encodeVarint(tag));
      buffer.add(_encodeVarint(value));
    }

    void writeDoubleField(int fieldNumber, double value) {
      final tag = (fieldNumber << 3) | 1;
      buffer.add(_encodeVarint(tag));
      buffer.add(_encodeFixed64(_doubleToBytes(value)));
    }

    for (final entry in data.entries) {
      final key = entry.key;
      final value = entry.value;

      if (key == 'id' ||
          key == 'timestamp' ||
          key == 'duration_seconds' ||
          key == 'attempt_date' ||
          key == 'projection_date' ||
          key == 'days_remaining' ||
          key == 'retry_count' ||
          key == 'created_at') {
        writeInt32Field(_getFieldNumber(key), value as int);
      } else if (key == 'score' ||
          key == 'percentage' ||
          key == 'completion_percentage' ||
          key == 'predicted_completion' ||
          key == 'current_velocity' ||
          key == 'required_velocity' ||
          key == 'average_score') {
        writeDoubleField(_getFieldNumber(key), value as double);
      } else if (key == 'user_id' ||
          key == 'topic_id' ||
          key == 'interaction_type' ||
          key == 'content_id' ||
          key == 'subject_code' ||
          key == 'paper_code' ||
          key == 'chapter_id' ||
          key == 'subchapter_id' ||
          key == 'session_type' ||
          key == 'source' ||
          key == 'table_name' ||
          key == 'operation' ||
          key == 'status' ||
          key == 'metadata') {
        writeStringField(_getFieldNumber(key), value.toString());
      } else if (key == 'is_mastered' || key == 'is_completed') {
        writeInt32Field(_getFieldNumber(key), value == true ? 1 : 0);
      }
    }

    return buffer.toBytes();
  }

  static Map<String, dynamic> decode(Uint8List bytes) {
    final result = <String, dynamic>{};
    int position = 0;

    while (position < bytes.length) {
      final tag = _decodeVarint(bytes, position);
      position += _varintLength(tag);

      final fieldNumber = tag >> 3;
      final wireType = tag & 0x7;

      switch (wireType) {
        case 0:
          final value = _decodeVarint(bytes, position);
          position += _varintLength(value);
          result[_getFieldName(fieldNumber)] = value;
          break;
        case 1:
          final value = _decodeFixed64(bytes, position);
          position += 8;
          result[_getFieldName(fieldNumber)] = _bytesToDouble(value);
          break;
        case 2:
          final length = _decodeVarint(bytes, position);
          position += _varintLength(length);
          final value = bytes.sublist(position, position + length);
          position += length;
          result[_getFieldName(fieldNumber)] = utf8.decode(value);
          break;
      }
    }

    return result;
  }

  static int _getFieldNumber(String name) {
    const fieldNumbers = {
      'id': 1,
      'user_id': 2,
      'topic_id': 3,
      'interaction_type': 4,
      'content_id': 5,
      'duration_seconds': 6,
      'score': 7,
      'timestamp': 8,
      'metadata': 9,
      'subject_code': 10,
      'paper_code': 11,
      'chapter_id': 12,
      'total_marks': 13,
      'obtained_marks': 14,
      'percentage': 15,
      'attempt_date': 16,
      'time_taken_seconds': 17,
      'source': 18,
      'subchapter_id': 19,
      'completion_percentage': 20,
      'questions_attempted': 21,
      'questions_correct': 22,
      'time_spent_seconds': 23,
      'last_accessed': 24,
      'is_mastered': 25,
      'start_time': 26,
      'end_time': 27,
      'session_type': 28,
      'topics_covered': 29,
      'projection_date': 30,
      'predicted_completion': 31,
      'current_velocity': 32,
      'required_velocity': 33,
      'days_remaining': 34,
      'trajectory': 35,
      'table_name': 36,
      'operation': 37,
      'payload': 38,
      'created_at': 39,
      'retry_count': 40,
      'status': 41,
    };
    return fieldNumbers[name] ?? 0;
  }

  static String _getFieldName(int fieldNumber) {
    const fieldNames = {
      1: 'id',
      2: 'user_id',
      3: 'topic_id',
      4: 'interaction_type',
      5: 'content_id',
      6: 'duration_seconds',
      7: 'score',
      8: 'timestamp',
      9: 'metadata',
      10: 'subject_code',
      11: 'paper_code',
      12: 'chapter_id',
      13: 'total_marks',
      14: 'obtained_marks',
      15: 'percentage',
      16: 'attempt_date',
      17: 'time_taken_seconds',
      18: 'source',
      19: 'subchapter_id',
      20: 'completion_percentage',
      21: 'questions_attempted',
      22: 'questions_correct',
      23: 'time_spent_seconds',
      24: 'last_accessed',
      25: 'is_mastered',
      26: 'start_time',
      27: 'end_time',
      28: 'session_type',
      29: 'topics_covered',
      30: 'projection_date',
      31: 'predicted_completion',
      32: 'current_velocity',
      33: 'required_velocity',
      34: 'days_remaining',
      35: 'trajectory',
      36: 'table_name',
      37: 'operation',
      38: 'payload',
      39: 'created_at',
      40: 'retry_count',
      41: 'status',
    };
    return fieldNames[fieldNumber] ?? 'unknown_$fieldNumber';
  }

  static int wireType(dynamic value) {
    if (value is int) return 0;
    if (value is double) return 1;
    if (value is String || value is List || value is Map) return 2;
    return 2;
  }

  static Uint8List _encodeVarint(int value) {
    final bytes = <int>[];
    while (value > 0x7F) {
      bytes.add((value & 0x7F) | 0x80);
      value >>= 7;
    }
    bytes.add(value);
    return Uint8List.fromList(bytes);
  }

  static int _decodeVarint(Uint8List bytes, int position) {
    int result = 0;
    int shift = 0;
    while (true) {
      final byte = bytes[position++];
      result |= (byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) break;
      shift += 7;
    }
    return result;
  }

  static int _varintLength(int value) {
    int length = 0;
    while (value > 0) {
      length++;
      value >>= 7;
    }
    return length == 0 ? 1 : length;
  }

  static Uint8List _encodeFixed64(Uint8List bytes) => bytes;

  static Uint8List _doubleToBytes(double value) {
    final byteData = ByteData(8);
    byteData.setFloat64(0, value);
    return byteData.buffer.asUint8List();
  }

  static double _bytesToDouble(Uint8List bytes) {
    final byteData = ByteData.sublistView(bytes);
    return byteData.getFloat64(0);
  }

  static Uint8List _decodeFixed64(Uint8List bytes, int position) {
    return bytes.sublist(position, position + 8);
  }
}

class AxonApiService {
  static const String _baseUrl = 'https://api.axon-study.com';

  Future<Uint8List> syncUserData(
      String odentId, Map<String, dynamic> data) async {
    final payload = {
      'user_id': odentId,
      ...data,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    };

    final encoded = ProtoCodec.encode(payload);

    final response = await _makeRequest('/sync', 'POST', encoded);
    return response;
  }

  Future<Uint8List> submitInteraction(
      String odentId, Map<String, dynamic> interaction) async {
    final payload = {
      'user_id': odentId,
      ...interaction,
    };

    final encoded = ProtoCodec.encode(payload);
    return await _makeRequest('/interactions', 'POST', encoded);
  }

  Future<Uint8List> submitMockScore(
      String odentId, Map<String, dynamic> score) async {
    final payload = {
      'user_id': odentId,
      ...score,
    };

    final encoded = ProtoCodec.encode(payload);
    return await _makeRequest('/scores', 'POST', encoded);
  }

  Future<Map<String, dynamic>?> getGrowthProjection(
      String odentId, String subjectCode) async {
    final data = {
      'user_id': odentId,
      'subject_code': subjectCode,
    };

    final encoded = ProtoCodec.encode(data);
    final response = await _makeRequest('/growth-projection', 'GET', encoded);

    if (response.isEmpty) return null;
    return ProtoCodec.decode(response);
  }

  Future<List<Map<String, dynamic>>> fetchDailyUpdates(String odentId) async {
    final data = {
      'user_id': odentId,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    };

    final encoded = ProtoCodec.encode(data);
    final response = await _makeRequest('/daily-updates', 'GET', encoded);

    if (response.isEmpty) return [];

    final decoded = ProtoCodec.decode(response);
    if (decoded['updates'] is List) {
      return (decoded['updates'] as List).cast<Map<String, dynamic>>();
    }
    return [];
  }

  Future<Uint8List> _makeRequest(
      String endpoint, String method, Uint8List body) async {
    try {
      final client = HttpClient();
      final uri = Uri.parse('$_baseUrl$endpoint');

      final request = await client.openUrl(method, uri);
      request.headers.contentType = ContentType('application', 'octet-stream');
      request.headers.set('X-Client-Version', '1.0.0');
      request.headers.set('Accept', 'application/octet-stream');

      if (method != 'GET') {
        request.add(body);
      }

      final response = await request.close();
      final responseBody = await response.fold<List<int>>(
        [],
        (prev, element) => prev..addAll(element),
      );

      return Uint8List.fromList(responseBody);
    } catch (e) {
      return Uint8List(0);
    }
  }

  Future<void> queueForOfflineSync(
      Map<String, dynamic> data, String tableName) async {
    await LocalDatabaseService().addToSyncQueue({
      'user_id': data['user_id'],
      'table_name': tableName,
      'operation': 'insert',
      'payload': jsonEncode(data),
      'created_at': DateTime.now().millisecondsSinceEpoch,
      'status': 'pending',
    });
  }

  Future<void> processSyncQueue() async {
    final pendingItems = await LocalDatabaseService().getPendingSyncItems();

    for (final item in pendingItems) {
      try {
        final data = jsonDecode(item['payload'] as String);
        await syncUserData(item['user_id'] as String, data);
        await LocalDatabaseService()
            .updateSyncItemStatus(item['id'] as int, 'synced');
      } catch (e) {
        await LocalDatabaseService().incrementSyncRetry(item['id'] as int);
        if ((item['retry_count'] as int) >= 2) {
          await LocalDatabaseService()
              .updateSyncItemStatus(item['id'] as int, 'failed');
        }
      }
    }
  }
}

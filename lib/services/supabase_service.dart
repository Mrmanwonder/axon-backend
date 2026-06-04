import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class SupabaseService {
  SupabaseService._();
  static final SupabaseService _instance = SupabaseService._();
  static SupabaseService get instance => _instance;

  SupabaseClient? _client;
  SupabaseClient get client {
    if (_client == null) throw StateError('SupabaseService not initialized. Call init() first.');
    return _client!;
  }

  bool get isInitialized => _client != null;

  Future<void> init() async {
    if (_client != null) return;
    final url = dotenv.env['SUPABASE_URL'];
    final anonKey = dotenv.env['SUPABASE_ANON_KEY'];
    if (url == null || url.isEmpty || url == 'https://your-project.supabase.co') {
      debugPrint('SupabaseService: SUPABASE_URL not configured. Skipping initialization.');
      return;
    }
    if (anonKey == null || anonKey.isEmpty || anonKey == 'your-anon-key') {
      debugPrint('SupabaseService: SUPABASE_ANON_KEY not configured. Skipping initialization.');
      return;
    }
    await Supabase.initialize(url: url, anonKey: anonKey);
    _client = Supabase.instance.client;
    debugPrint('SupabaseService: initialized');
  }

  Future<List<Map<String, dynamic>>> query(
    String table, {
    String method = 'select',
    Map<String, dynamic> params = const {},
  }) async {
    if (!isInitialized) return [];
    try {
      final mutableParams = Map<String, dynamic>.from(params);
      final selectCols = mutableParams.remove('select') as String?;
      dynamic query = client.from(table).select(selectCols ?? '*');
      for (final entry in mutableParams.entries) {
        if (entry.key == 'select') continue;
        final key = entry.key;
        final value = entry.value;
        if (key == 'order' && value is String) {
          final parts = value.split(',');
          for (final part in parts) {
            final tokens = part.trim().split('.');
            if (tokens.length >= 2) {
              query = query.order(tokens[0], ascending: tokens[1] == 'asc');
            } else {
              query = query.order(tokens[0]);
            }
          }
        } else if (key == 'limit' && value is int) {
          query = query.limit(value);
        } else if (value is String && value.startsWith('eq.')) {
          query = query.eq(key, value.substring(3));
        } else if (value is String && value.startsWith('neq.')) {
          query = query.neq(key, value.substring(4));
        } else if (value is String && value.startsWith('gt.')) {
          query = query.gt(key, value.substring(3));
        } else if (value is String && value.startsWith('gte.')) {
          query = query.gte(key, value.substring(4));
        } else if (value is String && value.startsWith('lt.')) {
          query = query.lt(key, value.substring(3));
        } else if (value is String && value.startsWith('lte.')) {
          query = query.lte(key, value.substring(4));
        } else if (value is String && value.startsWith('in.')) {
          final items = value.substring(3).split(',');
          query = query.inFilter(key, items);
        } else if (value is String && value.startsWith('like.')) {
          query = query.like(key, value.substring(5));
        } else if (value is String && value.startsWith('ilike.')) {
          query = query.ilike(key, value.substring(6));
        }
      }
      final response = await query;
      return (response as List<dynamic>).cast<Map<String, dynamic>>();
    } catch (e) {
      debugPrint('SupabaseService: $table $method error: $e');
      return [];
    }
  }

  Future<void> mutate(
    String table, {
    required String method,
    Map<String, dynamic> body = const {},
    Map<String, dynamic> params = const {},
  }) async {
    if (!isInitialized) return;
    try {
      switch (method) {
        case 'insert':
          await client.from(table).insert(body);
        case 'upsert':
          await client.from(table).upsert(body);
        case 'update':
          var query = client.from(table).update(body);
          for (final entry in params.entries) {
            final v = entry.value;
            if (v is String && v.startsWith('eq.')) {
              query = query.eq(entry.key, v.substring(3));
            }
          }
          await query;
        case 'delete':
          var query = client.from(table).delete();
          for (final entry in params.entries) {
            final v = entry.value;
            if (v is String && v.startsWith('eq.')) {
              query = query.eq(entry.key, v.substring(3));
            }
          }
          await query;
      }
    } catch (e) {
      debugPrint('SupabaseService: $table $method error: $e');
    }
  }
}

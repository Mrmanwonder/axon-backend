import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'supabase_service.dart';
import 'caie_models.dart';

class CaiePaperRepository {
  final SupabaseService _supabase;

  CaiePaperRepository(this._supabase);

  bool get _ready => _supabase.isInitialized;

  Future<List<CaiePaper>> fetchPapers({
    String? subjectCode,
    int? year,
    String? session,
    String? variant,
    String? search,
  }) async {
    if (!_ready) return [];
    try {
      var query = _supabase.client.from('papers').select('*');
      if (subjectCode != null) query = query.eq('subject_code', subjectCode);
      if (year != null) query = query.eq('year', year);
      if (session != null) query = query.eq('session', session);
      if (variant != null) query = query.eq('variant', variant);
      if (search != null && search.isNotEmpty) {
        query = query.or(
          'subject.ilike.%$search%,subject_code.ilike.%$search%,variant.ilike.%$search%',
        );
      }
      final response = await query.order('year', ascending: false);
      return (response as List<dynamic>)
          .map((r) => _rowToPaper(r as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('CaiePaperRepository.fetchPapers error: $e');
      return [];
    }
  }

  Future<CaiePaper?> getPaper(String id) async {
    if (!_ready) return null;
    try {
      final response = await _supabase.client
          .from('papers')
          .select('*')
          .eq('id', id)
          .maybeSingle();
      if (response == null) return null;
      return _rowToPaper(response);
    } catch (e) {
      debugPrint('CaiePaperRepository.getPaper error: $e');
      return null;
    }
  }

  Future<List<CaiePaper>> searchPapers(String query) async {
    if (!_ready) return [];
    try {
      final response = await _supabase.client
          .from('papers')
          .select('*')
          .or(
            'subject.ilike.%$query%,subject_code.ilike.%$query%,variant.ilike.%$query%,session.ilike.%$query%',
          )
          .order('year', ascending: false)
          .limit(50);
      return (response as List<dynamic>)
          .map((r) => _rowToPaper(r as Map<String, dynamic>))
          .toList();
    } catch (e) {
      debugPrint('CaiePaperRepository.searchPapers error: $e');
      return [];
    }
  }

  Future<List<CaieTopic>> fetchTopics(String subjectCode) async {
    if (!_ready) return [];
    try {
      final response = await _supabase.client
          .from('topics')
          .select('*')
          .eq('subject_code', subjectCode)
          .order('name');
      return (response as List<dynamic>).map((r) {
        final row = r as Map<String, dynamic>;
        return CaieTopic(
          id: row['id'] as String,
          name: row['name'] as String,
          subjectCode: row['subject_code'] as String,
        );
      }).toList();
    } catch (e) {
      debugPrint('CaiePaperRepository.fetchTopics error: $e');
      return [];
    }
  }

  Future<Set<String>> fetchBookmarkedIds() async {
    if (!_ready) return {};
    try {
      final response = await _supabase.client
          .from('user_bookmarks')
          .select('paper_id');
      return (response as List<dynamic>)
          .map((r) => (r as Map<String, dynamic>)['paper_id'] as String)
          .toSet();
    } catch (e) {
      debugPrint('CaiePaperRepository.fetchBookmarkedIds error: $e');
      return {};
    }
  }

  Future<void> toggleBookmark(String paperId) async {
    if (!_ready) return;
    try {
      final existing = await _supabase.client
          .from('user_bookmarks')
          .select('id')
          .eq('paper_id', paperId)
          .maybeSingle();
      if (existing != null) {
        await _supabase.client
            .from('user_bookmarks')
            .delete()
            .eq('paper_id', paperId);
      } else {
        await _supabase.client.from('user_bookmarks').insert({
          'paper_id': paperId,
        });
      }
    } catch (e) {
      debugPrint('CaiePaperRepository.toggleBookmark error: $e');
    }
  }

  Future<void> recordOpen(String paperId) async {
    if (!_ready) return;
    try {
      await _supabase.client.from('user_recent_papers').upsert({
        'paper_id': paperId,
        'opened_at': DateTime.now().toIso8601String(),
      });
    } catch (e) {
      debugPrint('CaiePaperRepository.recordOpen error: $e');
    }
  }

  Future<List<CaiePaper>> fetchRecentPapers({int limit = 10}) async {
    if (!_ready) return [];
    try {
      final response = await _supabase.client
          .from('user_recent_papers')
          .select('papers(*)')
          .order('opened_at', ascending: false)
          .limit(limit);
      return (response as List<dynamic>).map((r) {
        final data = r as Map<String, dynamic>;
        final paperData = data['papers'] as Map<String, dynamic>;
        return _rowToPaper(paperData);
      }).toList();
    } catch (e) {
      debugPrint('CaiePaperRepository.fetchRecentPapers error: $e');
      return [];
    }
  }

  CaiePaper _rowToPaper(Map<String, dynamic> row) => CaiePaper(
        id: row['id'] as String? ?? '',
        subject: row['subject'] as String? ?? '',
        subjectCode: row['subject_code'] as String? ?? '',
        year: row['year'] as int? ?? 0,
        session: row['session'] as String? ?? '',
        variant: row['variant'] as String? ?? '',
        difficulty: (row['difficulty'] as num?)?.toDouble() ?? 0.5,
        accuracy: (row['accuracy'] as num?)?.toDouble() ?? 0.0,
        downloaded: row['downloaded'] as bool? ?? false,
        bookmarked: row['bookmarked'] as bool? ?? false,
        solved: row['solved'] as bool? ?? false,
      );
}

final caiePaperRepositoryProvider = Provider<CaiePaperRepository>((ref) {
  return CaiePaperRepository(SupabaseService.instance);
});

final caieSupabasePapersProvider = FutureProvider<List<CaiePaper>>((ref) async {
  final repo = ref.read(caiePaperRepositoryProvider);
  return repo.fetchPapers();
});

final caieSupabaseSearchProvider =
    FutureProvider.family<List<CaiePaper>, String>((ref, query) async {
  final repo = ref.read(caiePaperRepositoryProvider);
  return repo.searchPapers(query);
});

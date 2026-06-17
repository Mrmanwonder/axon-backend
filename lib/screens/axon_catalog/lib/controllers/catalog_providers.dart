import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models/paper.dart';
import '../models/paper_series.dart';
import '../data/series_data.dart';

// ─── Enums ──────────────────────────────────────────────────────────

enum SmartCollection { all, downloaded, saved, unsolved, recommended }
enum CatalogLayout { grid, list }

// ─── Base Data Providers ────────────────────────────────────────────

final allSeriesProvider = Provider<List<PaperSeries>>((ref) {
  return buildAllSeries();
});

final allPapersProvider = Provider<List<Paper>>((ref) {
  final series = ref.watch(allSeriesProvider);
  return series.expand((s) => s.papers).toList();
});

// ─── Filter Providers ───────────────────────────────────────────────

final catalogSearchQueryProvider = StateProvider<String>((ref) => '');

final catalogSubjectFilterProvider = StateProvider<SubjectGroup?>((ref) => null);
final catalogYearFilterProvider = StateProvider<int?>((ref) => null);
final catalogSessionFilterProvider = StateProvider<Session?>((ref) => null);
final catalogVariantFilterProvider = StateProvider<int?>((ref) => null);

final catalogSmartCollectionProvider = StateProvider<SmartCollection>((ref) => SmartCollection.all);

final catalogLayoutProvider = StateProvider<CatalogLayout>((ref) => CatalogLayout.grid);

// ─── Derived Filtered Provider ──────────────────────────────────────

final filteredPapersProvider = Provider<List<Paper>>((ref) {
  final papers = ref.watch(allPapersProvider);
  final query = ref.watch(catalogSearchQueryProvider).toLowerCase();
  final subject = ref.watch(catalogSubjectFilterProvider);
  final year = ref.watch(catalogYearFilterProvider);
  final session = ref.watch(catalogSessionFilterProvider);
  final variant = ref.watch(catalogVariantFilterProvider);
  final smartCollection = ref.watch(catalogSmartCollectionProvider);

  return papers.where((p) {
    // 1. Text Search
    if (query.isNotEmpty) {
      final text = '${p.fullReference} ${p.subject} ${p.component}'.toLowerCase();
      if (!text.contains(query)) return false;
    }

    // 2. Funnel Filters
    if (subject != null && p.subjectGroup != subject) return false;
    if (variant != null && p.variant != variant) return false;
    
    // We need to resolve series for year and session
    final s = ref.watch(allSeriesProvider).firstWhere((s) => s.id == p.seriesId);
    if (year != null && s.year != year) return false;
    if (session != null && s.session != session) return false;

    // 3. Smart Collections (Mocked logic for now)
    switch (smartCollection) {
      case SmartCollection.downloaded:
        // Mock: say variants 1 are downloaded
        if (p.variant != 1) return false;
        break;
      case SmartCollection.saved:
        // Mock: say paper 4 is saved
        if (p.paperNumber != 4) return false;
        break;
      case SmartCollection.unsolved:
        // Mock
        if (p.totalMarks < 50) return false;
        break;
      case SmartCollection.recommended:
        // Mock
        if (p.variant != 2) return false;
        break;
      case SmartCollection.all:
      default:
        break;
    }

    return true;
  }).toList();
});

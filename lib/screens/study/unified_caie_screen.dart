// lib/screens/study/unified_caie_screen.dart
// Unified CAIE Papers Screen — live backend, zero mock data
// ─────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/caie_models.dart';
import '../../services/caie_offline_store.dart';
import '../../services/caie_search_index.dart';
import '../../services/caie_topic_mapping.dart';
import '../../services/caie_topic_engine.dart';
import '../../services/caie_recommendation_engine.dart';
import '../../services/caie_command_palette.dart';
import '../../services/caie_paper_repository.dart';

// ── Enums ──────────────────────────────────────────────────────────

enum _StudyMode { browse, unsolved, weakTopics, timedPractice }
enum _ResponsiveLayout { mobile, tablet, desktop }

// ── Providers ──────────────────────────────────────────────────────

final _searchProvider = StateProvider<String>((_) => '');
final _selectedSubjectProvider = StateProvider<String?>((_) => null);
final _selectedYearProvider = StateProvider<int?>((_) => null);
final _selectedSessionProvider = StateProvider<String>((_) => 'All');
final _selectedVariantProvider = StateProvider<String>((_) => 'All');
final _studyModeProvider = StateProvider<_StudyMode>((_) => _StudyMode.browse);
final _commandPaletteOpenProvider = StateProvider<bool>((_) => false);

final _bookmarkIdsProvider = StateProvider<Set<String>>((_) => {});
final _recentIdsProvider = StateProvider<List<String>>((_) => []);
final _seededProvider = StateProvider<bool>((_) => false);
final _papersCacheProvider = StateProvider<List<CaiePaper>>((_) => []);

final _searchIndexProvider = Provider<CaieSearchIndex>((ref) {
  return CaieSearchIndex(CaieOfflineStore());
});

final _topicMappingProvider = Provider<CaieTopicMapping>((ref) {
  return CaieTopicMapping(CaieOfflineStore());
});

final _topicEngineProvider = Provider<CaieTopicEngine>((ref) {
  return CaieTopicEngine(CaieOfflineStore(), ref.read(_topicMappingProvider));
});

final _recommendationEngineProvider = Provider<CaieRecommendationEngine>((ref) {
  return CaieRecommendationEngine(CaieOfflineStore(), ref.read(_topicEngineProvider));
});

final _commandPaletteServiceProvider = Provider<CaieCommandPaletteService>((ref) {
  return CaieCommandPaletteService(CaieOfflineStore(), ref.read(_searchIndexProvider));
});

final _allPapersProvider = Provider<List<CaiePaper>>((ref) {
  ref.watch(_seededProvider);
  ref.watch(_bookmarkIdsProvider);
  return ref.watch(_papersCacheProvider);
});

final _papersProvider = Provider<List<CaiePaper>>((ref) {
  final all = ref.watch(_allPapersProvider);
  final search = ref.watch(_searchProvider);
  final subject = ref.watch(_selectedSubjectProvider);
  final year = ref.watch(_selectedYearProvider);
  final session = ref.watch(_selectedSessionProvider);
  final variant = ref.watch(_selectedVariantProvider);
  final mode = ref.watch(_studyModeProvider);
  final bookmarks = ref.watch(_bookmarkIdsProvider);

  return all.where((p) {
    if (subject != null && p.subject != subject) return false;
    if (year != null && p.year != year) return false;
    if (session != 'All' && p.session != session) return false;
    if (variant != 'All' && p.variant != variant) return false;
    if (mode == _StudyMode.unsolved && p.solved) return false;
    if (search.isNotEmpty) {
      final q = search.toLowerCase();
      if (!p.subject.toLowerCase().contains(q) &&
          !p.subjectCode.contains(q) &&
          !p.variant.toLowerCase().contains(q) &&
          !p.year.toString().contains(q) &&
          !p.id.contains(q)) {
        return false;
      }
    }
    return true;
  }).map((p) {
    final isBookmarked = bookmarks.contains(p.id);
    return p.bookmarked == isBookmarked ? p : p.copyWith(bookmarked: isBookmarked);
  }).toList();
});

final _weakTopicsProvider = Provider<List<CaieTopicPerformance>>((ref) {
  final all = ref.watch(_allPapersProvider);
  final engine = ref.read(_topicEngineProvider);
  final codes = all.map((p) => p.subjectCode).toSet().toList();
  final results = <CaieTopicPerformance>[];
  for (final code in codes) {
    results.addAll(engine.getWeakTopics(code));
  }
  results.sort((a, b) => a.accuracy.compareTo(b.accuracy));
  return results;
});

final _recommendationsProvider = Provider<List<CaieRecommendation>>((ref) {
  final engine = ref.read(_recommendationEngineProvider);
  return engine.getRecommendations(limit: 6);
});

final _commandResultsProvider = Provider<List<Map<String, dynamic>>>((ref) {
  final search = ref.watch(_searchProvider);
  if (search.isEmpty) return [];
  final palette = ref.read(_commandPaletteServiceProvider);
  return palette.search(search);
});

final _recentPapersProvider = Provider<List<CaiePaper>>((ref) {
  ref.watch(_recentIdsProvider);
  final all = ref.watch(_allPapersProvider);
  final ids = ref.watch(_recentIdsProvider).toSet();
  return all.where((p) => ids.contains(p.id)).toList();
});

// ── Unified CAIE Screen ────────────────────────────────────────────

class UnifiedCaieScreen extends ConsumerStatefulWidget {
  const UnifiedCaieScreen({super.key});
  @override
  ConsumerState<UnifiedCaieScreen> createState() => _UnifiedCaieScreenState();
}

class _UnifiedCaieScreenState extends ConsumerState<UnifiedCaieScreen>
    with TickerProviderStateMixin {
  final _searchFocusNode = FocusNode();
  final _searchController = TextEditingController();
  late AnimationController _paletteSlideController;
  late Animation<Offset> _paletteSlideAnimation;

  @override
  void initState() {
    super.initState();
    _paletteSlideController = AnimationController(
      vsync: this,
      duration: 300.ms,
    );
    _paletteSlideAnimation = Tween<Offset>(
      begin: const Offset(0, -0.05),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _paletteSlideController,
      curve: Curves.easeOutCubic,
    ));
    _loadPersistedState();
    _searchFocusNode.addListener(_onSearchFocus);
  }

  @override
  void dispose() {
    _searchFocusNode.dispose();
    _searchController.dispose();
    _paletteSlideController.dispose();
    super.dispose();
  }

  Future<void> _loadPersistedState() async {
    final repo = ref.read(caiePaperRepositoryProvider);
    final papers = await repo.fetchPapers();
    if (!mounted) return;
    ref.read(_papersCacheProvider.notifier).state = papers;
    final bookmarkIds = await repo.fetchBookmarkedIds();
    if (!mounted) return;
    ref.read(_bookmarkIdsProvider.notifier).state = bookmarkIds;
    final recents = await repo.fetchRecentPapers(limit: 10);
    if (!mounted) return;
    ref.read(_recentIdsProvider.notifier).state = recents.map((p) => p.id).toList();
    ref.read(_seededProvider.notifier).state = true;
  }

  void _onSearchFocus() {
    if (_searchFocusNode.hasFocus) {
      ref.read(_commandPaletteOpenProvider.notifier).state = true;
      _paletteSlideController.forward();
    }
  }

  void _openCommandPalette() {
    ref.read(_commandPaletteOpenProvider.notifier).state = true;
    _searchFocusNode.requestFocus();
    _paletteSlideController.forward();
  }

  void _closeCommandPalette() {
    _searchFocusNode.unfocus();
    ref.read(_commandPaletteOpenProvider.notifier).state = false;
    _paletteSlideController.reverse();
  }

  void _onPaperTap(CaiePaper paper) async {
    final repo = ref.read(caiePaperRepositoryProvider);
    await repo.recordOpen(paper.id);
    if (!mounted) return;
    final ids = ref.read(_recentIdsProvider);
    ref.read(_recentIdsProvider.notifier).state = [
      paper.id,
      ...ids.where((id) => id != paper.id).take(29),
    ];
  }

  void _toggleBookmark(String id, bool current) async {
    final repo = ref.read(caiePaperRepositoryProvider);
    await repo.toggleBookmark(id);
    if (!mounted) return;
    final ids = Set<String>.from(ref.read(_bookmarkIdsProvider));
    if (current) {
      ids.remove(id);
    } else {
      ids.add(id);
    }
    ref.read(_bookmarkIdsProvider.notifier).state = ids;
  }

  _ResponsiveLayout _layout(double width) {
    if (width >= 900) return _ResponsiveLayout.desktop;
    if (width >= 600) return _ResponsiveLayout.tablet;
    return _ResponsiveLayout.mobile;
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      autofocus: true,
      onKeyEvent: (node, event) {
        if (event is KeyDownEvent) {
          if (event.logicalKey == LogicalKeyboardKey.escape) {
            if (ref.read(_commandPaletteOpenProvider)) {
              _closeCommandPalette();
              return KeyEventResult.handled;
            }
          }
          if (event.logicalKey == LogicalKeyboardKey.slash) {
            _openCommandPalette();
            return KeyEventResult.handled;
          }
        }
        return KeyEventResult.ignored;
      },
      child: Scaffold(
        backgroundColor: const Color(0xFF0B1020),
        body: LayoutBuilder(
          builder: (context, constraints) {
            final layout = _layout(constraints.maxWidth);
            return SafeArea(
              child: Stack(
                children: [
                  _buildBody(layout, constraints),
                  if (ref.watch(_commandPaletteOpenProvider))
                    _buildCommandPalette(),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _buildBody(_ResponsiveLayout layout, BoxConstraints constraints) {
    final isDesktop = layout == _ResponsiveLayout.desktop;
    return Row(
      children: [
        if (isDesktop)
          _buildSidebar(constraints),
        Expanded(
          child: _buildMainContent(layout, constraints),
        ),
      ],
    );
  }

  Widget _buildSidebar(BoxConstraints constraints) {
    return Container(
      width: 260,
      height: constraints.maxHeight,
      decoration: const BoxDecoration(
        color: Color(0xFF12182B),
        border: Border(right: BorderSide(color: Colors.white10)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 24),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Text('CAIE Papers',
              style: GoogleFonts.orbitron(
                fontSize: 22, fontWeight: FontWeight.w900, color: Colors.white,
              ),
            ),
          ),
          const SizedBox(height: 24),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: _buildSmartCollections(),
          ),
          const SizedBox(height: 20),
          Expanded(
            child: ListView(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              children: [
                _SidebarItem(icon: Icons.explore, label: 'Browse All',
                  selected: ref.watch(_studyModeProvider) == _StudyMode.browse,
                  onTap: () => ref.read(_studyModeProvider.notifier).state = _StudyMode.browse),
                _SidebarItem(icon: Icons.close, label: 'Unsolved',
                  selected: ref.watch(_studyModeProvider) == _StudyMode.unsolved,
                  onTap: () => ref.read(_studyModeProvider.notifier).state = _StudyMode.unsolved),
                _SidebarItem(icon: Icons.psychology, label: 'Weak Topics',
                  selected: ref.watch(_studyModeProvider) == _StudyMode.weakTopics,
                  onTap: () => ref.read(_studyModeProvider.notifier).state = _StudyMode.weakTopics),
                _SidebarItem(icon: Icons.timer, label: 'Timed Practice',
                  selected: ref.watch(_studyModeProvider) == _StudyMode.timedPractice,
                  onTap: () => ref.read(_studyModeProvider.notifier).state = _StudyMode.timedPractice),
                const Divider(color: Colors.white10, height: 32),
                const Padding(
                  padding: EdgeInsets.only(left: 12, bottom: 8),
                  child: Text('Topics', style: TextStyle(color: Colors.white38, fontSize: 12)),
                ),
                _SidebarItem(icon: Icons.bolt, label: 'Electricity', onTap: () {}),
                _SidebarItem(icon: Icons.waves, label: 'Waves', onTap: () {}),
                _SidebarItem(icon: Icons.speed, label: 'Momentum', onTap: () {}),
                _SidebarItem(icon: Icons.thermostat, label: 'Thermal', onTap: () {}),
              ],
            ),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 400.ms).slideX(begin: -0.05);
  }

  Widget _buildMainContent(_ResponsiveLayout layout, BoxConstraints constraints) {
    final isDesktop = layout == _ResponsiveLayout.desktop;
    final pad = isDesktop ? 24.0 : 16.0;

    return SingleChildScrollView(
      padding: EdgeInsets.all(pad),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isDesktop) _buildHeader(layout),
          if (!isDesktop) const SizedBox(height: 16),
          _buildSearchBar(),
          const SizedBox(height: 20),
          if (isDesktop) _buildSmartCollections(),
          if (isDesktop) const SizedBox(height: 20),
          _buildRecentPapers(),
          const SizedBox(height: 28),
          _buildStudyModeChips(),
          const SizedBox(height: 24),
          _buildFilterSection(layout),
          const SizedBox(height: 24),
          if (ref.watch(_studyModeProvider) == _StudyMode.weakTopics)
            _buildWeakTopics()
          else ...[
            _buildRecommendations(),
            const SizedBox(height: 24),
            _buildPaperFeed(layout),
          ],
        ],
      ),
    ).animate().fadeIn(duration: 400.ms, delay: 100.ms);
  }

  Widget _buildHeader(_ResponsiveLayout layout) {
    return Row(
      children: [
        Expanded(child: Text('CAIE Papers',
          style: GoogleFonts.orbitron(
            fontSize: 28, fontWeight: FontWeight.w900, color: Colors.white, letterSpacing: 1.2,
          ),
        )),
        IconButton(
          icon: const Icon(Icons.help_outline, color: Colors.white38),
          onPressed: () => _openCommandPalette(),
        ),
      ],
    );
  }

  Widget _buildSearchBar() {
    final isOpen = ref.watch(_commandPaletteOpenProvider);
    return AnimatedContainer(
      duration: 300.ms,
      curve: Curves.easeOutCubic,
      height: 52,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(isOpen ? 18 : 14),
        color: isOpen ? const Color(0xFF1A2240) : const Color(0xFF1A1A2E),
        border: Border.all(
          color: isOpen ? const Color(0xFF6C63FF).withOpacity(0.6) : Colors.white12,
        ),
      ),
      child: TextField(
        controller: _searchController,
        focusNode: _searchFocusNode,
        onChanged: (v) => ref.read(_searchProvider.notifier).state = v,
        style: const TextStyle(color: Colors.white, fontSize: 16),
        decoration: InputDecoration(
          hintText: 'Search 0625 P42 2024 or type / for commands...',
          hintStyle: const TextStyle(color: Colors.white24),
          prefixIcon: Icon(Icons.search,
            color: isOpen ? const Color(0xFF6C63FF) : Colors.white38,
          ),
          suffixIcon: ref.watch(_searchProvider).isNotEmpty
              ? IconButton(
                  icon: const Icon(Icons.close, color: Colors.white38, size: 18),
                  onPressed: () { _searchController.clear(); ref.read(_searchProvider.notifier).state = ''; },
                )
              : IconButton(
                  icon: const Icon(Icons.keyboard, color: Colors.white24, size: 18),
                  onPressed: _openCommandPalette,
                ),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 14),
        ),
      ),
    ).animate(target: isOpen ? 1 : 0).shimmer(
      duration: 1500.ms,
      color: const Color(0xFF6C63FF).withOpacity(0.08),
      delay: 500.ms,
    );
  }

  Widget _buildSmartCollections() {
    final collections = [
      ('🔥 Latest', Icons.trending_up),
      ('⭐ Saved', Icons.star),
      ('⬇ Downloaded', Icons.download_done),
      ('📈 Hardest', Icons.local_fire_department),
      ('🧠 Recommended', Icons.auto_awesome),
      ('❌ Unsolved', Icons.close),
    ];
    return SizedBox(
      height: 44,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: collections.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final (label, icon) = collections[i];
          return Container(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(14),
              color: Colors.white.withOpacity(0.06),
              border: Border.all(color: Colors.white.withOpacity(0.08)),
            ),
            child: Center(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(icon, size: 16, color: Colors.white60),
                  const SizedBox(width: 6),
                  Text(label, style: const TextStyle(color: Colors.white70, fontSize: 14)),
                ],
              ),
            ),
          ).animate().fadeIn(duration: 300.ms, delay: (i * 80).ms).slideX(begin: 0.1);
        },
      ),
    );
  }

  Widget _buildRecentPapers() {
    final papers = ref.watch(_recentPapersProvider);
    if (papers.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _SectionLabel('Recent Papers'),
        const SizedBox(height: 12),
        SizedBox(
          height: 100,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: papers.take(8).length,
            separatorBuilder: (_, __) => const SizedBox(width: 12),
            itemBuilder: (_, i) {
              final p = papers[i];
              return _RecentPaperCard(paper: p, onTap: () => _onPaperTap(p));
            },
          ),
        ),
      ],
    );
  }

  Widget _buildStudyModeChips() {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: _StudyMode.values.map((mode) {
          final selected = ref.watch(_studyModeProvider) == mode;
          return Padding(
            padding: const EdgeInsets.only(right: 8),
            child: ChoiceChip(
              label: Text(_modeLabel(mode)),
              selected: selected,
              selectedColor: const Color(0xFF6C63FF),
              backgroundColor: Colors.white.withOpacity(0.06),
              labelStyle: TextStyle(
                color: selected ? Colors.white : Colors.white60,
                fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
              ),
              side: BorderSide(color: selected ? const Color(0xFF6C63FF) : Colors.white.withOpacity(0.1)),
              onSelected: (_) => ref.read(_studyModeProvider.notifier).state = mode,
            ),
          ).animate().fadeIn(duration: 200.ms);
        }).toList(),
      ),
    );
  }

  Widget _buildFilterSection(_ResponsiveLayout layout) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        color: Colors.white.withOpacity(0.03),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _SectionLabel('Filters'),
          const SizedBox(height: 12),
          _buildFilterChipRow(
            label: 'Subject',
            items: ['Physics', 'Chemistry', 'Biology', 'Mathematics', 'Computer Science'],
            provider: _selectedSubjectProvider,
          ),
          const SizedBox(height: 12),
          _buildFilterChipRow(
            label: 'Year',
            items: [2026, 2025, 2024, 2023, 2022, 2021, 2020],
            provider: _selectedYearProvider,
            display: (y) => y.toString(),
            clearLabel: 'All Years',
          ),
          const SizedBox(height: 12),
          _buildFilterChipRow(
            label: 'Session',
            items: const ['MJ', 'ON'],
            provider: _selectedSessionProvider,
            clearLabel: 'All',
          ),
          const SizedBox(height: 12),
          _buildVariantChips(),
        ],
      ),
    ).animate().fadeIn(duration: 300.ms).slideY(begin: 0.05);
  }

  Widget _buildFilterChipRow<T>({
    required String label,
    required List<T> items,
    required dynamic provider,
    String Function(T)? display,
    String clearLabel = 'All',
  }) {
    final selected = ref.watch(provider);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white54, fontSize: 12, fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        SizedBox(
          height: 38,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: items.length + 1,
            separatorBuilder: (_, __) => const SizedBox(width: 6),
            itemBuilder: (_, i) {
              final isAll = i == 0;
              final isSelected = isAll ? selected == null : selected == items[i - 1];
              final labelText = isAll ? clearLabel : (display != null ? display(items[i - 1]!) : '${items[i - 1]}');
              return ChoiceChip(
                label: Text(labelText, style: const TextStyle(fontSize: 13)),
                selected: isSelected,
                selectedColor: const Color(0xFF6C63FF),
                backgroundColor: Colors.white.withOpacity(0.05),
                labelStyle: TextStyle(color: isSelected ? Colors.white : Colors.white54, fontSize: 13),
                side: BorderSide.none,
                onSelected: (_) => ref.read(provider.notifier).state = isAll ? null : items[i - 1],
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildVariantChips() {
    const variants = ['All', 'P11', 'P12', 'P13', 'P21', 'P22', 'P23', 'P41', 'P42', 'P43'];
    final selected = ref.watch(_selectedVariantProvider);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Variant', style: TextStyle(color: Colors.white54, fontSize: 12, fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        Wrap(
          spacing: 6, runSpacing: 6,
          children: variants.map((v) {
            final isSelected = selected == v;
            return ChoiceChip(
              label: Text(v, style: const TextStyle(fontSize: 13)),
              selected: isSelected,
              selectedColor: const Color(0xFF6C63FF),
              backgroundColor: Colors.white.withOpacity(0.05),
              labelStyle: TextStyle(color: isSelected ? Colors.white : Colors.white60, fontSize: 13),
              side: BorderSide.none,
              onSelected: (_) => ref.read(_selectedVariantProvider.notifier).state = v,
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildRecommendations() {
    final recommendations = ref.watch(_recommendationsProvider);
    if (recommendations.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _SectionLabel('Recommended'),
        const SizedBox(height: 12),
        ...recommendations.map((r) => Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                gradient: LinearGradient(
                  colors: [const Color(0xFF6C63FF).withOpacity(0.15), const Color(0xFF6C63FF).withOpacity(0.05)],
                  begin: Alignment.topLeft, end: Alignment.bottomRight,
                ),
                border: Border.all(color: const Color(0xFF6C63FF).withOpacity(0.2)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.auto_awesome, color: Color(0xFF6C63FF), size: 20),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(r.displayName, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                        const SizedBox(height: 2),
                        Text(r.reason, style: const TextStyle(color: Colors.white54, fontSize: 13)),
                      ],
                    ),
                  ),
                  const Icon(Icons.arrow_forward_ios, color: Colors.white24, size: 14),
                ],
              ),
            ).animate().fadeIn(duration: 300.ms).slideX(begin: 0.05)),
      ],
    );
  }

  Widget _buildWeakTopics() {
    final topics = ref.watch(_weakTopicsProvider);
    if (topics.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _SectionLabel('Weak Topics — Focus Here'),
        const SizedBox(height: 16),
        ...topics.map((t) {
          final pct = (t.accuracy * 100).round();
          final color = pct < 50 ? Colors.redAccent : (pct < 60 ? Colors.orangeAccent : Colors.amberAccent);
          return Container(
            margin: const EdgeInsets.only(bottom: 10),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              color: Colors.white.withOpacity(0.04),
              border: Border.all(color: color.withOpacity(0.2)),
            ),
            child: Row(
              children: [
                SizedBox(width: 48, height: 48,
                  child: Stack(alignment: Alignment.center, children: [
                    SizedBox(width: 44, height: 44,
                      child: TweenAnimationBuilder<double>(
                        tween: Tween(begin: 0, end: t.accuracy),
                        duration: 1.seconds, curve: Curves.easeOutCubic,
                        builder: (_, value, __) => CircularProgressIndicator(
                          value: value, strokeWidth: 4,
                          backgroundColor: Colors.white10, color: color,
                        ),
                      ),
                    ),
                    Text('$pct%', style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.bold)),
                  ]),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(t.topicName, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 16)),
                      Text('${t.attempted} attempts • ${t.correct} correct',
                        style: const TextStyle(color: Colors.white38, fontSize: 12)),
                    ],
                  ),
                ),
                Icon(Icons.arrow_forward_ios, color: Colors.white24, size: 14),
              ],
            ),
          ).animate().fadeIn(duration: 300.ms).slideX(begin: 0.05);
        }),
      ],
    );
  }

  Widget _buildPaperFeed(_ResponsiveLayout layout) {
    final papers = ref.watch(_papersProvider);
    if (papers.isEmpty) {
      return SizedBox(
        height: 200,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.search_off, color: Colors.white24, size: 48),
              const SizedBox(height: 12),
              const Text('No papers match your filters', style: TextStyle(color: Colors.white38)),
            ],
          ),
        ),
      );
    }
    final crossAxisCount = layout == _ResponsiveLayout.mobile ? 2 : (layout == _ResponsiveLayout.tablet ? 3 : 4);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const _SectionLabel('Papers'),
            const Spacer(),
            Text('${papers.length} found', style: const TextStyle(color: Colors.white38, fontSize: 13)),
          ],
        ),
        const SizedBox(height: 12),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: papers.length,
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            crossAxisSpacing: 14,
            mainAxisSpacing: 14,
            childAspectRatio: 0.78,
          ),
          itemBuilder: (_, i) => _PaperCard(
            paper: papers[i], index: i,
            onTap: () => _onPaperTap(papers[i]),
            onBookmark: () => _toggleBookmark(papers[i].id, papers[i].bookmarked),
          ),
        ),
      ],
    );
  }

  Widget _buildCommandPalette() {
    final results = ref.watch(_commandResultsProvider);
    return FadeTransition(
      opacity: _paletteSlideController,
      child: SlideTransition(
        position: _paletteSlideAnimation,
        child: Material(
          color: Colors.transparent,
          child: GestureDetector(
            onTap: _closeCommandPalette,
            child: Container(
              color: Colors.black.withOpacity(0.7),
              child: Center(
                child: Container(
                  width: 520,
                  constraints: const BoxConstraints(maxHeight: 400),
                  margin: const EdgeInsets.only(top: 100),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(24),
                    color: const Color(0xFF1A2240),
                    border: Border.all(color: const Color(0xFF6C63FF).withOpacity(0.3)),
                    boxShadow: [BoxShadow(
                      color: const Color(0xFF6C63FF).withOpacity(0.15),
                      blurRadius: 40, spreadRadius: 5,
                    )],
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: const BoxDecoration(
                          border: Border(bottom: BorderSide(color: Colors.white10)),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.terminal, color: Color(0xFF6C63FF), size: 20),
                            const SizedBox(width: 10),
                            const Text('Command Palette',
                              style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                            const Spacer(),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8),
                                color: Colors.white.withOpacity(0.1),
                              ),
                              child: const Text('ESC', style: TextStyle(color: Colors.white38, fontSize: 12)),
                            ),
                            const SizedBox(width: 8),
                            IconButton(
                              icon: const Icon(Icons.close, color: Colors.white38, size: 20),
                              onPressed: _closeCommandPalette,
                            ),
                          ],
                        ),
                      ),
                      if (results.isEmpty && ref.watch(_searchProvider).isNotEmpty)
                        const Padding(
                          padding: EdgeInsets.all(32),
                          child: Text('No results found', style: TextStyle(color: Colors.white38)),
                        )
                      else if (ref.watch(_searchProvider).isEmpty)
                        Padding(
                          padding: const EdgeInsets.all(16),
                          child: Column(
                            children: [
                              _PaletteHint(icon: Icons.search, label: 'Type to search papers', hint: '0625, Physics, P42...'),
                              const SizedBox(height: 8),
                              _PaletteHint(icon: Icons.download_done, label: '/downloaded', hint: 'Show downloaded'),
                              _PaletteHint(icon: Icons.star, label: '/bookmarks', hint: 'Show bookmarked'),
                              _PaletteHint(icon: Icons.close, label: '/unsolved', hint: 'Show unsolved'),
                              _PaletteHint(icon: Icons.psychology, label: '/weak', hint: 'Show weak topics'),
                            ],
                          ),
                        ),
                      Flexible(
                        child: ListView.builder(
                          shrinkWrap: true,
                          itemCount: results.length,
                          itemBuilder: (_, i) {
                            final r = results[i];
                            return ListTile(
                              leading: Icon(
                                r['type'] == 'command' ? Icons.terminal : Icons.description,
                                color: Colors.white54, size: 18,
                              ),
                              title: Text('${r['label']}', style: const TextStyle(color: Colors.white, fontSize: 14)),
                              subtitle: r['subtitle'] != null
                                  ? Text('${r['subtitle']}', style: const TextStyle(color: Colors.white38, fontSize: 12))
                                  : null,
                              onTap: () {
                                ref.read(_searchProvider.notifier).state = '${r['label']}';
                                _closeCommandPalette();
                              },
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  String _modeLabel(_StudyMode mode) {
    switch (mode) {
      case _StudyMode.browse: return 'Browse All';
      case _StudyMode.unsolved: return 'Unsolved';
      case _StudyMode.weakTopics: return 'Weak Topics';
      case _StudyMode.timedPractice: return 'Timed Practice';
    }
  }
}

// ── Sub-widgets ─────────────────────────────────────────────────────

class _SidebarItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _SidebarItem({required this.icon, required this.label, this.selected = false, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Material(
        color: selected ? const Color(0xFF6C63FF).withOpacity(0.15) : Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            child: Row(
              children: [
                Icon(icon, size: 18, color: selected ? const Color(0xFF6C63FF) : Colors.white38),
                const SizedBox(width: 12),
                Text(label, style: TextStyle(
                  color: selected ? Colors.white : Colors.white60,
                  fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
                  fontSize: 14,
                )),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);
  @override
  Widget build(BuildContext context) {
    return Text(text, style: const TextStyle(
      fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white, letterSpacing: 0.3,
    ));
  }
}

class _RecentPaperCard extends StatelessWidget {
  final CaiePaper paper;
  final VoidCallback onTap;
  const _RecentPaperCard({required this.paper, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 180,
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(18),
          color: Colors.white.withOpacity(0.05),
          border: Border.all(color: Colors.white.withOpacity(0.08)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(8),
                    color: const Color(0xFF6C63FF).withOpacity(0.2),
                  ),
                  child: Text(paper.variant, style: const TextStyle(color: Color(0xFF6C63FF), fontSize: 11)),
                ),
                const Spacer(),
                const Icon(Icons.history, color: Colors.white24, size: 14),
              ],
            ),
            const SizedBox(height: 10),
            Text(paper.subject, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 15)),
            const SizedBox(height: 2),
            Text('${paper.session} ${paper.year}', style: const TextStyle(color: Colors.white38, fontSize: 13)),
          ],
        ),
      ).animate().fadeIn(duration: 300.ms).slideX(begin: 0.05),
    );
  }
}

class _PaperCard extends StatelessWidget {
  final CaiePaper paper;
  final int index;
  final VoidCallback onTap;
  final VoidCallback onBookmark;
  const _PaperCard({
    required this.paper, required this.index,
    required this.onTap, required this.onBookmark,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          color: Colors.white.withOpacity(0.04),
          border: Border.all(color: Colors.white.withOpacity(0.06)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(10),
                    color: const Color(0xFF6C63FF).withOpacity(0.2),
                  ),
                  child: Text(paper.variant,
                    style: const TextStyle(color: Color(0xFF6C63FF), fontSize: 11, fontWeight: FontWeight.w600),
                  ),
                ),
                const Spacer(),
                GestureDetector(
                  onTap: onBookmark,
                  child: AnimatedSwitcher(
                    duration: 200.ms,
                    transitionBuilder: (child, anim) => ScaleTransition(scale: anim, child: child),
                    child: Icon(
                      paper.bookmarked ? Icons.star : Icons.star_border,
                      key: ValueKey(paper.bookmarked),
                      color: paper.bookmarked ? Colors.amberAccent : Colors.white24,
                      size: 20,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(paper.subject, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600, fontSize: 15)),
            const SizedBox(height: 2),
            Text('${paper.subjectCode} • ${paper.session} ${paper.year}',
              style: const TextStyle(color: Colors.white38, fontSize: 12)),
            const SizedBox(height: 10),
            Row(
              children: [
                _StatBadge(icon: Icons.local_fire_department, label: '${paper.difficulty.toInt()}/10', color: Colors.orangeAccent),
                const SizedBox(width: 8),
                _StatBadge(icon: Icons.analytics, label: '${paper.accuracy.toInt()}%', color: Colors.cyanAccent),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(paper.downloaded ? Icons.download_done : Icons.cloud_outlined, size: 14, color: Colors.white24),
                const SizedBox(width: 4),
                Text(paper.downloaded ? 'Offline' : 'Online', style: const TextStyle(color: Colors.white24, fontSize: 11)),
                const Spacer(),
                if (paper.solved) const Icon(Icons.check_circle, color: Colors.greenAccent, size: 14),
              ],
            ),
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: onTap,
                style: FilledButton.styleFrom(
                  backgroundColor: const Color(0xFF6C63FF),
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Open', style: TextStyle(fontSize: 13)),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 300.ms, delay: (index * 30).ms).slideY(begin: 0.05);
  }
}

class _StatBadge extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  const _StatBadge({required this.icon, required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: color),
        const SizedBox(width: 3),
        Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w500)),
      ],
    );
  }
}

class _PaletteHint extends StatelessWidget {
  final IconData icon;
  final String label;
  final String hint;
  const _PaletteHint({required this.icon, required this.label, required this.hint});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        color: Colors.white.withOpacity(0.04),
      ),
      child: Row(
        children: [
          Icon(icon, color: Colors.white38, size: 18),
          const SizedBox(width: 10),
          Text(label, style: const TextStyle(color: Colors.white, fontSize: 14)),
          const Spacer(),
          Text(hint, style: const TextStyle(color: Colors.white24, fontSize: 12)),
        ],
      ),
    );
  }
}

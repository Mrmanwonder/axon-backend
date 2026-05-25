// lib/screens/auth/subject_selection_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:go_router/go_router.dart';

import '../../services/app_state.dart';
import '../../services/curriculum_catalog_service.dart';
import '../../theme/app_theme.dart';
import '../../theme/subject_themes.dart';
import '../../widgets/common/auth_frame.dart';
import '../../widgets/common/rose_loader.dart';

class SubjectSelectionScreen extends ConsumerStatefulWidget {
  final String board;

  const SubjectSelectionScreen({super.key, required this.board});

  @override
  ConsumerState<SubjectSelectionScreen> createState() =>
      _SubjectSelectionScreenState();
}

class _SubjectSelectionScreenState
    extends ConsumerState<SubjectSelectionScreen> {
  late Set<String> _selectedSubjects;
  late List<String> _subjects;
  Map<String, int> _chapterCounts = const {};
  final TextEditingController _searchController = TextEditingController();
  String _searchQuery = '';
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _subjects = const [];
    _selectedSubjects = {'Mathematics'};
    _loadSubjects();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      final savedSubjects =
          ref.read(authStateProvider).user?.subjects ?? const [];
      debugPrint('SubjectSelectionScreen: saved subjects = $savedSubjects');
      if (mounted) {
        setState(() {
          _selectedSubjects = savedSubjects.isEmpty
              ? <String>{'Mathematics'}
              : savedSubjects.toSet();
        });
      }
    });
  }

  Future<void> _loadSubjects() async {
    final subjects =
        await CurriculumCatalogService.instance.subjectsForBoard(widget.board);
    final counts = <String, int>{};
    for (final subject in subjects) {
      counts[subject] = await CurriculumCatalogService.instance.chapterCount(
        board: widget.board,
        subject: subject,
      );
    }
    if (!mounted) return;
    setState(() {
      _subjects = subjects;
      _chapterCounts = counts;
      _loading = false;
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  List<String> get _filteredSubjects {
    if (_searchQuery.isEmpty) return _subjects;
    return _subjects
        .where((s) => s.toLowerCase().contains(_searchQuery.toLowerCase()))
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    return AuthFrame(
      eyebrow: 'STEP 2 OF 3',
      title: 'Choose your subjects.',
      subtitle:
          'Pick the subjects Axon should prioritize across study, mocks, and resources.',
      onBack: () => context.go('/auth/board'),
      floatingButton: _FloatingDoneButton(
        count: _selectedSubjects.length,
        enabled: _selectedSubjects.isNotEmpty,
        onTap: () => context.go('/auth/motivation', extra: {
          'board': widget.board,
          'subjects': _selectedSubjects.toList(),
        }),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Container(
              height: 48,
              decoration: BoxDecoration(
                color: AxonColors.surface,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AxonColors.divider),
              ),
              child: Row(
                children: [
                  const SizedBox(width: 14),
                  Icon(
                    Icons.search_rounded,
                    color: AxonColors.textSecondary,
                    size: 20,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: TextField(
                      controller: _searchController,
                      onChanged: (v) => setState(() => _searchQuery = v),
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 15,
                      ),
                      decoration: InputDecoration(
                        hintText: 'Search subjects...',
                        hintStyle: GoogleFonts.googleSans(
                          color: AxonColors.textSecondary,
                          fontSize: 15,
                        ),
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.zero,
                      ),
                    ),
                  ),
                  if (_searchQuery.isNotEmpty)
                    GestureDetector(
                      onTap: () {
                        _searchController.clear();
                        setState(() => _searchQuery = '');
                      },
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Icon(
                          Icons.close_rounded,
                          color: AxonColors.textSecondary,
                          size: 18,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: _loading
                  ? const Center(child: RoseLoader(size: 24))
                  : ListView.builder(
                      itemCount: _filteredSubjects.length,
                      itemBuilder: (context, index) {
                        final subject = _filteredSubjects[index];
                        final selected = _selectedSubjects.contains(subject);
                        final theme = themeFor(subject);
                        final chapters = _chapterCounts[subject] ?? 0;

                        return GestureDetector(
                          onTap: () {
                            HapticFeedback.selectionClick();
                            setState(() {
                              if (selected) {
                                _selectedSubjects.remove(subject);
                              } else {
                                _selectedSubjects.add(subject);
                              }
                            });
                          },
                          child: Container(
                            margin: const EdgeInsets.only(bottom: 10),
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: selected
                                  ? theme.primaryColor.withValues(alpha: 0.12)
                                  : AxonColors.surface,
                              borderRadius: BorderRadius.circular(14),
                              border: Border.all(
                                color: selected
                                    ? theme.primaryColor
                                    : AxonColors.divider,
                                width: selected ? 1.5 : 1,
                              ),
                              boxShadow: selected
                                  ? [
                                      BoxShadow(
                                        color: theme.primaryColor
                                            .withValues(alpha: 0.15),
                                        blurRadius: 8,
                                        offset: const Offset(0, 2),
                                      )
                                    ]
                                  : null,
                            ),
                            child: Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(10),
                                  decoration: BoxDecoration(
                                    color: theme.primaryColor
                                        .withValues(alpha: 0.1),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: Icon(
                                    theme.icon,
                                    color: theme.primaryColor,
                                    size: 22,
                                  ),
                                ),
                                const SizedBox(width: 14),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        subject,
                                        style: GoogleFonts.googleSans(
                                          color: selected
                                              ? theme.primaryColor
                                              : AxonColors.textPrimary,
                                          fontSize: 15,
                                          fontWeight: FontWeight.w600,
                                        ),
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        '$chapters chapters',
                                        style: GoogleFonts.googleSans(
                                          color: AxonColors.textTertiary,
                                          fontSize: 12,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          )
                              .animate(
                                delay: Duration(milliseconds: 30 * index),
                              )
                              .fadeIn(duration: 250.ms)
                              .slideX(begin: -0.05, end: 0),
                        );
                      },
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

class _FloatingDoneButton extends StatelessWidget {
  final int count;
  final bool enabled;
  final VoidCallback onTap;

  const _FloatingDoneButton({
    required this.count,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: enabled ? onTap : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        decoration: BoxDecoration(
          color: enabled ? AxonColors.accent : AxonColors.divider,
          borderRadius: BorderRadius.circular(28),
          boxShadow: enabled
              ? [
                  BoxShadow(
                    color: AxonColors.accent.withValues(alpha: 0.4),
                    blurRadius: 16,
                    offset: const Offset(0, 4),
                  ),
                ]
              : null,
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              enabled ? 'Done' : 'Select subjects',
              style: GoogleFonts.googleSans(
                color: Colors.white,
                fontWeight: FontWeight.w600,
                fontSize: 14,
              ),
            ),
            if (enabled && count > 0) ...[
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  count.toString(),
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
            const SizedBox(width: 8),
            Icon(
              Icons.arrow_forward_rounded,
              color: Colors.white,
              size: 18,
            ),
          ],
        ),
      ),
    );
  }
}

// lib/screens/auth/board_selection_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:go_router/go_router.dart';

import '../../services/app_state.dart';
import '../../services/curriculum_catalog_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/common/auth_frame.dart';

class BoardSelectionScreen extends ConsumerStatefulWidget {
  final String? initialBoard;

  const BoardSelectionScreen({super.key, this.initialBoard});

  @override
  ConsumerState<BoardSelectionScreen> createState() =>
      _BoardSelectionScreenState();
}

class _BoardSelectionScreenState extends ConsumerState<BoardSelectionScreen> {
  late String _selectedBoard;

  @override
  void initState() {
    super.initState();
    final savedBoard = ref.read(authStateProvider).user?.board;
    final preferredBoard = widget.initialBoard ?? savedBoard;
    final supported = CurriculumCatalogService.instance.supportedBoards;
    final exact = supported.where((board) => board.label == preferredBoard);
    _selectedBoard =
        exact.isNotEmpty ? exact.first.label : supported.first.label;
  }

  @override
  Widget build(BuildContext context) {
    return AuthFrame(
      eyebrow: 'STEP 1 OF 3',
      title: 'Select your board.',
      subtitle: 'This tunes the exam structure, resources, and timeline.',
      onBack: () => context.go('/auth/login'),
      floatingButton: FloatingActionButton.extended(
        onPressed: () => context.go('/auth/subjects?board=$_selectedBoard'),
        backgroundColor: AxonColors.accent,
        icon: const Icon(Icons.arrow_forward, color: Colors.white),
        label: Text(
          'Continue',
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 8),
            ...CurriculumCatalogService.instance.supportedBoards.map((entry) {
              final selected = entry.label == _selectedBoard;
              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: GestureDetector(
                  onTap: () => setState(() => _selectedBoard = entry.label),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: selected
                          ? AxonColors.accent.withValues(alpha: 0.15)
                          : AxonColors.surface,
                      borderRadius: BorderRadius.circular(AxonRadius.md),
                      border: Border.all(
                        color:
                            selected ? AxonColors.accent : AxonColors.divider,
                        width: selected ? 1.5 : 1,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                entry.label,
                                style: GoogleFonts.googleSans(
                                  color: selected
                                      ? AxonColors.accent
                                      : AxonColors.textPrimary,
                                  fontWeight: FontWeight.w600,
                                  fontSize: 16,
                                ),
                              ),
                            ),
                            if (selected)
                              Icon(
                                Icons.check_circle,
                                color: AxonColors.accent,
                                size: 20,
                              ),
                          ],
                        ),
                        if (entry.description.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Text(
                            entry.description,
                            style: GoogleFonts.googleSans(
                              color: AxonColors.textSecondary,
                              fontSize: 12,
                            ),
                          ),
                        ],
                        if (entry.regions.isNotEmpty) ...[
                          const SizedBox(height: 4),
                          Row(
                            children: [
                              Icon(
                                Icons.public,
                                size: 12,
                                color: AxonColors.textTertiary,
                              ),
                              const SizedBox(width: 4),
                              Text(
                                entry.regions,
                                style: GoogleFonts.googleSans(
                                  color: AxonColors.textTertiary,
                                  fontSize: 11,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              );
            }),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

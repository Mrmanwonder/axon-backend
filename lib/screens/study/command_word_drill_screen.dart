import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../services/app_state.dart';
import '../../models/command_word_drill_models.dart';
import '../../services/command_word_drill_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/common/rose_loader.dart';

class CommandWordDrillScreen extends ConsumerStatefulWidget {
  const CommandWordDrillScreen({
    super.key,
    required this.objectiveId,
  });

  final String objectiveId;

  @override
  ConsumerState<CommandWordDrillScreen> createState() => _CommandWordDrillScreenState();
}

class _CommandWordDrillScreenState extends ConsumerState<CommandWordDrillScreen> {
  final _service = CommandWordDrillService();
  final _pageController = PageController(viewportFraction: 0.92);
  final _responses = <String, TextEditingController>{};
  CommandWordDrillBundle? _bundle;
  CommandWordDrillEvaluation? _evaluation;
  bool _loading = true;
  bool _submitting = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(navbarVisibleProvider.notifier).state = false;
    });
    _load();
  }

  @override
  void dispose() {
    ref.read(navbarVisibleProvider.notifier).state = true;
    _pageController.dispose();
    for (final controller in _responses.values) {
      controller.dispose();
    }
    super.dispose();
  }

  Future<void> _load() async {
    final bundle =
        await _service.generateDrill(objectiveId: widget.objectiveId);
    for (final card in bundle.cards) {
      _responses.putIfAbsent(card.commandWord, () => TextEditingController());
    }
    if (!mounted) return;
    setState(() {
      _bundle = bundle;
      _loading = false;
    });
  }

  Future<void> _submit() async {
    final bundle = _bundle;
    if (bundle == null) return;
    setState(() => _submitting = true);
    try {
      final evaluation = await _service.evaluateDrill(
        objectiveId: bundle.objectiveId,
        cards: bundle.cards,
        responses: {
          for (final entry in _responses.entries)
            entry.key: entry.value.text.trim(),
        },
      );
      if (!mounted) return;
      setState(() => _evaluation = evaluation);
    } finally {
      if (mounted) {
        setState(() => _submitting = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        body: Center(child: RoseLoader(size: 24)),
      );
    }

    final bundle = _bundle!;
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        title: Text(
          'Command-Word Drill',
          style: GoogleFonts.googleSans(fontWeight: FontWeight.w700),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              bundle.title,
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 22,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              '${bundle.subject} | ${bundle.paper} | ${bundle.topic}',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: PageView.builder(
                controller: _pageController,
                itemCount: bundle.cards.length,
                itemBuilder: (context, index) {
                  final card = bundle.cards[index];
                  return Padding(
                    padding: const EdgeInsets.only(right: 12),
                    child: _SwipePromptCard(
                      card: card,
                      controller: _responses[card.commandWord]!,
                    ),
                  );
                },
              ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _submitting ? null : _submit,
                child: Text(_submitting ? 'Evaluating…' : 'Grade Drill'),
              ),
            ),
            if (_evaluation != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AxonColors.surface,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AxonColors.divider),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Overall Score: ${_evaluation!.overallScore.toStringAsFixed(1)}',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _evaluation!.feedback,
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textSecondary,
                        height: 1.5,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _SwipePromptCard extends StatelessWidget {
  const _SwipePromptCard({
    required this.card,
    required this.controller,
  });

  final CommandWordDrillCard card;
  final TextEditingController controller;

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: AxonColors.electricCyan.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                card.commandWord.toUpperCase(),
                style: GoogleFonts.googleSans(
                  color: AxonColors.electricCyan,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
            const SizedBox(height: 18),
            Text(
              card.prompt,
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              card.depthExpectation,
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                height: 1.45,
              ),
            ),
            const SizedBox(height: 18),
            Expanded(
              child: TextField(
                controller: controller,
                maxLines: null,
                expands: true,
                decoration: const InputDecoration(
                  hintText: 'Write your answer here…',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.swipe_rounded, size: 16),
                const SizedBox(width: 8),
                Text(
                  'Swipe to move to the next command word.',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

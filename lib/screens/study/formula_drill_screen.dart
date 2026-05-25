import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../services/app_state.dart';
import '../../services/formula_drill_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/math/math_expression.dart';

class FormulaDrillScreen extends ConsumerStatefulWidget {
  final String title;
  final List<FormulaDrillCard> cards;
  final Future<void> Function(String cardId, bool remembered) onReview;

  const FormulaDrillScreen({
    super.key,
    required this.title,
    required this.cards,
    required this.onReview,
  });

  @override
  ConsumerState<FormulaDrillScreen> createState() => _FormulaDrillScreenState();
}

class _FormulaDrillScreenState extends ConsumerState<FormulaDrillScreen>
    with SingleTickerProviderStateMixin {
  late final List<FormulaDrillCard> _cards = List.of(widget.cards);
  int _index = 0;
  bool _reveal = false;
  int _remembered = 0;
  late AnimationController _flipController;
  late Animation<double> _flipAnimation;
  late final StateController<bool> _navbarNotifier;

  bool get _isComplete => _index >= _cards.length;

  @override
  void initState() {
    super.initState();
    _navbarNotifier = ref.read(navbarVisibleProvider.notifier);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _navbarNotifier.state = false;
    });
    _flipController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _flipAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _flipController, curve: Curves.easeInOutCubic),
    );
  }

  @override
  void dispose() {
    _navbarNotifier.state = true;
    _flipController.dispose();
    super.dispose();
  }

  void _toggleReveal() {
    if (_reveal) {
      _flipController.reverse();
    } else {
      _flipController.forward();
    }
    setState(() => _reveal = !_reveal);
  }

  void _resetCard() {
    _flipController.reset();
    setState(() => _reveal = false);
  }

  Future<void> _submit(bool remembered) async {
    final current = _cards[_index];
    await widget.onReview(current.id, remembered);
    if (!mounted) return;
    setState(() {
      if (remembered) _remembered++;
      _index++;
    });
    _resetCard();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AxonThemeMode.isDark;
    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF0D0D0D) : const Color(0xFFF5F5F5),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          onPressed: () => Navigator.of(context).pop(),
          icon: Icon(Icons.close, color: isDark ? Colors.white54 : Colors.black54),
        ),
        title: Text(
          widget.title,
          style: GoogleFonts.googleSans(
            color: isDark ? Colors.white : Colors.black,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: _isComplete ? _buildSummary() : _buildCard(),
      ),
    );
  }

  Widget _buildSummary() {
    final total = _cards.length;
    final isDark = AxonThemeMode.isDark;
    final textColor = isDark ? Colors.white : Colors.black;
    final mutedColor = isDark ? Colors.white70 : Colors.black54;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.bolt_rounded, color: AxonColors.accent, size: 56),
            const SizedBox(height: 16),
            Text(
              'Formula drill complete',
              style: GoogleFonts.googleSans(
                color: textColor,
                fontSize: 24,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '$_remembered / $total remembered on first pass.',
              style: GoogleFonts.googleSans(
                color: mutedColor,
                fontSize: 15,
              ),
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Return to Planner'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCard() {
    final isDark = AxonThemeMode.isDark;
    final textColor = isDark ? Colors.white : Colors.black;
    final mutedColor = isDark ? Colors.white54 : Colors.black54;
    final mutedBg = isDark ? Colors.white.withValues(alpha: 0.04) : Colors.black.withValues(alpha: 0.04);
    final borderColor = isDark ? Colors.white12 : Colors.black12;
    final card = _cards[_index];
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Spacer(),
              Text(
                'CARD ${_index + 1} / ${_cards.length}',
                style: GoogleFonts.robotoMono(
                  color: mutedColor.withValues(alpha: 0.5),
                  fontSize: 10,
                  letterSpacing: 1.6,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            '${card.subject} · ${card.topicLabel}',
            style: GoogleFonts.googleSans(
              color: AxonColors.accent,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 24),
          Expanded(
            child: GestureDetector(
              onTap: _toggleReveal,
              child: AnimatedBuilder(
                animation: _flipAnimation,
                builder: (context, child) {
                  final angle = _flipAnimation.value * math.pi;
                  final isBack = _flipAnimation.value >= 0.5;
                  return Transform(
                    alignment: Alignment.center,
                    transform: Matrix4.identity()
                      ..setEntry(3, 2, 0.001)
                      ..rotateY(angle),
                    child: isBack
                        ? Transform(
                            alignment: Alignment.center,
                            transform: Matrix4.identity()..rotateY(math.pi),
                            child: _buildCardFace(card, true, isDark, textColor, mutedColor, mutedBg, borderColor),
                          )
                        : _buildCardFace(card, false, isDark, textColor, mutedColor, mutedBg, borderColor),
                  );
                },
              ),
            ),
          ),
          const SizedBox(height: 16),
          if (_reveal)
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () => _submit(false),
                    child: const Text('Not Yet'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton(
                    onPressed: () => _submit(true),
                    child: const Text('Memorized'),
                  ),
                ),
              ],
            )
          else
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: _toggleReveal,
                child: const Text('Reveal Formula'),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildCardFace(FormulaDrillCard card, bool isRevealed, bool isDark, Color textColor, Color mutedColor, Color mutedBg, Color borderColor) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: mutedBg,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: borderColor),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            card.name.toUpperCase(),
            textAlign: TextAlign.center,
            style: GoogleFonts.robotoMono(
              color: mutedColor,
              fontSize: 12,
              letterSpacing: 1.6,
            ),
          ),
          const SizedBox(height: 20),
          if (isRevealed)
            MathExpression(
              formulaTex: card.expression,
              tintColor: AxonColors.accent,
              fontSize: 22,
            ),
        ],
      ),
    );
  }
}
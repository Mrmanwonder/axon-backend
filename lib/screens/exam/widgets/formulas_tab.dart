// lib/screens/exam/widgets/formulas_tab.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../services/app_state.dart';
import '../../../services/exam_planner_service.dart';
import '../../../services/exam_planner_repository.dart';
import '../../../services/formula_drill_service.dart';
import '../../../widgets/math/math_expression.dart';
import '../../study/formula_drill_screen.dart';

class FormulasTab extends ConsumerWidget {
  const FormulasTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(examPlannerProvider);

    return state.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (error, _) => Center(child: Text('Error: $error')),
      data: (data) {
        final formulas = data.formulas;
        final topics = formulas.keys.toList();

        if (topics.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.functions_rounded, color: Colors.white24, size: 48),
                const SizedBox(height: 16),
                Text('No formulas available',
                    style: TextStyle(color: Colors.white54)),
                const SizedBox(height: 8),
                MathExpression(
                  formulaTex: r'\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1',
                  fontSize: 20,
                ),
              ],
            ),
          );
        }

        return FormulasContent(formulas: formulas, topics: topics);
      },
    );
  }
}

class FormulasContent extends ConsumerStatefulWidget {
  final Map<String, FormulaSheet> formulas;
  final List<String> topics;

  const FormulasContent({required this.formulas, required this.topics});

  @override
  ConsumerState<FormulasContent> createState() => FormulasContentState();
}

class FormulasContentState extends ConsumerState<FormulasContent> {
  String _selectedTopic = 'algebra';
  final _searchController = TextEditingController();
  String _searchQuery = '';

  @override
  void initState() {
    super.initState();
    if (widget.topics.isNotEmpty) {
      _selectedTopic = widget.topics.first;
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  String _formatTopic(String topic) {
    return topic
        .split('_')
        .map((w) => w[0].toUpperCase() + w.substring(1))
        .join(' ');
  }

  Future<void> _openDrill(List<FormulaItem> filtered) async {
    final topics = filtered.map((item) => item.topicName).toSet().toList();

    List<FormulaDrillCard> allDueCards = [];
    for (final topic in topics) {
      final cards = await FormulaDrillService.instance.getDueCards(
        widget.formulas,
        topicKey: topic,
      );
      allDueCards.addAll(cards);
    }

    final allowedIds =
        filtered.map((item) => '${item.topicName}::${item.name}').toSet();
    final deck =
        allDueCards.where((card) => allowedIds.contains(card.id)).toList();
    if (!mounted || deck.isEmpty) return;

    ref.read(navbarVisibleProvider.notifier).state = false;
    await Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (context) => FormulaDrillScreen(
          title: filtered.length == 1 || topics.length == 1
              ? '${_formatTopic(topics.isNotEmpty ? (topics.first ?? '') : '')} Drill'
              : 'Formula Drill',
          cards: deck,
          onReview: FormulaDrillService.instance.reviewCard,
        ),
      ),
    );
    if (mounted) {
      ref.read(navbarVisibleProvider.notifier).state = true;
      setState(() {});
    }
  }

  Future<void> _toggleFavorite(FormulaItem formula) async {
    await ref
        .read(examPlannerProvider.notifier)
        .toggleFormulaFavorite(_selectedTopic, formula.name);
  }

  @override
  Widget build(BuildContext context) {
    List<FormulaItem> filtered = [];

    if (_searchQuery.isNotEmpty) {
      for (final entry in widget.formulas.entries) {
        final topicFormulas = entry.value.formulas
            .where((f) =>
                f.name.toLowerCase().contains(_searchQuery.toLowerCase()) ||
                f.expression.toLowerCase().contains(_searchQuery.toLowerCase()))
            .toList();
        for (final f in topicFormulas) {
          filtered.add(FormulaItem(
            name: f.name,
            expression: f.expression,
            explanation: f.explanation,
            isFavorite: f.isFavorite,
            topicName: entry.key,
          ));
        }
      }
    } else {
      final sheet = widget.formulas[_selectedTopic];
      filtered = sheet?.formulas.toList() ?? [];
    }

    return Column(
      children: [
        Container(
          color: const Color(0xFF0A0A0A),
          padding: const EdgeInsets.only(top: 8),
          child: Column(
            children: [
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _searchController,
                        onChanged: (v) => setState(() => _searchQuery = v),
                        style: const TextStyle(color: Colors.white),
                        decoration: InputDecoration(
                          hintText: 'Search formulas...',
                          hintStyle: TextStyle(
                              color: Colors.white.withValues(alpha: 0.3)),
                          prefixIcon: Icon(Icons.search,
                              color: Colors.white.withValues(alpha: 0.5),
                              size: 20),
                          suffixIcon: _searchQuery.isNotEmpty
                              ? IconButton(
                                  icon: const Icon(Icons.clear,
                                      size: 18, color: Colors.white54),
                                  onPressed: () {
                                    _searchController.clear();
                                    setState(() => _searchQuery = '');
                                  },
                                )
                              : null,
                          filled: true,
                          fillColor: Colors.white.withValues(alpha: 0.05),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(24),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: const EdgeInsets.symmetric(
                              horizontal: 20, vertical: 14),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    AnimatedDrillButton(
                        onPressed: filtered.isEmpty
                            ? null
                            : () => _openDrill(filtered)),
                  ],
                ),
              ),
              SizedBox(
                height: 40,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  itemCount: widget.topics.length,
                  itemBuilder: (context, i) {
                    final topic = widget.topics[i];
                    final isSelected = topic == _selectedTopic;
                    return GestureDetector(
                      key: ValueKey('topic-${topic}'),
                      onTap: () => setState(() => _selectedTopic = topic),
                      child: Container(
                        margin: const EdgeInsets.only(right: 8),
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: isSelected
                              ? const Color(0xFF3A86FF).withValues(alpha: 0.2)
                              : Colors.white.withValues(alpha: 0.03),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: isSelected
                                ? const Color(0xFF3A86FF).withValues(alpha: 0.4)
                                : Colors.transparent,
                          ),
                        ),
                        alignment: Alignment.center,
                        child: Text(
                          _formatTopic(topic),
                          style: TextStyle(
                            color: isSelected
                                ? const Color(0xFF3A86FF)
                                : Colors.white54,
                            fontSize: 12,
                            fontWeight:
                                isSelected ? FontWeight.w600 : FontWeight.w500,
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: filtered.isEmpty
              ? const Center(
                  child: Text('No formulas found',
                      style: TextStyle(color: Colors.white54)))
              : ListView.builder(
                  padding: const EdgeInsets.fromLTRB(20, 20, 20, 100),
                  itemCount: filtered.length,
                  itemBuilder: (context, i) => FormulaCard(
                    formula: filtered[i],
                    onFavorite: () => _toggleFavorite(filtered[i]),
                  ),
                ),
        ),
      ],
    );
  }
}

class FormulaCard extends StatelessWidget {
  final FormulaItem formula;
  final VoidCallback onFavorite;

  const FormulaCard({required this.formula, required this.onFavorite, super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.15)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  formula.name.toUpperCase(),
                  style: GoogleFonts.robotoMono(
                      color: Colors.white38, fontSize: 8, letterSpacing: 1.5),
                ),
                const SizedBox(height: 8),
                MathExpression(
                  formulaTex: formula.latexExpression,
                  tintColor: const Color(0xFF3A86FF),
                  fontSize: 18,
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: onFavorite,
            icon: Icon(
              formula.isFavorite
                  ? Icons.star_rounded
                  : Icons.star_outline_rounded,
              color: formula.isFavorite ? Colors.amber : Colors.white10,
              size: 20,
            ),
          ),
        ],
      ),
    );
  }
}

class AnimatedDrillButton extends StatefulWidget {
  final VoidCallback? onPressed;
  const AnimatedDrillButton({this.onPressed, super.key});

  @override
  State<AnimatedDrillButton> createState() => AnimatedDrillButtonState();
}

class AnimatedDrillButtonState extends State<AnimatedDrillButton> {
  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: widget.onPressed,
      child: Container(
        width: 48,
        height: 48,
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.3),
            width: 1,
          ),
        ),
        child: Icon(
          Icons.bolt_rounded,
          color: Colors.white,
          size: 24,
        ),
      ),
    );
  }
}

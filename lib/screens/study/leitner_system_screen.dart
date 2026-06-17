import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/study_techniques_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/rose_loader.dart';

class LeitnerSystemScreen extends ConsumerStatefulWidget {
  const LeitnerSystemScreen({super.key});

  @override
  ConsumerState<LeitnerSystemScreen> createState() =>
      _LeitnerSystemScreenState();
}

class _LeitnerSystemScreenState extends ConsumerState<LeitnerSystemScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  List<LeitnerCard> _cards = [];
  List<LeitnerBox> _boxes = [];
  bool _isLoading = true;
  LeitnerCard? _currentCard;
  bool _showAnswer = false;
  int _currentBox = 1;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _loadCards();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _loadCards() async {
    setState(() => _isLoading = true);
    final cards = await StudyTechniquesService.instance.getLeitnerCards();
    final boxes = StudyTechniquesService.instance.organizeCardsIntoBoxes(cards);
    if (mounted) {
      setState(() {
        _cards = cards;
        _boxes = boxes;
        _isLoading = false;
      });
    }
  }

  Future<void> _loadDueCards() async {
    final cards = await StudyTechniquesService.instance.getDueCards();
    if (cards.isNotEmpty) {
      setState(() {
        _currentCard = cards.first;
        _showAnswer = false;
      });
    } else {
      setState(() => _currentCard = null);
    }
  }

  void _showAddCardDialog() {
    final frontController = TextEditingController();
    final backController = TextEditingController();
    final subjectController = TextEditingController();
    final topicController = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.75,
        decoration: BoxDecoration(
          color: AxonColors.oxfordBlue,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.inbox, color: Color(0xFF95E1D3), size: 28),
                const SizedBox(width: 12),
                Text(
                  'Add Flashcard',
                  style: GoogleFonts.inter(
                    color: AxonColors.textPrimary,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            TextField(
              controller: subjectController,
              style: TextStyle(color: AxonColors.textPrimary),
              decoration: _inputDecoration('Subject', Icons.book),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: topicController,
              style: TextStyle(color: AxonColors.textPrimary),
              decoration: _inputDecoration('Topic', Icons.category),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: frontController,
              style: TextStyle(color: AxonColors.textPrimary),
              maxLines: 2,
              decoration: _inputDecoration('Front (Question)', Icons.flip),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: backController,
              style: TextStyle(color: AxonColors.textPrimary),
              maxLines: 2,
              decoration: _inputDecoration('Back (Answer)', Icons.check_circle),
            ),
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () async {
                  if (frontController.text.isEmpty ||
                      backController.text.isEmpty) {
                    return;
                  }

                  HapticFeedback.mediumImpact();
                  final card = LeitnerCard(
                    id: 'leitner_${DateTime.now().millisecondsSinceEpoch}',
                    front: frontController.text,
                    back: backController.text,
                    subject: subjectController.text,
                    topic: topicController.text,
                  );

                  await StudyTechniquesService.instance.addLeitnerCard(card);
                  await _loadCards();
                  if (context.mounted) {
                    Navigator.pop(context);
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF95E1D3),
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Add Card',
                  style: TextStyle(
                      color: Colors.black, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  InputDecoration _inputDecoration(String label, IconData icon) {
    return InputDecoration(
      labelText: label,
      labelStyle: TextStyle(color: AxonColors.textSecondary),
      prefixIcon: Icon(icon, color: AxonColors.textTertiary),
      filled: true,
      fillColor: AxonColors.textPrimary.withValues(alpha: 0.05),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFF95E1D3)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: AxonColors.background,
        leading: IconButton(
          onPressed: () => popOrGo(context, '/timer'),
          icon: Icon(Icons.arrow_back_rounded, color: AxonColors.textPrimary),
        ),
        title: Row(
          children: [
            Icon(Icons.inbox, color: Color(0xFF95E1D3)),
            const SizedBox(width: 8),
            Text(
              'Leitner System',
              style: GoogleFonts.inter(
                  color: AxonColors.textPrimary, fontWeight: FontWeight.bold),
            ),
          ],
        ),
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: Color(0xFF95E1D3),
          labelColor: Color(0xFF95E1D3),
          unselectedLabelColor: AxonColors.textSecondary,
          tabs: const [
            Tab(text: 'Boxes'),
            Tab(text: 'Study'),
            Tab(text: 'Add'),
          ],
        ),
      ),
      body: _isLoading
          ? const Center(child: RoseLoader(size: 24, color: Color(0xFF95E1D3)))
          : TabBarView(
              controller: _tabController,
              children: [
                _buildBoxesTab(),
                _buildStudyTab(),
                _buildAddTab(),
              ],
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: _showAddCardDialog,
        backgroundColor: Color(0xFF95E1D3),
        child: const Icon(Icons.add, color: Colors.black),
      ),
    );
  }

  Widget _buildBoxesTab() {
    if (_boxes.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.inbox_outlined, color: AxonColors.textTertiary, size: 64),
            const SizedBox(height: 16),
            Text('No cards yet', style: TextStyle(color: AxonColors.textSecondary)),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _boxes.length,
      itemBuilder: (context, index) {
        final box = _boxes[index];
        return _LeitnerBoxCard(
          box: box,
          onTap: () {
            setState(() => _currentBox = box.boxNumber);
            _tabController.animateTo(1);
          },
        );
      },
    );
  }

  Widget _buildStudyTab() {
    if (_currentCard == null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.celebration, color: Color(0xFF95E1D3), size: 64),
          const SizedBox(height: 16),
          Text('No cards due for review!',
              style: TextStyle(color: AxonColors.textSecondary, fontSize: 18)),
          const SizedBox(height: 8),
          Text('Check back later or add more cards',
              style: TextStyle(color: AxonColors.textTertiary)),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: _loadDueCards,
            style: ElevatedButton.styleFrom(
              backgroundColor: Color(0xFF95E1D3),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
            child: const Text('Check for Due Cards',
                style: TextStyle(color: Colors.black)),
          ),
        ],
      );
    }

    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(5, (i) {
              final isActive = i + 1 == _currentBox;
              return Container(
                margin: const EdgeInsets.symmetric(horizontal: 4),
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: isActive
                      ? Color(0xFF95E1D3).withValues(alpha: 0.2)
                      : Colors.white.withValues(alpha: 0.05),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: isActive ? Color(0xFF95E1D3) : Colors.transparent,
                  ),
                ),
                child: Text(
                  'Box ${i + 1}',
                  style: TextStyle(
                    color: isActive ? Color(0xFF95E1D3) : Colors.white54,
                    fontSize: 12,
                  ),
                ),
              );
            }),
          ),
        ),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: GestureDetector(
              onTap: () => setState(() => _showAnswer = !_showAnswer),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  color: AxonColors.oxfordBlue,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: _showAnswer
                        ? Color(0xFF95E1D3)
                        : Colors.white.withValues(alpha: 0.1),
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (!_showAnswer) ...[
                      Text(
                        _currentCard!.front,
                        textAlign: TextAlign.center,
                        style: GoogleFonts.inter(
                          color: AxonColors.textPrimary,
                          fontSize: 24,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 24),
                      Text(
                        'Tap to reveal answer',
                        style: TextStyle(color: AxonColors.textTertiary),
                      ),
                    ] else ...[
                      Text(
                        _currentCard!.front,
                        textAlign: TextAlign.center,
                        style: TextStyle(color: AxonColors.textSecondary, fontSize: 16),
                      ),
                      const SizedBox(height: 24),
                      Container(
                        height: 2,
                        width: 100,
                        color: Color(0xFF95E1D3),
                      ),
                      const SizedBox(height: 24),
                      Text(
                        _currentCard!.back,
                        textAlign: TextAlign.center,
                        style: GoogleFonts.inter(
                          color: Color(0xFF95E1D3),
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
        ),
        if (_showAnswer)
          Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () async {
                      HapticFeedback.mediumImpact();
                      await StudyTechniquesService.instance
                          .processLeitnerReview(_currentCard!.id, false);
                      await _loadDueCards();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.redAccent.withValues(alpha: 0.2),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                        side: const BorderSide(color: Colors.redAccent),
                      ),
                    ),
                    child: const Text('Again',
                        style: TextStyle(color: Colors.redAccent)),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () async {
                      HapticFeedback.mediumImpact();
                      await StudyTechniquesService.instance
                          .processLeitnerReview(_currentCard!.id, true);
                      await _loadDueCards();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFF95E1D3),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: const Text('Got it!',
                        style: TextStyle(
                            color: Colors.black, fontWeight: FontWeight.bold)),
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _buildAddTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Color(0xFF95E1D3).withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.info_outline, color: Color(0xFF95E1D3)),
                    const SizedBox(width: 8),
                    Text(
                      'How it works',
                      style: TextStyle(
                        color: Color(0xFF95E1D3),
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _explanationRow('Box 1', 'Review daily', '1 day'),
                _explanationRow('Box 2', 'Every 2 days', '2 days'),
                _explanationRow('Box 3', 'Every 4 days', '4 days'),
                _explanationRow('Box 4', 'Every 8 days', '8 days'),
                _explanationRow('Box 5', 'Every 16 days', '16 days'),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'Stats',
            style: TextStyle(
                color: AxonColors.textPrimary, fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _statCard('Total', _cards.length.toString(), Color(0xFF95E1D3)),
              const SizedBox(width: 12),
              _statCard(
                  'Due',
                  _cards
                      .where((c) => c.nextReview.isBefore(DateTime.now()))
                      .length
                      .toString(),
                  Colors.orange),
            ],
          ),
        ],
      ),
    );
  }

  Widget _explanationRow(String box, String desc, String interval) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(box,
              style:
                  TextStyle(color: AxonColors.textPrimary, fontWeight: FontWeight.bold)),
          const SizedBox(width: 8),
          Expanded(child: Text(desc, style: TextStyle(color: AxonColors.textSecondary))),
          Text(interval, style: TextStyle(color: Color(0xFF95E1D3))),
        ],
      ),
    );
  }

  Widget _statCard(String label, String value, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Text(value,
                style: TextStyle(
                    color: color, fontSize: 24, fontWeight: FontWeight.bold)),
            Text(label, style: TextStyle(color: AxonColors.textSecondary, fontSize: 12)),
          ],
        ),
      ),
    );
  }
}

class _LeitnerBoxCard extends StatelessWidget {
  final LeitnerBox box;
  final VoidCallback onTap;

  const _LeitnerBoxCard({required this.box, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final interval = pow(2, box.boxNumber - 1).toInt();

    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AxonColors.textPrimary.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: Color(0xFF95E1D3).withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(
                child: Text(
                  '${box.boxNumber}',
                  style: TextStyle(
                    color: Color(0xFF95E1D3),
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Box ${box.boxNumber}',
                    style: TextStyle(
                      color: AxonColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    'Review every $interval day${interval > 1 ? 's' : ''}',
                    style: TextStyle(color: AxonColors.textTertiary, fontSize: 12),
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Color(0xFF95E1D3).withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                '${box.cards.length}',
                style: TextStyle(
                  color: Color(0xFF95E1D3),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(width: 8),
            Icon(Icons.chevron_right, color: AxonColors.textTertiary),
          ],
        ),
      ),
    );
  }
}

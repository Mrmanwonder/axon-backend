import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/supabase_data_service.dart';
import '../../services/study_techniques_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/rose_loader.dart';

class ActiveRecallScreen extends ConsumerStatefulWidget {
  final String? subject;

  const ActiveRecallScreen({super.key, this.subject});

  @override
  ConsumerState<ActiveRecallScreen> createState() => _ActiveRecallScreenState();
}

class _ActiveRecallScreenState extends ConsumerState<ActiveRecallScreen> {
  List<Map<String, dynamic>> _items = [];
  bool _isLoading = true;
  int _currentIndex = 0;
  bool _showAnswer = false;
  bool _sessionActive = false;
  int _correctCount = 0;
  int _totalCount = 0;

  @override
  void initState() {
    super.initState();
    _loadItems();
  }

  Future<void> _loadItems() async {
    setState(() => _isLoading = true);
    var items = await StudyTechniquesService.instance
        .getActiveRecallItems(subject: widget.subject);
    if (items.isEmpty) {
      items = await _loadSeededGlobalRecallItems();
    }
    if (mounted) {
      setState(() {
        _items = items;
        _isLoading = false;
      });
    }
  }

  Future<List<Map<String, dynamic>>> _loadSeededGlobalRecallItems() async {
    final globalNotes = await SupabaseDataService.instance.getAllGlobalNotes();
    return globalNotes
        .where((note) {
          final subjectCode = (note['subject_code'] ?? '').toString();
          return widget.subject == null ||
              widget.subject!.isEmpty ||
              subjectCode.toLowerCase().contains(widget.subject!.toLowerCase()) ||
              widget.subject!.toLowerCase().contains(subjectCode.toLowerCase());
        })
        .take(20)
        .map((note) => {
              'id': 'global_${note['id'] ?? DateTime.now().millisecondsSinceEpoch}',
              'question':
                  'Explain ${(note['title'] ?? note['chapter_id'] ?? 'this concept').toString().trim()}',
              'answer': (note['content'] ?? note['note_text'] ?? '')
                  .toString()
                  .trim(),
              'subject': (note['subject_code'] ?? widget.subject ?? 'General')
                  .toString(),
              'createdAt': DateTime.now().toIso8601String(),
              'reviewedCount': 0,
            })
        .where((item) => '${item['answer']}'.isNotEmpty)
        .toList();
  }

  void _startSession() {
    if (_items.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Add recall items first')),
      );
      return;
    }

    setState(() {
      _sessionActive = true;
      _currentIndex = 0;
      _showAnswer = false;
      _correctCount = 0;
      _totalCount = 0;
    });
  }

  void _endSession() {
    if (_totalCount > 0) {
      StudyTechniquesService.instance.recordSession(
        techniqueId: 'active_recall',
        duration: Duration(minutes: _totalCount),
        problemsCompleted: _totalCount,
        correctAnswers: _correctCount,
      );
    }

    setState(() {
      _sessionActive = false;
      _currentIndex = 0;
      _showAnswer = false;
    });
  }

  void _showAddItemDialog() {
    final questionController = TextEditingController();
    final answerController = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.65,
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
                Icon(Icons.psychology, color: AxonColors.accent),
                const SizedBox(width: 12),
                Text(
                  'Add Recall Item',
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
              controller: questionController,
              style: TextStyle(color: AxonColors.textPrimary),
              maxLines: 2,
              decoration: InputDecoration(
                labelText: 'Question / Prompt',
                labelStyle: TextStyle(color: AxonColors.textSecondary),
                prefixIcon:
                    Icon(Icons.help_outline, color: AxonColors.textTertiary),
                filled: true,
                fillColor: AxonColors.textPrimary.withValues(alpha: 0.05),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: AxonColors.accent),
                ),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: answerController,
              style: TextStyle(color: AxonColors.textPrimary),
              maxLines: 2,
              decoration: InputDecoration(
                labelText: 'Answer',
                labelStyle: TextStyle(color: AxonColors.textSecondary),
                prefixIcon:
                    Icon(Icons.lightbulb_outline, color: AxonColors.textTertiary),
                filled: true,
                fillColor: AxonColors.textPrimary.withValues(alpha: 0.05),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: AxonColors.accent),
                ),
              ),
            ),
            const Spacer(),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () async {
                  if (questionController.text.isEmpty ||
                      answerController.text.isEmpty) {
                    return;
                  }

                  HapticFeedback.mediumImpact();
                  await StudyTechniquesService.instance.saveActiveRecallItem({
                    'id': 'recall_${DateTime.now().millisecondsSinceEpoch}',
                    'question': questionController.text,
                    'answer': answerController.text,
                    'subject': widget.subject ?? 'General',
                    'createdAt': DateTime.now().toIso8601String(),
                    'reviewedCount': 0,
                  });
                  await _loadItems();
                  if (!context.mounted) return;
                  Navigator.pop(context);
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: Text(
                  'Add Item',
                  style: TextStyle(
                      color: AxonColors.background, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_sessionActive) {
      return _buildActiveSession();
    }

    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: AxonColors.background,
        leading: IconButton(
          onPressed: () => popOrGo(context, '/study'),
          icon: Icon(Icons.arrow_back_rounded, color: AxonColors.textPrimary),
        ),
        title: Row(
          children: [
            Icon(Icons.psychology, color: AxonColors.accent),
            const SizedBox(width: 8),
            Text(
              'Active Recall',
              style: GoogleFonts.inter(
                  color: AxonColors.textPrimary, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
      body: _isLoading
          ? Center(child: RoseLoader(size: 24, color: AxonColors.accent))
          : _items.isEmpty
              ? _buildEmptyState()
              : _buildItemList(),
      floatingActionButton: FloatingActionButton(
        onPressed: _showAddItemDialog,
        backgroundColor: AxonColors.accent,
        child: Icon(Icons.add, color: AxonColors.background),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.psychology_outlined, color: AxonColors.textTertiary, size: 64),
          const SizedBox(height: 16),
          Text('No recall items yet',
              style: TextStyle(color: AxonColors.textSecondary, fontSize: 18)),
          const SizedBox(height: 8),
          Text('Add questions to test yourself',
              style: TextStyle(color: AxonColors.textTertiary)),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: _showAddItemDialog,
            icon: const Icon(Icons.add),
            label: const Text('Add First Item'),
            style: ElevatedButton.styleFrom(
              backgroundColor: AxonColors.accent,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildItemList() {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Text(
                '${_items.length} items',
                style: TextStyle(color: AxonColors.textSecondary),
              ),
              const Spacer(),
              ElevatedButton.icon(
                onPressed: _startSession,
                icon: const Icon(Icons.play_arrow, size: 18),
                label: const Text('Start Practice'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            itemCount: _items.length,
            itemBuilder: (context, index) {
              final item = _items[index];
              return _RecallItemCard(
                item: item,
                onDelete: () async {
                  await StudyTechniquesService.instance
                      .deleteActiveRecallItem('${item['id']}');
                  await _loadItems();
                },
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _buildActiveSession() {
    if (_currentIndex >= _items.length) {
      return Scaffold(
        backgroundColor: AxonColors.background,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.celebration, color: AxonColors.accent, size: 64),
              const SizedBox(height: 16),
              Text('Session Complete!',
                  style: TextStyle(color: AxonColors.textPrimary, fontSize: 24)),
              const SizedBox(height: 8),
              Text(
                '$_correctCount / $_totalCount correct (${(_totalCount > 0 ? (_correctCount / _totalCount * 100).round() : 0)}%)',
                style: TextStyle(color: AxonColors.textSecondary, fontSize: 18),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _endSession,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: Text('Finish',
                    style: TextStyle(
                        color: AxonColors.background, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
        ),
      );
    }

    final item = _items[_currentIndex];

    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: AxonColors.background,
        leading: IconButton(
          onPressed: _endSession,
          icon: Icon(Icons.close, color: AxonColors.textSecondary),
        ),
        title: Text(
          '${_currentIndex + 1} / ${_items.length}',
          style: TextStyle(color: AxonColors.textPrimary),
        ),
      ),
      body: Column(
        children: [
          LinearProgressIndicator(
            value: _currentIndex / _items.length,
            backgroundColor: Colors.white10,
            valueColor: AlwaysStoppedAnimation(AxonColors.accent),
          ),
          Expanded(
            child: GestureDetector(
              onTap: () => setState(() => _showAnswer = !_showAnswer),
              child: Container(
                margin: const EdgeInsets.all(24),
                padding: const EdgeInsets.all(32),
                decoration: BoxDecoration(
                  color: AxonColors.oxfordBlue,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(
                    color: _showAnswer
                        ? AxonColors.accent
                        : Colors.white.withValues(alpha: 0.1),
                    width: 2,
                  ),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (!_showAnswer) ...[
                      Text(
                        item['question'] ?? '',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.inter(
                          color: AxonColors.textPrimary,
                          fontSize: 24,
                        ),
                      ),
                      const SizedBox(height: 32),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.touch_app, color: AxonColors.textTertiary),
                          const SizedBox(width: 8),
                          Text(
                            'Tap to reveal answer',
                            style: TextStyle(color: AxonColors.textTertiary),
                          ),
                        ],
                      ),
                    ] else ...[
                      Text(
                        item['question'] ?? '',
                        style: TextStyle(color: AxonColors.textSecondary, fontSize: 16),
                      ),
                      const SizedBox(height: 24),
                      Container(height: 2, width: 80, color: AxonColors.accent),
                      const SizedBox(height: 24),
                      Text(
                        item['answer'] ?? '',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.inter(
                          color: AxonColors.accent,
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
          if (_showAnswer)
            Padding(
              padding: const EdgeInsets.all(24),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        HapticFeedback.mediumImpact();
                        setState(() {
                          _totalCount++;
                          _showAnswer = false;
                          _currentIndex++;
                        });
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor:
                            Colors.redAccent.withValues(alpha: 0.2),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                          side: const BorderSide(color: Colors.redAccent),
                        ),
                      ),
                      child: const Text('Wrong',
                          style: TextStyle(color: Colors.redAccent)),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () {
                        HapticFeedback.mediumImpact();
                        setState(() {
                          _totalCount++;
                          _correctCount++;
                          _showAnswer = false;
                          _currentIndex++;
                        });
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AxonColors.accent,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: Text('Got it!',
                          style: TextStyle(
                              color: AxonColors.background,
                              fontWeight: FontWeight.bold)),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _RecallItemCard extends StatelessWidget {
  final Map<String, dynamic> item;
  final VoidCallback onDelete;

  const _RecallItemCard({required this.item, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.textPrimary.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  item['question'] ?? '',
                  style: TextStyle(
                    color: AxonColors.textPrimary,
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              IconButton(
                onPressed: onDelete,
                icon:
                    Icon(Icons.delete_outline, color: AxonColors.textTertiary, size: 20),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            item['answer'] ?? '',
            style: TextStyle(color: AxonColors.textSecondary, fontSize: 13),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

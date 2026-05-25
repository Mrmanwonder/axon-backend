import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/supabase_data_service.dart';
import '../../services/study_techniques_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/rose_loader.dart';

class FeynmanTechniqueScreen extends ConsumerStatefulWidget {
  const FeynmanTechniqueScreen({super.key});

  @override
  ConsumerState<FeynmanTechniqueScreen> createState() =>
      _FeynmanTechniqueScreenState();
}

class _FeynmanTechniqueScreenState extends ConsumerState<FeynmanTechniqueScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  List<FeynmanNote> _notes = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _loadNotes();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Future<void> _loadNotes() async {
    setState(() => _isLoading = true);
    final localNotes = await StudyTechniquesService.instance.getFeynmanNotes();
    var notes = localNotes;
    if (notes.isEmpty) {
      notes = await _loadSeededGlobalNotes();
    }
    if (mounted) {
      setState(() {
        _notes = notes;
        _isLoading = false;
      });
    }
  }

  Future<List<FeynmanNote>> _loadSeededGlobalNotes() async {
    final globalNotes = await SupabaseDataService.instance.getAllGlobalNotes();
    return globalNotes.take(12).map((note) {
      final title = (note['title'] ?? note['chapter_id'] ?? 'Global Note')
          .toString()
          .trim();
      final content = (note['content'] ?? note['note_text'] ?? '')
          .toString()
          .trim();
      return FeynmanNote(
        id: 'global_${note['id'] ?? title}',
        topic: (note['subject_code'] ?? 'Global').toString(),
        concept: title.isEmpty ? 'Global Note' : title,
        explanation: content,
        simpleExplanation: content,
      );
    }).where((note) => note.explanation.isNotEmpty).toList();
  }

  void _showCreateNoteDialog() {
    final conceptController = TextEditingController();
    final explanationController = TextEditingController();
    final simpleController = TextEditingController();
    final topicController = TextEditingController();

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.85,
        decoration: BoxDecoration(
          color: AxonColors.oxfordBlue,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Padding(
          padding: EdgeInsets.only(
            left: 24,
            right: 24,
            top: 24,
            bottom: MediaQuery.of(context).viewInsets.bottom + 24,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(Icons.lightbulb, color: Color(0xFFFFE66D), size: 28),
                  const SizedBox(width: 12),
                  Text(
                    'New Feynman Note',
                    style: GoogleFonts.inter(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              TextField(
                controller: topicController,
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration('Topic', Icons.bookmark),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: conceptController,
                style: const TextStyle(color: Colors.white),
                decoration: _inputDecoration('Concept', Icons.fingerprint),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: explanationController,
                style: const TextStyle(color: Colors.white),
                maxLines: 3,
                decoration: _inputDecoration('Full Explanation', Icons.article),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: simpleController,
                style: const TextStyle(color: Colors.white),
                maxLines: 2,
                decoration: _inputDecoration(
                  'Simple Explanation (Child can understand)',
                  Icons.child_care,
                ),
              ),
              const Spacer(),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () async {
                    if (conceptController.text.isEmpty) return;

                    HapticFeedback.mediumImpact();
                    final note = FeynmanNote(
                      id: 'feynman_${DateTime.now().millisecondsSinceEpoch}',
                      topic: topicController.text,
                      concept: conceptController.text,
                      explanation: explanationController.text,
                      simpleExplanation: simpleController.text,
                    );

                    await StudyTechniquesService.instance.addFeynmanNote(note);
                    await _loadNotes();
                    if (!context.mounted) return;
                    Navigator.pop(context);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Color(0xFFFFE66D),
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text(
                    'Create Note',
                    style: TextStyle(
                      color: Colors.black,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  InputDecoration _inputDecoration(String label, IconData icon) {
    return InputDecoration(
      labelText: label,
      labelStyle: const TextStyle(color: Colors.white54),
      prefixIcon: Icon(icon, color: Colors.white38),
      filled: true,
      fillColor: Colors.white.withValues(alpha: 0.05),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Color(0xFFFFE66D)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        leading: IconButton(
          onPressed: () => popOrGo(context, '/timer'),
          icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
        ),
        title: Row(
          children: [
            Icon(Icons.lightbulb, color: Color(0xFFFFE66D)),
            const SizedBox(width: 8),
            Text(
              'Feynman Technique',
              style: GoogleFonts.inter(
                  color: Colors.white, fontWeight: FontWeight.bold),
            ),
          ],
        ),
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: Color(0xFFFFE66D),
          labelColor: Color(0xFFFFE66D),
          unselectedLabelColor: Colors.white54,
          tabs: const [
            Tab(text: 'Learn'),
            Tab(text: 'Create'),
            Tab(text: 'Review'),
          ],
        ),
      ),
      body: _isLoading
          ? const Center(child: RoseLoader(size: 24, color: Color(0xFFFFE66D)))
          : TabBarView(
              controller: _tabController,
              children: [
                _buildLearnTab(),
                _buildCreateTab(),
                _buildReviewTab(),
              ],
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: _showCreateNoteDialog,
        backgroundColor: Color(0xFFFFE66D),
        child: const Icon(Icons.add, color: Colors.black),
      ),
    );
  }

  Widget _buildLearnTab() {
    if (_notes.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.lightbulb_outline, color: Colors.white24, size: 64),
            const SizedBox(height: 16),
            Text(
              'No Feynman notes yet',
              style: TextStyle(color: Colors.white54, fontSize: 16),
            ),
            const SizedBox(height: 8),
            Text(
              'Create one to start learning',
              style: TextStyle(color: Colors.white38, fontSize: 14),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _notes.length,
      itemBuilder: (context, index) {
        final note = _notes[index];
        return _FeynmanNoteCard(
          note: note,
          onTap: () => _showNoteDetail(note),
        );
      },
    );
  }

  Widget _buildCreateTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Color(0xFFFFE66D).withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(16),
              border:
                  Border.all(color: Color(0xFFFFE66D).withValues(alpha: 0.3)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.school, color: Color(0xFFFFE66D)),
                    const SizedBox(width: 8),
                    Text(
                      'How to use the Feynman Technique',
                      style: TextStyle(
                        color: Color(0xFFFFE66D),
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _stepItem('1', 'Choose a concept you want to learn'),
                _stepItem('2', 'Explain it simply (as if to a child)'),
                _stepItem('3', 'Identify gaps in your understanding'),
                _stepItem('4', 'Review and simplify again'),
                _stepItem('5', 'Use analogies to solidify understanding'),
              ],
            ),
          ),
          const SizedBox(height: 24),
          Text(
            'Quick Create',
            style: TextStyle(
                color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          _quickCreateButton('Physics', Icons.science),
          _quickCreateButton('Math', Icons.calculate),
          _quickCreateButton('Chemistry', Icons.biotech),
          _quickCreateButton('Biology', Icons.eco),
        ],
      ),
    );
  }

  Widget _stepItem(String number, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(
            width: 24,
            height: 24,
            decoration: BoxDecoration(
              color: Color(0xFFFFE66D).withValues(alpha: 0.2),
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(number,
                  style: TextStyle(color: Color(0xFFFFE66D), fontSize: 12)),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(text, style: TextStyle(color: Colors.white70)),
          ),
        ],
      ),
    );
  }

  Widget _quickCreateButton(String subject, IconData icon) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: GestureDetector(
        onTap: () {
          HapticFeedback.lightImpact();
          _showCreateNoteDialog();
        },
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.05),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              Icon(icon, color: Colors.white54),
              const SizedBox(width: 12),
              Text(subject,
                  style: TextStyle(color: Colors.white, fontSize: 16)),
              const Spacer(),
              Icon(Icons.add, color: Colors.white38),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildReviewTab() {
    final reviewNotes = _notes
        .where((n) => DateTime.now().difference(n.lastReviewed).inDays >= 1)
        .toList();

    if (reviewNotes.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.check_circle, color: Colors.green, size: 64),
            const SizedBox(height: 16),
            Text(
              'All caught up!',
              style: TextStyle(color: Colors.white54, fontSize: 16),
            ),
            const SizedBox(height: 8),
            Text(
              'Review your notes tomorrow',
              style: TextStyle(color: Colors.white38, fontSize: 14),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: reviewNotes.length,
      itemBuilder: (context, index) {
        final note = reviewNotes[index];
        return _FeynmanNoteCard(
          note: note,
          onTap: () => _showNoteDetail(note),
          showReview: true,
        );
      },
    );
  }

  void _showNoteDetail(FeynmanNote note) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.8,
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
                Expanded(
                  child: Text(
                    note.concept,
                    style: GoogleFonts.inter(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: () async {
                    HapticFeedback.mediumImpact();
                    await StudyTechniquesService.instance
                        .deleteFeynmanNote(note.id);
                    await _loadNotes();
                    if (!context.mounted) return;
                    Navigator.pop(context);
                  },
                  icon: Icon(Icons.delete, color: Colors.redAccent),
                ),
              ],
            ),
            if (note.topic.isNotEmpty)
              Text(
                note.topic,
                style: TextStyle(color: Colors.white54, fontSize: 14),
              ),
            const SizedBox(height: 24),
            _detailSection('Full Explanation', note.explanation),
            const SizedBox(height: 16),
            _detailSection('Simple Explanation', note.simpleExplanation,
                highlight: true),
            const Spacer(),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      HapticFeedback.mediumImpact();
                      Navigator.pop(context);
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFFFFE66D),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: const Text(
                      'Mark Reviewed',
                      style: TextStyle(
                          color: Colors.black, fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _detailSection(String title, String content,
      {bool highlight = false}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: TextStyle(
            color: highlight ? Color(0xFFFFE66D) : Colors.white54,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: highlight
                ? Color(0xFFFFE66D).withValues(alpha: 0.1)
                : Colors.white.withValues(alpha: 0.05),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            content.isEmpty ? 'Not provided' : content,
            style: TextStyle(
              color: Colors.white,
              fontSize: 14,
              height: 1.5,
            ),
          ),
        ),
      ],
    );
  }
}

class _FeynmanNoteCard extends StatelessWidget {
  final FeynmanNote note;
  final VoidCallback onTap;
  final bool showReview;

  const _FeynmanNoteCard({
    required this.note,
    required this.onTap,
    this.showReview = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    note.concept,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                if (showReview)
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Color(0xFFFFE66D).withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      'Review',
                      style: TextStyle(
                        color: Color(0xFFFFE66D),
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
              ],
            ),
            if (note.topic.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(
                note.topic,
                style: const TextStyle(color: Colors.white38, fontSize: 12),
              ),
            ],
            const SizedBox(height: 8),
            Text(
              note.simpleExplanation.isNotEmpty
                  ? note.simpleExplanation
                  : note.explanation,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(color: Colors.white54, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}

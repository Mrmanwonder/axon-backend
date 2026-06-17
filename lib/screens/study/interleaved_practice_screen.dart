import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/study_techniques_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';

class InterleavedPracticeScreen extends ConsumerStatefulWidget {
  const InterleavedPracticeScreen({super.key});

  @override
  ConsumerState<InterleavedPracticeScreen> createState() =>
      _InterleavedPracticeScreenState();
}

class _InterleavedPracticeScreenState
    extends ConsumerState<InterleavedPracticeScreen> {
  final List<String> _selectedSubjects = [];
  final TextEditingController _questionController = TextEditingController();
  final TextEditingController _answerController = TextEditingController();
  final TextEditingController _topicController = TextEditingController();

  List<InterleavedSession> _sessions = [];
  List<InterleavedProblem> _currentProblems = [];
  int _currentProblemIndex = 0;
  bool _showAnswer = false;
  bool _sessionActive = false;
  DateTime? _sessionStart;

  @override
  void initState() {
    super.initState();
    _loadSessions();
  }

  @override
  void dispose() {
    _questionController.dispose();
    _answerController.dispose();
    _topicController.dispose();
    super.dispose();
  }

  Future<void> _loadSessions() async {
    final sessions =
        await StudyTechniquesService.instance.getInterleavedSessions();
    if (mounted) {
      setState(() => _sessions = sessions);
    }
  }

  List<String> get _availableSubjects =>
      ['Physics', 'Math', 'Chemistry', 'Biology', 'English', 'History'];

  void _startSession() {
    if (_selectedSubjects.length < 2) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
            content: Text('Select at least 2 subjects for interleaving')),
      );
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text(
            'No interleaved practice problems available yet. Complete more study sessions to generate personalized problems.'),
      ),
    );
  }

  void _endSession() {
    if (_sessionStart != null) {
      final duration = DateTime.now().difference(_sessionStart!);
      StudyTechniquesService.instance.recordSession(
        techniqueId: 'interleaving',
        duration: duration,
        problemsCompleted: _currentProblemIndex,
        correctAnswers:
            _currentProblems.where((p) => p.id.endsWith('_correct')).length,
      );
    }

    setState(() {
      _sessionActive = false;
      _currentProblems = [];
      _currentProblemIndex = 0;
      _sessionStart = null;
    });
    _loadSessions();
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
          onPressed: () => popOrGo(context, '/timer'),
          icon: Icon(Icons.arrow_back_rounded, color: AxonColors.textPrimary),
        ),
        title: Row(
          children: [
            Icon(Icons.shuffle, color: Color(0xFFA8E6CF)),
            const SizedBox(width: 8),
            Text(
              'Interleaved Practice',
              style: GoogleFonts.inter(
                  color: AxonColors.textPrimary, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildExplanationCard(),
            const SizedBox(height: 24),
            _buildSubjectSelector(),
            const SizedBox(height: 24),
            _buildHistorySection(),
          ],
        ),
      ),
      floatingActionButton: _selectedSubjects.length >= 2
          ? FloatingActionButton.extended(
              onPressed: _startSession,
              backgroundColor: Color(0xFFA8E6CF),
              icon: const Icon(Icons.play_arrow, color: Colors.black),
              label: const Text('Start Session',
                  style: TextStyle(color: Colors.black)),
            )
          : null,
    );
  }

  Widget _buildExplanationCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Color(0xFFA8E6CF).withValues(alpha: 0.2),
            Color(0xFFA8E6CF).withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.psychology, color: Color(0xFFA8E6CF)),
              const SizedBox(width: 8),
              Text(
                'Why Interleaving?',
                style: TextStyle(
                  color: Color(0xFFA8E6CF),
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Mixing different topics during practice improves your ability to discriminate between concepts and transfer learning to new situations.',
            style: TextStyle(color: AxonColors.textSecondary, height: 1.5),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _benefitChip('Better retention', Color(0xFFA8E6CF)),
              const SizedBox(width: 8),
              _benefitChip('Flexible thinking', Color(0xFFA8E6CF)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _benefitChip(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(text, style: TextStyle(color: color, fontSize: 11)),
    );
  }

  Widget _buildSubjectSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Select Subjects',
          style: TextStyle(
              color: AxonColors.textPrimary, fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        Text(
          'Choose at least 2 subjects to interleave',
          style: TextStyle(color: AxonColors.textSecondary, fontSize: 14),
        ),
        const SizedBox(height: 16),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: _availableSubjects.map((subject) {
            final isSelected = _selectedSubjects.contains(subject);
            return GestureDetector(
              onTap: () {
                HapticFeedback.lightImpact();
                setState(() {
                  if (isSelected) {
                    _selectedSubjects.remove(subject);
                  } else {
                    _selectedSubjects.add(subject);
                  }
                });
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: isSelected
                      ? Color(0xFFA8E6CF).withValues(alpha: 0.2)
                      : Colors.white.withValues(alpha: 0.05),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: isSelected ? Color(0xFFA8E6CF) : Colors.transparent,
                    width: 2,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isSelected ? Icons.check_circle : Icons.circle_outlined,
                      color: isSelected ? Color(0xFFA8E6CF) : Colors.white38,
                      size: 18,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      subject,
                      style: TextStyle(
                        color: isSelected ? Color(0xFFA8E6CF) : Colors.white,
                        fontWeight:
                            isSelected ? FontWeight.w600 : FontWeight.normal,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildHistorySection() {
    if (_sessions.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Recent Sessions',
          style: TextStyle(
              color: AxonColors.textPrimary, fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 16),
        ..._sessions.take(3).map((session) => Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AxonColors.textPrimary.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  Icon(Icons.history, color: AxonColors.textTertiary),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          session.subjects.join(", "),
                          style: TextStyle(
                              color: AxonColors.textPrimary, fontWeight: FontWeight.w600),
                        ),
                        Text(
                          '${session.totalProblems} problems',
                          style: TextStyle(color: AxonColors.textTertiary, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                  Text(
                    _formatDate(session.startedAt),
                    style: TextStyle(color: AxonColors.textSecondary, fontSize: 12),
                  ),
                ],
              ),
            )),
      ],
    );
  }

  Widget _buildActiveSession() {
    if (_currentProblems.isEmpty ||
        _currentProblemIndex >= _currentProblems.length) {
      return Scaffold(
        backgroundColor: AxonColors.background,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.celebration, color: Color(0xFFA8E6CF), size: 64),
              const SizedBox(height: 16),
              Text('Session Complete!',
                  style: TextStyle(color: AxonColors.textPrimary, fontSize: 24)),
              const SizedBox(height: 8),
              Text('You practiced $_currentProblemIndex problems',
                  style: TextStyle(color: AxonColors.textSecondary)),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _endSession,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFFA8E6CF),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Finish',
                    style: TextStyle(
                        color: Colors.black, fontWeight: FontWeight.bold)),
              ),
            ],
          ),
        ),
      );
    }

    final problem = _currentProblems[_currentProblemIndex];

    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: AxonColors.background,
        leading: IconButton(
          onPressed: _endSession,
          icon: Icon(Icons.close, color: AxonColors.textSecondary),
        ),
        title: Text(
          '${_currentProblemIndex + 1} / ${_currentProblems.length}',
          style: TextStyle(color: AxonColors.textPrimary),
        ),
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: Color(0xFFA8E6CF).withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              problem.subject,
              style: TextStyle(color: Color(0xFFA8E6CF), fontSize: 12),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          LinearProgressIndicator(
            value: _currentProblemIndex / _currentProblems.length,
            backgroundColor: Colors.white10,
            valueColor: AlwaysStoppedAnimation(Color(0xFFA8E6CF)),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                children: [
                  if (problem.topic.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: AxonColors.textPrimary.withValues(alpha: 0.05),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        problem.topic,
                        style: TextStyle(color: AxonColors.textSecondary, fontSize: 12),
                      ),
                    ),
                  const SizedBox(height: 24),
                  Expanded(
                    child: GestureDetector(
                      onTap: () => setState(() => _showAnswer = !_showAnswer),
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(32),
                        decoration: BoxDecoration(
                          color: AxonColors.oxfordBlue,
                          borderRadius: BorderRadius.circular(24),
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            if (!_showAnswer) ...[
                              Text(
                                problem.question,
                                textAlign: TextAlign.center,
                                style: GoogleFonts.inter(
                                  color: AxonColors.textPrimary,
                                  fontSize: 20,
                                ),
                              ),
                              const SizedBox(height: 24),
                              Text(
                                'Tap to reveal',
                                style: TextStyle(color: AxonColors.textTertiary),
                              ),
                            ] else ...[
                              Text(
                                problem.question,
                                style: TextStyle(
                                    color: AxonColors.textSecondary, fontSize: 14),
                              ),
                              const SizedBox(height: 24),
                              Container(
                                  height: 2,
                                  width: 80,
                                  color: Color(0xFFA8E6CF)),
                              const SizedBox(height: 24),
                              Text(
                                problem.answer,
                                textAlign: TextAlign.center,
                                style: GoogleFonts.inter(
                                  color: Color(0xFFA8E6CF),
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              if (problem.explanation != null) ...[
                                const SizedBox(height: 16),
                                Text(
                                  problem.explanation!,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                      color: AxonColors.textSecondary, fontSize: 14),
                                ),
                              ],
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () {
                      HapticFeedback.mediumImpact();
                      setState(() {
                        _currentProblemIndex++;
                        _showAnswer = false;
                      });
                    },
                    style: OutlinedButton.styleFrom(
                      side: BorderSide(
                          color: AxonColors.textPrimary.withValues(alpha: 0.3)),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    child: Text('Skip',
                        style: TextStyle(color: AxonColors.textSecondary)),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      HapticFeedback.mediumImpact();
                      setState(() {
                        _currentProblemIndex++;
                        _showAnswer = false;
                      });
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Color(0xFFA8E6CF),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    child: const Text('Next',
                        style: TextStyle(
                            color: Colors.black, fontWeight: FontWeight.bold)),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final diff = now.difference(date);
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Yesterday';
    return '${diff.inDays} days ago';
  }
}

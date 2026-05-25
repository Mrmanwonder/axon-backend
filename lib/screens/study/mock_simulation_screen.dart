import 'dart:async';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/exam_planner_service.dart';
import '../../services/mock_exam_service.dart';

class MockSimulationScreen extends StatefulWidget {
  final PastPaperPack paper;

  const MockSimulationScreen({
    super.key,
    required this.paper,
  });

  @override
  State<MockSimulationScreen> createState() => _MockSimulationScreenState();
}

class _MockSimulationScreenState extends State<MockSimulationScreen>
    with WidgetsBindingObserver {
  Timer? _timer;
  late int _secondsRemaining;
  int _currentQuestion = 1;
  final List<int> _flaggedQuestions = <int>[];
  bool _isFinishing = false;
  bool _lifecycleViolationCaptured = false;

  int get _questionCount {
    final estimate = (widget.paper.maxMarks / 5).round();
    return estimate.clamp(8, 24);
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _secondsRemaining = widget.paper.duration * 60;
    unawaited(MockExamService.instance.startSession(widget.paper));
    _startTimer();
    HapticFeedback.heavyImpact();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _timer?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_isFinishing) return;
    if (state == AppLifecycleState.paused && !_lifecycleViolationCaptured) {
      _lifecycleViolationCaptured = true;
      unawaited(MockExamService.instance.recordViolation('app left during simulation'));
    } else if (state == AppLifecycleState.resumed) {
      _lifecycleViolationCaptured = false;
    }
  }

  void _startTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted || _isFinishing) return;
      if (_secondsRemaining <= 0) {
        timer.cancel();
        _openReflectionPhase(forceComplete: true);
        return;
      }
      setState(() => _secondsRemaining--);
      if (_secondsRemaining % 60 == 0) {
        unawaited(
          MockExamService.instance.updateCheckpoint(
            currentQuestion: _currentQuestion,
            flaggedQuestions: _flaggedQuestions,
            remainingSeconds: _secondsRemaining,
          ),
        );
      }
    });
  }

  Future<bool> _confirmExit() async {
    if (_isFinishing) return true;
    final shouldLeave = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Abandon simulation?'),
            content: const Text(
              'Leaving now will count as a protocol breach and the paper will be marked abandoned.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(false),
                child: const Text('Stay'),
              ),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(true),
                child: const Text('Abandon'),
              ),
            ],
          ),
        ) ??
        false;

    if (shouldLeave) {
      _timer?.cancel();
      await MockExamService.instance.abandonSession(
        reason: 'student abandoned the paper before submission',
      );
    }
    return shouldLeave;
  }

  String _formatTime(int seconds) {
    final minutes = seconds ~/ 60;
    final secs = seconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
  }

  Future<void> _openReflectionPhase({bool forceComplete = false}) async {
    if (_isFinishing) return;
    _isFinishing = true;
    _timer?.cancel();
    final activeSession = await MockExamService.instance.getActiveSession();
    if (!mounted) return;

    final result = await showModalBottomSheet<MockExamResult>(
      context: context,
      isDismissible: false,
      enableDrag: false,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF0F1117),
      builder: (context) => _ReflectionSheet(
        maxMarks: widget.paper.maxMarks,
        paperTitle: '${widget.paper.subject} ${widget.paper.year} ${widget.paper.variant}',
        flaggedCount: _flaggedQuestions.length,
        violationCount: activeSession?.violationCount ?? 0,
        forceComplete: forceComplete,
      ),
    );

    if (!mounted) return;
    if (result == null) {
      _isFinishing = false;
      return;
    }

    await MockExamService.instance.completeSession(
      score: result.score,
      reflection: result.reflection,
      remainingSeconds: _secondsRemaining,
    );
    if (!mounted) return;
    Navigator.of(context).pop(result);
  }

  void _toggleFlag() {
    setState(() {
      if (_flaggedQuestions.contains(_currentQuestion)) {
        _flaggedQuestions.remove(_currentQuestion);
      } else {
        _flaggedQuestions.add(_currentQuestion);
      }
    });
    HapticFeedback.selectionClick();
    unawaited(
      MockExamService.instance.updateCheckpoint(
        currentQuestion: _currentQuestion,
        flaggedQuestions: _flaggedQuestions,
        remainingSeconds: _secondsRemaining,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final lowTime = _secondsRemaining <= 300;

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        final shouldLeave = await _confirmExit();
        if (shouldLeave && context.mounted) {
          Navigator.of(context).pop();
        }
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        body: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(24, 24, 24, 12),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'SIMULATION ACTIVE',
                            style: GoogleFonts.robotoMono(
                              color: Colors.redAccent,
                              fontSize: 10,
                              letterSpacing: 2,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            '${widget.paper.subject} ${widget.paper.year} ${widget.paper.variant}',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '${widget.paper.maxMarks} marks · ${widget.paper.duration} minutes',
                            style: GoogleFonts.robotoMono(
                              color: Colors.white38,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      decoration: BoxDecoration(
                        color: lowTime
                            ? Colors.red.withValues(alpha: 0.12)
                            : Colors.white.withValues(alpha: 0.05),
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(
                          color: lowTime ? Colors.red : Colors.white24,
                        ),
                      ),
                      child: Text(
                        _formatTime(_secondsRemaining),
                        style: GoogleFonts.robotoMono(
                          color: lowTime ? Colors.red : Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const Divider(color: Colors.white10, height: 1),
              Expanded(
                child: Container(
                  margin: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.02),
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.05),
                    ),
                  ),
                  child: Stack(
                    children: [
                      Positioned.fill(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(24),
                          child: BackdropFilter(
                            filter: ImageFilter.blur(sigmaX: 12, sigmaY: 12),
                            child: const SizedBox.expand(),
                          ),
                        ),
                      ),
                      Center(
                        child: Padding(
                          padding: const EdgeInsets.all(24),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                'QUESTION $_currentQuestion',
                                style: GoogleFonts.robotoMono(
                                  color: const Color(0xFF3A86FF),
                                  fontSize: 12,
                                  letterSpacing: 2,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              const SizedBox(height: 12),
                              const Text(
                                'Secure paper mode is active.\nUse triage below to move through the paper.\nKeep all AI aids disabled until reflection.',
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  color: Colors.white70,
                                  fontSize: 16,
                                  height: 1.5,
                                ),
                              ),
                              const SizedBox(height: 20),
                              Text(
                                _flaggedQuestions.isEmpty
                                    ? 'No questions flagged yet.'
                                    : '${_flaggedQuestions.length} questions flagged for review.',
                                style: GoogleFonts.robotoMono(
                                  color: Colors.white38,
                                  fontSize: 11,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              SizedBox(
                height: 68,
                child: ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  scrollDirection: Axis.horizontal,
                  itemCount: _questionCount,
                  itemBuilder: (context, index) {
                    final question = index + 1;
                    final isCurrent = question == _currentQuestion;
                    final isFlagged = _flaggedQuestions.contains(question);
                    return GestureDetector(
                      onTap: () {
                        setState(() => _currentQuestion = question);
                        HapticFeedback.selectionClick();
                        unawaited(
                          MockExamService.instance.updateCheckpoint(
                            currentQuestion: _currentQuestion,
                            flaggedQuestions: _flaggedQuestions,
                            remainingSeconds: _secondsRemaining,
                          ),
                        );
                      },
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 180),
                        width: 42,
                        margin: const EdgeInsets.symmetric(
                          horizontal: 4,
                          vertical: 10,
                        ),
                        decoration: BoxDecoration(
                          color: isCurrent
                              ? const Color(0xFF3A86FF)
                              : Colors.white.withValues(alpha: 0.05),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: isFlagged ? Colors.orange : Colors.transparent,
                            width: 1.5,
                          ),
                        ),
                        alignment: Alignment.center,
                        child: Text(
                          '$question',
                          style: TextStyle(
                            color: isCurrent ? Colors.white : Colors.white54,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(24, 12, 24, 24),
                child: Row(
                  children: [
                    Expanded(
                      child: _MockActionButton(
                        label: _flaggedQuestions.contains(_currentQuestion)
                            ? 'UNFLAG'
                            : 'FLAG QUESTION',
                        icon: Icons.outlined_flag_rounded,
                        color: Colors.orange,
                        onTap: _toggleFlag,
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _MockActionButton(
                        label: 'SUBMIT PAPER',
                        icon: Icons.check_circle_outline_rounded,
                        color: const Color(0xFF3A86FF),
                        onTap: _openReflectionPhase,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MockActionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _MockActionButton({
    required this.label,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        height: 56,
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withValues(alpha: 0.3)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 8),
            Text(
              label,
              style: GoogleFonts.robotoMono(
                color: color,
                fontSize: 10,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ReflectionSheet extends StatefulWidget {
  final String paperTitle;
  final int maxMarks;
  final int flaggedCount;
  final int violationCount;
  final bool forceComplete;

  const _ReflectionSheet({
    required this.paperTitle,
    required this.maxMarks,
    required this.flaggedCount,
    required this.violationCount,
    required this.forceComplete,
  });

  @override
  State<_ReflectionSheet> createState() => _ReflectionSheetState();
}

class _ReflectionSheetState extends State<_ReflectionSheet> {
  final _scoreController = TextEditingController();
  final _reflectionController = TextEditingController();

  @override
  void dispose() {
    _scoreController.dispose();
    _reflectionController.dispose();
    super.dispose();
  }

  void _submit() {
    final score = int.tryParse(_scoreController.text.trim());
    final reflection = _reflectionController.text.trim();
    if (score == null || score < 0 || score > widget.maxMarks) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Enter a score between 0 and ${widget.maxMarks}.')),
      );
      return;
    }
    if (reflection.length < 12) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Add a short reflection before finishing.')),
      );
      return;
    }
    Navigator.of(context).pop(
      MockExamResult(
        score: score,
        reflection: reflection,
        violationCount: widget.violationCount,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.fromLTRB(
        24,
        24,
        24,
        MediaQuery.of(context).viewInsets.bottom + 24,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            widget.forceComplete ? 'Time expired' : 'Reflection phase',
            style: GoogleFonts.robotoMono(
              color: const Color(0xFF3A86FF),
              fontSize: 11,
              letterSpacing: 1.6,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            widget.paperTitle,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Record your mark now. This closes the simulation and updates your exam signal with a real post-paper outcome.',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.75),
              height: 1.5,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _ReflectionStat(
                label: 'FLAGGED',
                value: '${widget.flaggedCount}',
              ),
              const SizedBox(width: 12),
              _ReflectionStat(
                label: 'PROTOCOL',
                value: widget.violationCount == 0 ? 'Clean' : 'Breach',
              ),
            ],
          ),
          const SizedBox(height: 20),
          TextField(
            controller: _scoreController,
            keyboardType: TextInputType.number,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              labelText: 'Score',
              hintText: '0 - ${widget.maxMarks}',
              filled: true,
              fillColor: Colors.white.withValues(alpha: 0.05),
              labelStyle: const TextStyle(color: Colors.white54),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide.none,
              ),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _reflectionController,
            maxLines: 3,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              labelText: 'Reflection',
              hintText: 'Where did you leak marks, and what gets drilled next?',
              filled: true,
              fillColor: Colors.white.withValues(alpha: 0.05),
              labelStyle: const TextStyle(color: Colors.white54),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide.none,
              ),
            ),
          ),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: _submit,
              child: const Text('Lock Result'),
            ),
          ),
        ],
      ),
    );
  }
}

class _ReflectionStat extends StatelessWidget {
  final String label;
  final String value;

  const _ReflectionStat({
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.04),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.white12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              label,
              style: GoogleFonts.robotoMono(
                color: Colors.white38,
                fontSize: 10,
                letterSpacing: 1.4,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

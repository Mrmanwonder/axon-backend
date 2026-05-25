// lib/screens/study/flash_cards_screen.dart
// ─────────────────────────────────────────────────────────────────
// Flash Cards Screen - Separate from Active Recall
// ─────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../services/app_state.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';

class FlashCardsScreen extends ConsumerStatefulWidget {
  final String subject;
  
  const FlashCardsScreen({super.key, this.subject = ''});

  @override
  ConsumerState<FlashCardsScreen> createState() => _FlashCardsScreenState();
}

class _FlashCardsScreenState extends ConsumerState<FlashCardsScreen> {
  late List<FlashCard> _cards;
  int _currentIndex = 0;
  bool _showAnswer = false;
  List<String> _subjects = [];
  bool _isLoading = true;
  
  @override
  void initState() {
    super.initState();
    _loadData();
  }
  
  Future<void> _loadData() async {
    try {
      final auth = ref.read(authStateProvider);
      final board = auth.user?.board ?? 'CAIE IGCSE';
      _subjects = await SubjectCache.getSubjects(board);
      
      final selectedSubject = widget.subject.isNotEmpty 
          ? widget.subject 
          : (_subjects.isNotEmpty ? _subjects.first : '');
      
      if (selectedSubject.isNotEmpty) {
        await _loadCardsForSubject(selectedSubject);
      }
    } catch (e) {
      debugPrint('FlashCardsScreen: Error loading data: $e');
    }
    if (mounted) {
      setState(() => _isLoading = false);
    }
  }
  
  Future<void> _loadCardsForSubject(String subject) async {
    try {
      final chapters = await SubjectCache.getChapters('CAIE IGCSE', subject);
      _cards = chapters.map((chapter) => FlashCard(
        front: chapter,
        back: 'Review this chapter: $chapter',
        topic: chapter,
        chapter: subject,
      )).toList();
      
      _cards.shuffle();
    } catch (e) {
      _cards = [];
    }
  }
  
  void _nextCard() {
    if (_currentIndex < _cards.length - 1) {
      setState(() {
        _currentIndex++;
        _showAnswer = false;
      });
    }
  }
  
  void _prevCard() {
    if (_currentIndex > 0) {
      setState(() {
        _currentIndex--;
        _showAnswer = false;
      });
    }
  }
  
  void _flipCard() {
    setState(() {
      _showAnswer = !_showAnswer;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Flash Cards', style: GoogleFonts.poppins()),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => popOrGo(context, '/study'),
        ),
        actions: [
          if (_subjects.length > 1)
            PopupMenuButton<String>(
              onSelected: (subject) => _loadCardsForSubject(subject).then((_) {
                if (mounted) setState(() {});
              }),
              itemBuilder: (context) => _subjects
                  .map((s) => PopupMenuItem(value: s, child: Text(s)))
                  .toList(),
              child: Padding(
                padding: const EdgeInsets.all(8),
                child: Row(
                  children: [
                    Text(widget.subject.isEmpty ? (_subjects.first) : widget.subject),
                    const Icon(Icons.arrow_drop_down),
                  ],
                ),
              ),
            ),
          TextButton(
            onPressed: () => setState(() => _cards.shuffle()),
            child: const Text('Shuffle'),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _cards.isEmpty 
              ? Center(
                  child: Text(
                    'No flash cards available',
                    style: GoogleFonts.poppins(),
                  ),
                )
              : Column(
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(16),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            widget.subject.isEmpty ? _subjects.first : widget.subject,
                            style: GoogleFonts.poppins(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            '${_currentIndex + 1} / ${_cards.length}',
                            style: GoogleFonts.poppins(
                              color: Colors.grey,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Expanded(
                      child: GestureDetector(
                        onTap: _flipCard,
                        child: Container(
                          margin: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                AxonColors.accent.withAlpha(50),
                                AxonColors.electricCyan.withAlpha(50),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(24),
                            boxShadow: [
                              BoxShadow(
                                color: AxonColors.accent.withAlpha(30),
                                blurRadius: 20,
                                offset: const Offset(0, 10),
                              ),
                            ],
                          ),
                          child: Center(
                            child: Padding(
                              padding: const EdgeInsets.all(32),
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    _showAnswer ? 'ANSWER' : 'QUESTION',
                                    style: GoogleFonts.poppins(
                                      fontSize: 12,
                                      fontWeight: FontWeight.bold,
                                      color: AxonColors.accent,
                                      letterSpacing: 2,
                                    ),
                                  ),
                                  const SizedBox(height: 24),
                                  Text(
                                    _showAnswer 
                                        ? _cards[_currentIndex].back
                                        : _cards[_currentIndex].front,
                                    style: GoogleFonts.poppins(
                                      fontSize: 22,
                                      fontWeight: FontWeight.w600,
                                    ),
                                    textAlign: TextAlign.center,
                                  ),
                                  const SizedBox(height: 24),
                                  Text(
                                    _cards[_currentIndex].chapter,
                                    style: GoogleFonts.poppins(
                                      fontSize: 14,
                                      color: Colors.grey,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(24),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                        children: [
                          IconButton(
                            onPressed: _currentIndex > 0 ? _prevCard : null,
                            icon: const Icon(Icons.arrow_back_ios),
                            iconSize: 32,
                          ),
                          ElevatedButton.icon(
                            onPressed: _flipCard,
                            icon: const Icon(Icons.flip),
                            label: Text(_showAnswer ? 'Show Question' : 'Show Answer'),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 24,
                                vertical: 12,
                              ),
                            ),
                          ),
                          IconButton(
                            onPressed: _currentIndex < _cards.length - 1 ? _nextCard : null,
                            icon: const Icon(Icons.arrow_forward_ios),
                            iconSize: 32,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
    );
  }
}

class FlashCard {
  final String front;
  final String back;
  final String topic;
  final String chapter;
  
  FlashCard({
    required this.front,
    required this.back,
    required this.topic,
    required this.chapter,
  });
}
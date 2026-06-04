import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_pdfviewer/pdfviewer.dart';
import 'package:google_fonts/google_fonts.dart';

class PdfViewerScreen extends ConsumerStatefulWidget {
  final String? url;
  final String? filePath;
  final String? title;

  const PdfViewerScreen({
    super.key,
    this.url,
    this.filePath,
    this.title,
  });

  @override
  ConsumerState<PdfViewerScreen> createState() => _PdfViewerScreenState();
}

class _PdfViewerScreenState extends ConsumerState<PdfViewerScreen> {
  late final PdfViewerController _controller;
  bool _nativeMode = false;
  List<Map<String, dynamic>> _questions = [];
  int _currentQuestion = 0;

  @override
  void initState() {
    super.initState();
    _controller = PdfViewerController();
    _loadQuestions();
  }

  Future<void> _loadQuestions() async {
    try {
      final source = widget.filePath ?? widget.url;
      if (source == null) return;

      String fileName = source.split('/').last.replaceAll('.pdf', '');
      if (widget.title != null) {
        fileName = widget.title!.replaceAll('.pdf', '');
      }

      final appDir = Platform.environment['APPDATA'] ?? 
          Platform.environment['HOME'] ?? '';
      final baseDir = appDir.contains('Roaming') 
          ? appDir.replaceAll('Roaming', 'Axon/axon')
          : '$appDir/../Axon/axon';
      final jsonPath = '$baseDir/extracted_v18/$fileName.json';
      
      final jsonFile = File(jsonPath);
      if (jsonFile.existsSync()) {
        final content = jsonDecode(jsonFile.readAsStringSync());
        setState(() {
          _questions = List<Map<String, dynamic>>.from(content['questions'] ?? []);
        });
      }
    } catch (e) {
      debugPrint('Error loading questions: $e');
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF000000),
      appBar: AppBar(
        backgroundColor: const Color(0xFF0A0A0A),
        foregroundColor: Colors.white,
        title: Text(widget.title ?? 'PDF Viewer'),
        elevation: 0,
        actions: [
          if (_questions.isNotEmpty) ...[
            Text(
              _nativeMode ? 'Questions' : 'PDF',
              style: GoogleFonts.robotoMono(
                color: Colors.white54,
                fontSize: 12,
              ),
            ),
            const SizedBox(width: 8),
            Switch(
              value: _nativeMode,
              onChanged: (value) {
                setState(() => _nativeMode = value);
                HapticFeedback.selectionClick();
              },
              activeTrackColor: const Color(0xFF3A86FF),
            ),
          ],
          const SizedBox(width: 8),
        ],
      ),
      body: _nativeMode && _questions.isNotEmpty 
          ? _buildNativeView() 
          : _buildViewer(),
    );
  }

  Widget _buildNativeView() {
    if (_questions.isEmpty) {
      return const Center(
        child: Text(
          'No questions extracted for this paper',
          style: TextStyle(color: Colors.white54),
        ),
      );
    }

    final q = _questions[_currentQuestion];
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          color: const Color(0xFF1A1A1A),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back_ios, color: Colors.white54, size: 20),
                    onPressed: _currentQuestion > 0 
                        ? () => setState(() => _currentQuestion--) 
                        : null,
                  ),
                  Text(
                    '${_currentQuestion + 1}/${_questions.length}',
                    style: GoogleFonts.robotoMono(color: Colors.white70),
                  ),
                  IconButton(
                    icon: const Icon(Icons.arrow_forward_ios, color: Colors.white54, size: 20),
                    onPressed: _currentQuestion < _questions.length - 1 
                        ? () => setState(() => _currentQuestion++) 
                        : null,
                  ),
                ],
              ),
              if (q['marks'] != null)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFF3A86FF).withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '[${q['marks']}]',
                    style: GoogleFonts.robotoMono(
                      color: const Color(0xFF3A86FF),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
            ],
          ),
        ),
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Question ${q['number']}',
                    style: GoogleFonts.robotoMono(
                      color: const Color(0xFF3A86FF),
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(
                    q['text'] ?? '',
                    style: GoogleFonts.inter(color: Colors.white, fontSize: 14),
                  ),
                  if (q['parts'] != null && (q['parts'] as List).isNotEmpty) ...[
                    const SizedBox(height: 24),
                    const Divider(color: Colors.white24),
                    const SizedBox(height: 16),
                    ...(q['parts'] as List).map((p) => _buildPart(p)),
                  ],
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildPart(Map<String, dynamic> part) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  '(${part['id']})',
                  style: GoogleFonts.robotoMono(color: Colors.white70),
                ),
              ),
              if (part['marks'] != null) ...[
                const SizedBox(width: 8),
                Text(
                  '[${part['marks']}]',
                  style: GoogleFonts.robotoMono(
                    color: Colors.white38,
                    fontSize: 12,
                  ),
                ),
              ],
            ],
          ),
          if (part['text'] != null && (part['text'] as String).isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              part['text'],
              style: GoogleFonts.inter(
                color: Colors.white.withValues(alpha: 0.8),
                fontSize: 13,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildViewer() {
    final source = widget.filePath ?? widget.url;
    if (source == null || source.isEmpty) {
      return const Center(
        child: Text(
          'No PDF source provided',
          style: TextStyle(color: Colors.white54),
        ),
      );
    }

    if (source.startsWith('http')) {
      return SfPdfViewer.network(
        source,
        controller: _controller,
        canShowScrollHead: false,
        pageSpacing: 8,
      );
    } else {
      final file = File(source);
      if (!file.existsSync()) {
        return const Center(
          child: Text(
            'File not found',
            style: TextStyle(color: Colors.white54),
          ),
        );
      }
      return SfPdfViewer.file(
        file,
        controller: _controller,
        canShowScrollHead: false,
        pageSpacing: 8,
      );
    }
  }
}
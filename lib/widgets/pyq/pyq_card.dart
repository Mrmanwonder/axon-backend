import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../services/local_pyqs_service.dart';
import '../../services/pyq/pyq_session_service.dart';
import '../../theme/app_theme.dart';
import '../../theme/subject_themes.dart';
import 'segmented_progress_bar.dart';

class PyqCard extends StatefulWidget {
  final String subjectCode;
  final String subjectName;
  final SubjectTheme theme;

  const PyqCard({
    super.key,
    required this.subjectCode,
    required this.subjectName,
    required this.theme,
  });

  @override
  State<PyqCard> createState() => _PyqCardState();
}

class _PyqCardState extends State<PyqCard> {
  PyqSession? _lastSession;
  double _progress = 0.0;
  int _remaining = 0;
  int _totalYears = 0;
  int _totalPapers = 0;
  bool _loaded = false;

  List<PastPaperRecord>? _cachedPapers;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<List<PastPaperRecord>> _getSubjectPapers() async {
    if (_cachedPapers != null) return _cachedPapers!;

    debugPrint('PYQ Card: Loading papers for ${widget.subjectName} (${widget.subjectCode})');
    
    var papers = await LocalPyqsService.instance.getPapersBySubject(widget.subjectCode);
    debugPrint('PYQ Card: Method 1 - getPapersBySubject: ${papers.length} papers');
    
    if (papers.isEmpty) {
      debugPrint('PYQ Card: Method 2 - getAllPapers and filter');
      final allPapers = await LocalPyqsService.instance.getAllPapers();
      debugPrint('PYQ Card: Total papers in CSV: ${allPapers.length}');
      if (allPapers.isNotEmpty) {
        debugPrint('PYQ Card: Sample CSV entry - code: "${allPapers.first.subjectCode}", name: "${allPapers.first.subject}"');
      }
      
      papers = allPapers.where((p) {
        final codeMatch = p.subjectCode.toLowerCase() == widget.subjectCode.toLowerCase();
        final nameMatch = p.subject.toLowerCase().contains(widget.subjectName.toLowerCase()) ||
            widget.subjectName.toLowerCase().contains(p.subject.toLowerCase());
        return codeMatch || nameMatch;
      }).toList();
      debugPrint('PYQ Card: Method 2 filtered: ${papers.length} papers');
    }
    
    if (papers.isEmpty) {
      debugPrint('PYQ Card: Method 3 - searchByCurriculum');
      final curriculumPapers = await LocalPyqsService.instance.searchByCurriculum('A Levels');
      papers = curriculumPapers.where((p) {
        final codeMatch = p.subjectCode.toLowerCase() == widget.subjectCode.toLowerCase();
        final nameMatch = p.subject.toLowerCase().contains(widget.subjectName.toLowerCase()) ||
            widget.subjectName.toLowerCase().contains(p.subject.toLowerCase());
        return codeMatch || nameMatch;
      }).toList();
      debugPrint('PYQ Card: Method 3 filtered: ${papers.length} papers');
    }
    
    _cachedPapers = papers;
    return papers;
  }

  Future<void> _loadData() async {
    final session = await PyqSessionService.instance.getSession(widget.subjectCode);
    final progress = await PyqSessionService.instance.getProgress(widget.subjectCode);
    final remaining = await PyqSessionService.instance.getRemainingPapers(widget.subjectCode);

    final papers = await _getSubjectPapers();
    final years = papers.map((p) => p.year).where((y) => y.isNotEmpty).toSet().toList()
      ..sort((a, b) => b.compareTo(a));

    debugPrint('PYQ Card: Final - ${papers.length} papers, ${years.length} years: $years');

    if (mounted) {
      setState(() {
        _lastSession = session;
        _progress = progress;
        _remaining = remaining;
        _totalYears = years.length;
        _totalPapers = papers.length;
        _loaded = true;
      });
    }
  }

  void _navigateToHub() {
    HapticFeedback.mediumImpact();
    context.push(
      '/study/pyq-hub',
      extra: {
        'subjectCode': widget.subjectCode,
        'subjectName': widget.subjectName,
      },
    ).then((_) => _loadData());
  }

  void _resumeSession() {
    if (_lastSession == null) {
      _navigateToHub();
      return;
    }
    HapticFeedback.mediumImpact();
    context.push(
      '/study/pyq-hub',
      extra: {
        'subjectCode': widget.subjectCode,
        'subjectName': widget.subjectName,
        'resumeYear': _lastSession!.year,
        'resumeSeries': _lastSession!.series,
        'resumeVariant': _lastSession!.paperVariant,
      },
    ).then((_) => _loadData());
  }

  @override
  Widget build(BuildContext context) {
    if (!_loaded) {
      return _buildSkeleton();
    }

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: widget.theme.primary.withValues(alpha: 0.15)),
        boxShadow: [
          BoxShadow(
            color: widget.theme.primary.withValues(alpha: 0.08),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        children: [
          _buildTopZone(),
          Divider(height: 1, color: widget.theme.primary.withValues(alpha: 0.1)),
          _buildBottomZone(),
        ],
      ),
    );
  }

  Widget _buildTopZone() {
    return InkWell(
      onTap: _navigateToHub,
      borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Container(
                      width: 32,
                      height: 32,
                      decoration: BoxDecoration(
                        color: widget.theme.primary.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Icon(
                        Icons.description_outlined,
                        color: widget.theme.primary,
                        size: 18,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'PYQs',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ],
                ),
                Icon(
                  Icons.arrow_forward_rounded,
                  color: widget.theme.primary.withValues(alpha: 0.6),
                  size: 18,
                ),
              ],
            ),
            const SizedBox(height: 16),
            SegmentedProgressBar(
              progress: _progress,
              segments: _totalYears > 0 ? _totalYears : 6,
              activeColor: widget.theme.primary,
              remainingText: _remaining > 0 ? '$_remaining left' : 'Complete',
            ),
            if (_totalYears > 0) ...[
              const SizedBox(height: 8),
              Text(
                '$_totalYears years • $_totalPapers papers',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 10,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildBottomZone() {
    return InkWell(
      onTap: _resumeSession,
      borderRadius: const BorderRadius.vertical(bottom: Radius.circular(28)),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: widget.theme.primary.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(
                Icons.play_arrow_rounded,
                color: widget.theme.primary,
                size: 22,
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _lastSession != null
                        ? '${widget.subjectName} ${_lastSession!.subjectCode}/${_lastSession!.paperVariant}'
                        : 'Start your first PYQ',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    _lastSession != null
                        ? '${_getSeriesName(_lastSession!.series)} ${_lastSession!.year} • ${(_lastSession!.progress * 100).toInt()}% complete'
                        : 'Tap to begin',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _getSeriesName(String series) {
    switch (series.toLowerCase()) {
      case 'm':
      case 'm_j':
      case 'mj':
      case 'may/jun':
      case 'may/june':
        return 'May/June';
      case 'o':
      case 'o_n':
      case 'on':
      case 'oct/nov':
      case 'oct/november':
        return 'Oct/Nov';
      case 'f':
      case 'f_m':
      case 'fm':
      case 'feb/mar':
      case 'feb/march':
        return 'Feb/March';
      default:
        return series;
    }
  }

  Widget _buildSkeleton() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      height: 140,
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: Center(
        child: SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            valueColor: AlwaysStoppedAnimation<Color>(widget.theme.primary),
          ),
        ),
      ),
    );
  }
}

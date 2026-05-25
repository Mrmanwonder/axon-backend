import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme/app_theme.dart';

class SyllabusChapter {
  final String id;
  final String name;
  final int index;
  final double readinessLevel;
  final String? pdfPath;

  SyllabusChapter({
    required this.id,
    required this.name,
    required this.index,
    this.readinessLevel = 0.0,
    this.pdfPath,
  });
}

class SyllabusCompletionHeatmap extends StatelessWidget {
  final List<SyllabusChapter> chapters;
  final Function(SyllabusChapter)? onChapterTap;
  final int columns;

  const SyllabusCompletionHeatmap({
    super.key,
    required this.chapters,
    this.onChapterTap,
    this.columns = 10,
  });

  @override
  Widget build(BuildContext context) {
    if (chapters.isEmpty) {
      return Center(
        child: Text(
          'No syllabus data available',
          style: TextStyle(color: Colors.white.withValues(alpha: 0.5)),
        ),
      );
    }

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: columns,
                  crossAxisSpacing: 4,
                  mainAxisSpacing: 4,
                  childAspectRatio: 1,
                ),
                itemCount: chapters.length,
                itemBuilder: (context, index) {
                  final chapter = chapters[index];
                  return _HeatmapCell(
                    chapter: chapter,
                    onTap: () {
                      HapticFeedback.selectionClick();
                      onChapterTap?.call(chapter);
                    },
                  );
                },
              ),
              const SizedBox(height: 16),
              _buildLegend(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLegend() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Less',
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.5),
            fontSize: 10,
          ),
        ),
        const SizedBox(width: 4),
        _legendCell(0.0),
        _legendCell(0.25),
        _legendCell(0.5),
        _legendCell(0.75),
        _legendCell(1.0),
        const SizedBox(width: 4),
        Text(
          'Mastered',
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.5),
            fontSize: 10,
          ),
        ),
      ],
    );
  }

  Widget _legendCell(double readiness) {
    return Container(
      width: 12,
      height: 12,
      margin: const EdgeInsets.symmetric(horizontal: 2),
      decoration: BoxDecoration(
        color: _getCellColor(readiness),
        borderRadius: BorderRadius.circular(2),
      ),
    );
  }

  Color _getCellColor(double readiness) {
    if (readiness == 0) {
      return const Color(0xFF1A1A1A);
    }
    return Color.lerp(
      const Color(0xFF1A1A1A),
      const Color(0xFF00E676),
      readiness,
    )!;
  }
}

class _HeatmapCell extends StatelessWidget {
  final SyllabusChapter chapter;
  final VoidCallback? onTap;

  const _HeatmapCell({
    required this.chapter,
    this.onTap,
  });

  Color get _cellColor {
    if (chapter.readinessLevel == 0) {
      return const Color(0xFF1A1A1A);
    }
    return Color.lerp(
      const Color(0xFF1A1A1A),
      const Color(0xFF00E676),
      chapter.readinessLevel.clamp(0.0, 1.0),
    )!;
  }

  bool get _showGlow => chapter.readinessLevel > 0.8;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 28,
        height: 28,
        decoration: BoxDecoration(
          color: _cellColor,
          borderRadius: BorderRadius.circular(4),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.1),
            width: 0.5,
          ),
          boxShadow: _showGlow
              ? [
                  BoxShadow(
                    color: const Color(0xFF00E676).withValues(alpha: 0.4),
                    blurRadius: 8,
                    spreadRadius: 2,
                  ),
                ]
              : null,
        ),
        child: Tooltip(
          message:
              '${chapter.name}\n${(chapter.readinessLevel * 100).toInt()}% ready',
          child: const SizedBox.expand(),
        ),
      ),
    );
  }
}

class MF19MasteryGauge extends StatelessWidget {
  final int memorizedCount;
  final int appliedCount;
  final int masteredCount;
  final int totalFormulas;

  const MF19MasteryGauge({
    super.key,
    required this.memorizedCount,
    required this.appliedCount,
    required this.masteredCount,
    required this.totalFormulas,
  });

  @override
  Widget build(BuildContext context) {
    final total = memorizedCount + appliedCount + masteredCount;
    final progress = totalFormulas > 0 ? total / totalFormulas : 0.0;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF121212),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.1),
          width: 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'MF19 Mastery',
            style: GoogleFonts.inter(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            height: 8,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: progress.clamp(0.0, 1.0),
                backgroundColor: const Color(0xFF222222),
                valueColor: AlwaysStoppedAnimation(AxonColors.electricCyan),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _GaugeMarker(
                label: 'Memorized',
                count: memorizedCount,
                color: Colors.orange,
              ),
              _GaugeMarker(
                label: 'Applied',
                count: appliedCount,
                color: Colors.blue,
              ),
              _GaugeMarker(
                label: 'Mastered',
                count: masteredCount,
                color: const Color(0xFF00E676),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _GaugeMarker extends StatelessWidget {
  final String label;
  final int count;
  final Color color;

  const _GaugeMarker({
    required this.label,
    required this.count,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: color.withValues(alpha: 0.4),
                blurRadius: 4,
              ),
            ],
          ),
        ),
        const SizedBox(height: 4),
        Text(
          count.toString(),
          style: TextStyle(
            color: color,
            fontSize: 16,
            fontWeight: FontWeight.w700,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.5),
            fontSize: 10,
          ),
        ),
      ],
    );
  }
}

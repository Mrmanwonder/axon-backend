import 'package:flutter/material.dart';
import '../models/paper.dart';
import '../models/paper_series.dart';

class FolderPainter extends CustomPainter {
  final PaperSeries series;
  final bool isCenter;
  final bool isOpen;
  final double depth; // 0.4 (edge) → 1.0 (center)

  static const double kTabWidth = 72.0;
  static const double kTabHeight = 26.0;
  static const double kTabRadius = 6.0;
  static const double kBodyRadius = 10.0;

  const FolderPainter({
    required this.series,
    required this.isCenter,
    required this.isOpen,
    required this.depth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final bodyTop = kTabHeight - 1.0;

    _drawShadow(canvas, size, bodyTop);
    _drawPeekingFileTabs(canvas, size, bodyTop);
    _drawTab(canvas, bodyTop);
    _drawBody(canvas, size, bodyTop);
    _drawBodyTopStrips(canvas, size, bodyTop);
    _drawLabels(canvas, size, bodyTop);
    _drawSubjectDots(canvas, size);
    if (isOpen) _drawOpenGlow(canvas, size, bodyTop);
  }

  // ── 1. Drop shadow ─────────────────────────────────────────────────────────
  void _drawShadow(Canvas canvas, Size size, double bodyTop) {
    final shadowPaint = Paint()
      ..color = Colors.black.withOpacity(0.55 * depth)
      ..maskFilter = MaskFilter.blur(BlurStyle.normal, 22 * depth);
    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(10, bodyTop + 14, size.width - 20, size.height - bodyTop),
        const Radius.circular(kBodyRadius),
      ),
      shadowPaint,
    );
  }

  // ── 2. Peeking file tabs (above body, right of folder tab) ─────────────────
  void _drawPeekingFileTabs(Canvas canvas, Size size, double bodyTop) {
    final startX = kTabWidth + 6;
    final availW = size.width - startX - 2;
    if (availW <= 0) return;

    final papersBySubject = _groupBySubject();
    if (papersBySubject.isEmpty) return;

    final total = series.papers.length.toDouble();
    double x = startX;
    const peekTop = 12.0;
    const peekH = kTabHeight - 12.0 - 1.0;

    for (final entry in papersBySubject.entries) {
      final proportion = entry.value / total;
      final w = proportion * availW;
      if (w < 1) { x += w; continue; }

      canvas.drawRRect(
        RRect.fromRectAndCorners(
          Rect.fromLTWH(x, peekTop, w - 1.5, peekH),
          topLeft: const Radius.circular(3),
          topRight: const Radius.circular(3),
        ),
        Paint()..color = entry.key.color.withOpacity(0.75 * depth),
      );
      x += w;
    }
  }

  // ── 3. Folder tab (top-left label protrusion) ──────────────────────────────
  void _drawTab(Canvas canvas, double bodyTop) {
    final tabColor = _lerpDepthColor(
      const Color(0xFF1E1E2D),
      const Color(0xFF282836),
    );

    final tabPath = Path()
      ..moveTo(0, bodyTop)
      ..lineTo(0, kTabRadius)
      ..arcToPoint(const Offset(kTabRadius, 0),
          radius: const Radius.circular(kTabRadius))
      ..lineTo(kTabWidth - kTabRadius, 0)
      ..arcToPoint(Offset(kTabWidth, kTabRadius),
          radius: const Radius.circular(kTabRadius))
      ..lineTo(kTabWidth, bodyTop)
      ..close();

    canvas.drawPath(tabPath, Paint()..color = tabColor);

    // Subtle tab top highlight
    canvas.drawLine(
      const Offset(kTabRadius, 0.5),
      Offset(kTabWidth - kTabRadius, 0.5),
      Paint()
        ..color = Colors.white.withOpacity(0.07 * depth)
        ..strokeWidth = 1,
    );
  }

  // ── 4. Folder body ─────────────────────────────────────────────────────────
  void _drawBody(Canvas canvas, Size size, double bodyTop) {
    final bodyRect = RRect.fromRectAndCorners(
      Rect.fromLTWH(0, bodyTop, size.width, size.height - bodyTop),
      topLeft: Radius.zero,
      topRight: const Radius.circular(kBodyRadius),
      bottomLeft: const Radius.circular(kBodyRadius),
      bottomRight: const Radius.circular(kBodyRadius),
    );

    // Gradient body
    final topColor = _lerpDepthColor(
      const Color(0xFF1A1A27),
      const Color(0xFF23232F),
    );
    final botColor = _lerpDepthColor(
      const Color(0xFF141420),
      const Color(0xFF1C1C28),
    );

    final shader = LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [topColor, botColor],
    ).createShader(
      Rect.fromLTWH(0, bodyTop, size.width, size.height - bodyTop),
    );

    canvas.drawRRect(bodyRect, Paint()..shader = shader);

    // Top inner edge highlight
    canvas.drawLine(
      Offset(kTabWidth, bodyTop + 0.5),
      Offset(size.width, bodyTop + 0.5),
      Paint()
        ..color = Colors.white.withOpacity(0.05 * depth)
        ..strokeWidth = 1,
    );

    // Left edge line
    canvas.drawLine(
      Offset(0, bodyTop),
      Offset(0, size.height - kBodyRadius),
      Paint()
        ..color = Colors.white.withOpacity(0.04 * depth)
        ..strokeWidth = 1,
    );

    // Right edge line
    canvas.drawLine(
      Offset(size.width, bodyTop),
      Offset(size.width, size.height - kBodyRadius),
      Paint()
        ..color = Colors.white.withOpacity(0.04 * depth)
        ..strokeWidth = 1,
    );
  }

  // ── 5. Colored strips at top of body (inside folder) ──────────────────────
  void _drawBodyTopStrips(Canvas canvas, Size size, double bodyTop) {
    final papersBySubject = _groupBySubject();
    if (papersBySubject.isEmpty) return;

    final total = series.papers.length.toDouble();
    const stripH = 3.5;
    const stripY = kTabHeight + 1.0;
    double x = 1;
    final availW = size.width - 2;

    for (final entry in papersBySubject.entries) {
      final w = (entry.value / total) * availW;
      if (w < 1) { x += w; continue; }
      canvas.drawRect(
        Rect.fromLTWH(x, stripY, w - 0.5, stripH),
        Paint()..color = entry.key.color.withOpacity(0.6 * depth),
      );
      x += w;
    }
  }

  // ── 6. Labels — series ID on tab, paper count on body ─────────────────────
  void _drawLabels(Canvas canvas, Size size, double bodyTop) {
    // Series short label on the tab
    _drawText(
      canvas,
      text: series.shortLabel,
      offset: const Offset(10, 6),
      fontSize: 10.5,
      fontWeight: FontWeight.w700,
      color: Colors.white.withOpacity(0.82 * depth),
      letterSpacing: 1.4,
    );

    // Paper count badge — top right of body
    _drawText(
      canvas,
      text: '×${series.paperCount}',
      offset: Offset(size.width - 36, bodyTop + 8),
      fontSize: 9.5,
      fontWeight: FontWeight.w500,
      color: Colors.white.withOpacity(0.28 * depth),
      letterSpacing: 0.5,
    );

    // Year label — bottom center of body
    _drawText(
      canvas,
      text: '${series.year}',
      offset: Offset(size.width / 2 - 14, size.height - 20),
      fontSize: 10,
      fontWeight: FontWeight.w400,
      color: Colors.white.withOpacity(0.22 * depth),
      letterSpacing: 1,
    );
  }

  // ── 7. Subject colour dots — bottom of body ────────────────────────────────
  void _drawSubjectDots(Canvas canvas, Size size) {
    final subjects = _groupBySubject().keys.toList();
    if (subjects.isEmpty) return;

    const r = 2.8;
    const gap = 6.0;
    final totalW = subjects.length * r * 2 + (subjects.length - 1) * gap;
    double x = (size.width - totalW) / 2;
    final y = size.height - 10.0;

    for (final s in subjects) {
      canvas.drawCircle(
        Offset(x + r, y),
        r,
        Paint()..color = s.color.withOpacity(0.72 * depth),
      );
      x += r * 2 + gap;
    }
  }

  // ── 8. Open state glow / border ────────────────────────────────────────────
  void _drawOpenGlow(Canvas canvas, Size size, double bodyTop) {
    const accent = Color(0xFFC9AA71);

    final bodyRect = RRect.fromRectAndCorners(
      Rect.fromLTWH(0, bodyTop, size.width, size.height - bodyTop),
      topLeft: Radius.zero,
      topRight: const Radius.circular(kBodyRadius),
      bottomLeft: const Radius.circular(kBodyRadius),
      bottomRight: const Radius.circular(kBodyRadius),
    );

    // Outer glow
    canvas.drawRRect(
      bodyRect,
      Paint()
        ..color = accent.withOpacity(0.22)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 14),
    );
    // Crisp border
    canvas.drawRRect(
      bodyRect,
      Paint()
        ..color = accent.withOpacity(0.65)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.4,
    );
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  Map<SubjectGroup, int> _groupBySubject() {
    final map = <SubjectGroup, int>{};
    for (final p in series.papers) {
      map[p.subjectGroup] = (map[p.subjectGroup] ?? 0) + 1;
    }
    return map;
  }

  Color _lerpDepthColor(Color dark, Color bright) =>
      Color.lerp(dark, bright, (depth - 0.4) / 0.6)!;

  void _drawText(
    Canvas canvas, {
    required String text,
    required Offset offset,
    required double fontSize,
    required FontWeight fontWeight,
    required Color color,
    double letterSpacing = 0,
  }) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          fontFamily: 'monospace',
          fontSize: fontSize,
          fontWeight: fontWeight,
          color: color,
          letterSpacing: letterSpacing,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: double.infinity);
    tp.paint(canvas, offset);
  }

  @override
  bool shouldRepaint(FolderPainter old) =>
      old.isOpen != isOpen ||
      old.isCenter != isCenter ||
      old.depth != depth ||
      old.series.id != series.id;
}

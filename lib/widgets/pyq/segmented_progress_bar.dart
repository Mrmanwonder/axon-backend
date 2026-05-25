import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

class SegmentedProgressBar extends StatelessWidget {
  final double progress;
  final int segments;
  final Color? activeColor;
  final Color? inactiveColor;
  final String? remainingText;

  const SegmentedProgressBar({
    super.key,
    required this.progress,
    this.segments = 8,
    this.activeColor,
    this.inactiveColor,
    this.remainingText,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: SizedBox(
            height: 8,
            child: CustomPaint(
              painter: _SegmentedBarPainter(
                progress: progress,
                segments: segments,
                activeColor: activeColor ?? AxonColors.accent,
                inactiveColor: inactiveColor ?? Colors.white.withValues(alpha: 0.08),
              ),
            ),
          ),
        ),
        if (remainingText != null) ...[
          const SizedBox(width: 12),
          Text(
            remainingText!,
            style: TextStyle(
              color: AxonColors.textTertiary,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ],
    );
  }
}

class _SegmentedBarPainter extends CustomPainter {
  final double progress;
  final int segments;
  final Color activeColor;
  final Color inactiveColor;

  _SegmentedBarPainter({
    required this.progress,
    required this.segments,
    required this.activeColor,
    required this.inactiveColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final barHeight = size.height;
    final gap = 2.0;
    final totalGaps = (segments - 1) * gap;
    final segmentWidth = (size.width - totalGaps) / segments;
    final radius = barHeight / 2;

    final activeSegments = (progress * segments).clamp(0, segments).toInt();
    final partialProgress = (progress * segments) - activeSegments;

    for (int i = 0; i < segments; i++) {
      final left = i * (segmentWidth + gap);
      final rect = RRect.fromRectAndRadius(
        Rect.fromLTWH(left, 0, segmentWidth, barHeight),
        Radius.circular(radius),
      );

      final paint = Paint()..style = PaintingStyle.fill;

      if (i < activeSegments) {
        paint.color = activeColor;
      } else if (i == activeSegments && partialProgress > 0) {
        paint.color = Color.lerp(inactiveColor, activeColor, partialProgress)!;
      } else {
        paint.color = inactiveColor;
      }

      canvas.drawRRect(rect, paint);
    }

    if (progress > 0 && progress < 1) {
      final indicatorX = progress * size.width;
      final indicatorPaint = Paint()
        ..color = activeColor.withValues(alpha: 0.8)
        ..style = PaintingStyle.fill;

      final indicatorRect = RRect.fromRectAndRadius(
        Rect.fromCenter(
          center: Offset(indicatorX, barHeight / 2),
          width: 2,
          height: barHeight + 4,
        ),
        const Radius.circular(1),
      );

      canvas.drawRRect(indicatorRect, indicatorPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _SegmentedBarPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.segments != segments ||
        oldDelegate.activeColor != activeColor ||
        oldDelegate.inactiveColor != inactiveColor;
  }
}

import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../theme/app_theme.dart';

class VanishCountdownRing extends StatefulWidget {
  final Duration totalDuration;
  final Duration remainingDuration;
  final bool isRunning;
  final double size;
  final Color ringColor;
  final Color backgroundColor;
  final Function()? onComplete;

  const VanishCountdownRing({
    super.key,
    required this.totalDuration,
    required this.remainingDuration,
    this.isRunning = false,
    this.size = 200,
    this.ringColor = Colors.white,
    this.backgroundColor = Colors.black,
    this.onComplete,
  });

  @override
  State<VanishCountdownRing> createState() => _VanishCountdownRingState();
}

class _VanishCountdownRingState extends State<VanishCountdownRing>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  int _lastSecond = 0;
  double _dissolveProgress = 0.0;
  final List<_DissolvingDot> _dots = [];
  final math.Random _random = math.Random();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.totalDuration,
    );
    _controller.addListener(_updateDissolve);
    _controller.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        widget.onComplete?.call();
      }
    });
    _initDots();
  }

  void _initDots() {
    _dots.clear();
    for (int i = 0; i < 60; i++) {
      _dots.add(_DissolvingDot(
        angle: (i / 60) * 2 * math.pi,
        startRadius: 0.7,
        delay: _random.nextDouble() * 0.3,
      ));
    }
  }

  void _updateDissolve() {
    final currentSecond = widget.remainingDuration.inSeconds;
    if (currentSecond != _lastSecond && widget.isRunning) {
      _lastSecond = currentSecond;
      HapticFeedback.selectionClick();
    }

    final progress = 1 - _controller.value;
    setState(() {
      _dissolveProgress = progress;
    });
  }

  @override
  void didUpdateWidget(VanishCountdownRing oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isRunning && !_controller.isAnimating) {
      _controller.forward(from: 0);
    } else if (!widget.isRunning && _controller.isAnimating) {
      _controller.stop();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  String _formatTime(Duration duration) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (duration.inHours > 0) {
      final hours = duration.inHours.toString().padLeft(2, '0');
      return '$hours:$minutes:$seconds';
    }
    return '$minutes:$seconds';
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: Size(widget.size, widget.size),
            painter: _VanishRingPainter(
              progress: widget.totalDuration.inSeconds > 0
                  ? widget.remainingDuration.inSeconds /
                      widget.totalDuration.inSeconds
                  : 0,
              ringColor: widget.ringColor,
              backgroundColor: widget.backgroundColor,
              dissolveProgress: _dissolveProgress,
              dots: _dots,
            ),
          ),
          Text(
            _formatTime(widget.remainingDuration),
            style: TextStyle(
              color: widget.ringColor,
              fontSize: widget.size * 0.18,
              fontWeight: FontWeight.w200,
              letterSpacing: 2,
            ),
          ),
        ],
      ),
    );
  }
}

class _DissolvingDot {
  final double angle;
  final double startRadius;
  final double delay;

  _DissolvingDot({
    required this.angle,
    required this.startRadius,
    required this.delay,
  });
}

class _VanishRingPainter extends CustomPainter {
  final double progress;
  final Color ringColor;
  final Color backgroundColor;
  final double dissolveProgress;
  final List<_DissolvingDot> dots;

  _VanishRingPainter({
    required this.progress,
    required this.ringColor,
    required this.backgroundColor,
    required this.dissolveProgress,
    required this.dots,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width * 0.4;
    final strokeWidth = size.width * 0.02;

    final bgPaint = Paint()
      ..color = backgroundColor.withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;
    canvas.drawCircle(center, radius, bgPaint);

    if (progress > 0) {
      final progressPaint = Paint()
        ..color = ringColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        -math.pi / 2,
        2 * math.pi * progress,
        false,
        progressPaint,
      );
    }

    final dissolveThreshold = 0.3;
    if (dissolveProgress < dissolveThreshold && dots.isNotEmpty) {
      final dotPaint = Paint()
        ..color = ringColor.withValues(alpha: 0.8)
        ..style = PaintingStyle.fill;

      for (final dot in dots) {
        final dotProgress = ((dissolveProgress / dissolveThreshold) - dot.delay)
            .clamp(0.0, 1.0);
        if (dotProgress > 0) {
          final currentRadius = radius *
              (dot.startRadius + (1 - dot.startRadius) * (1 - dotProgress));
          final x =
              center.dx + math.cos(dot.angle - math.pi / 2) * currentRadius;
          final y =
              center.dy + math.sin(dot.angle - math.pi / 2) * currentRadius;
          final dotSize = 2.0 * dotProgress;
          canvas.drawCircle(Offset(x, y), dotSize, dotPaint);
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant _VanishRingPainter oldDelegate) =>
      oldDelegate.progress != progress ||
      oldDelegate.dissolveProgress != dissolveProgress;
}

class SyllabusVelocityTracker extends StatelessWidget {
  final double pagesPerHour;
  final int completedSubchapters;
  final int totalSubchapters;
  final String subject;

  const SyllabusVelocityTracker({
    super.key,
    required this.pagesPerHour,
    required this.completedSubchapters,
    required this.totalSubchapters,
    required this.subject,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AxonColors.electricCyan.withValues(alpha: 0.3),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.speed,
            color: AxonColors.electricCyan,
            size: 16,
          ),
          const SizedBox(width: 8),
          Text(
            '${pagesPerHour.toStringAsFixed(1)} Pages/hr',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '$completedSubchapters/$totalSubchapters',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 10,
            ),
          ),
        ],
      ),
    );
  }
}

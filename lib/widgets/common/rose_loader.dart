import 'dart:math' as math;
import 'package:flutter/material.dart';

class RoseLoader extends StatefulWidget {
  final double size;
  final Color? color;

  const RoseLoader({super.key, this.size = 60, this.color});

  @override
  State<RoseLoader> createState() => _RoseLoaderState();
}

class _RoseLoaderState extends State<RoseLoader>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.size,
      height: widget.size,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return CustomPaint(
            painter: RoseLoaderPainter(
              progress: _controller.value,
              color: widget.color ?? Colors.white,
            ),
          );
        },
      ),
    );
  }
}

class RoseLoaderPainter extends CustomPainter {
  final double progress;
  final Color color;

  RoseLoaderPainter({required this.progress, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final double a = size.width / 2 * 0.7;

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    double totalTheta = math.pi * 2 * progress;

    Path path = Path();
    for (double t = 0; t <= totalTheta; t += 0.05) {
      double r = a * math.sin(3 * t);
      double x = center.dx + r * math.cos(t);
      double y = center.dy + r * math.sin(t);

      if (t == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(RoseLoaderPainter oldDelegate) =>
      oldDelegate.progress != progress || oldDelegate.color != color;
}

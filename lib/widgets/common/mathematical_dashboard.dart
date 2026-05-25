import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../theme/app_theme.dart';

class RadialFocusOrbit extends StatelessWidget {
  final double progress;
  final double secondaryProgress;
  final String? label;
  final String? subLabel;

  const RadialFocusOrbit({
    super.key,
    required this.progress,
    this.secondaryProgress = 0.0,
    this.label,
    this.subLabel,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 280,
      padding: const EdgeInsets.all(16),
      child: Stack(
        alignment: Alignment.center,
        children: [
          Container(
            width: 180,
            height: 180,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: AxonColors.accent.withValues(alpha: 0.15),
                  blurRadius: 100,
                  spreadRadius: 20,
                ),
              ],
            ),
          ),
          CustomPaint(
            size: const Size(260, 260),
            painter: OrbitPainter(
              progress: progress,
              secondaryProgress: secondaryProgress,
            ),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                label ?? "DAILY GOAL",
                style: TextStyle(
                  color: AxonColors.textSecondary,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 1.2,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                "${(progress * 100).toInt()}%",
                style: TextStyle(
                  color: AxonColors.textPrimary,
                  fontSize: 48,
                  fontWeight: FontWeight.w800,
                ),
              ),
              if (subLabel != null)
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.keyboard_arrow_up,
                      color: AxonColors.accent,
                      size: 16,
                    ),
                    Text(
                      subLabel!,
                      style: TextStyle(
                        color: AxonColors.accent,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class OrbitPainter extends CustomPainter {
  final double progress;
  final double secondaryProgress;

  OrbitPainter({
    required this.progress,
    required this.secondaryProgress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final baseRadius = size.width / 2 - 20;

    final trackPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.1)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 12
      ..strokeCap = StrokeCap.round;

    final progressPaint = Paint()
      ..color = AxonColors.accent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 12
      ..strokeCap = StrokeCap.round;

    final glowPaint = Paint()
      ..color = AxonColors.accent.withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 20
      ..strokeCap = StrokeCap.round
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8);

    final secondaryPaint = Paint()
      ..color = AxonColors.accentPurple
      ..style = PaintingStyle.stroke
      ..strokeWidth = 8
      ..strokeCap = StrokeCap.round;

    canvas.drawCircle(center, baseRadius, trackPaint);
    final secondaryRect = Rect.fromCircle(center: center, radius: baseRadius);
    canvas.drawArc(secondaryRect, -pi / 2, 2 * pi * secondaryProgress, false,
        secondaryPaint);

    canvas.drawCircle(center, baseRadius - 20, trackPaint);
    final progressRect =
        Rect.fromCircle(center: center, radius: baseRadius - 20);
    canvas.drawArc(progressRect, -pi / 2, 2 * pi * progress, false, glowPaint);
    canvas.drawArc(
        progressRect, -pi / 2, 2 * pi * progress, false, progressPaint);

    final innerPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.2)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    canvas.drawCircle(center, baseRadius - 40, innerPaint);

    final tickPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.3)
      ..strokeWidth = 2;
    for (int i = 0; i < 12; i++) {
      final angle = (i * 30) * pi / 180 - pi / 2;
      final innerPoint = Offset(center.dx + (baseRadius - 50) * cos(angle),
          center.dy + (baseRadius - 50) * sin(angle));
      final outerPoint = Offset(center.dx + (baseRadius - 45) * cos(angle),
          center.dy + (baseRadius - 45) * sin(angle));
      canvas.drawLine(innerPoint, outerPoint, tickPaint);
    }
  }

  @override
  bool shouldRepaint(covariant OrbitPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.secondaryProgress != secondaryProgress;
  }
}

class SubjectLayeredStack extends StatelessWidget {
  final List<Map<String, dynamic>> subjects;
  final Function(String)? onTap;

  const SubjectLayeredStack({super.key, required this.subjects, this.onTap});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 200,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 24),
        itemCount: subjects.length,
        itemBuilder: (context, index) {
          final subject = subjects[index];
          final progress = subject['progress'] as double? ?? 0.0;
          final code = subject['code'] as String? ?? '';
          final name = subject['name'] as String? ?? '';

          return Padding(
            padding: const EdgeInsets.only(right: 16),
            child: GestureDetector(
              onTap: () => onTap?.call(code),
              child: Container(
                width: 160,
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.05),
                  borderRadius: BorderRadius.circular(24),
                  border:
                      Border.all(color: Colors.white.withValues(alpha: 0.1)),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(24),
                  child: BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                    child: Padding(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 4),
                            decoration: BoxDecoration(
                              color: AxonColors.accent.withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(code,
                                style: TextStyle(
                                    color: AxonColors.accent,
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700)),
                          ),
                          const SizedBox(height: 8),
                          Text(name,
                              style: TextStyle(
                                  color: AxonColors.textPrimary,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                  height: 1.2),
                              maxLines: 2),
                          const SizedBox(height: 12),
                          ClipRRect(
                            borderRadius: BorderRadius.circular(2),
                            child: LinearProgressIndicator(
                              value: progress,
                              backgroundColor:
                                  Colors.white.withValues(alpha: 0.1),
                              valueColor:
                                  AlwaysStoppedAnimation(AxonColors.accent),
                              minHeight: 3,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text("${(progress * 100).toInt()}% complete",
                              style: TextStyle(
                                  color: AxonColors.textTertiary,
                                  fontSize: 10)),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class FloatingCommandBar extends StatelessWidget {
  final VoidCallback? onTap;
  final String hintText;

  const FloatingCommandBar(
      {super.key, this.onTap, this.hintText = "Ask Axon..."});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 30),
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          height: 64,
          decoration: BoxDecoration(
            color: const Color(0xFF0D1520),
            borderRadius: BorderRadius.circular(32),
            border: Border.all(color: AxonColors.accent.withValues(alpha: 0.2)),
            boxShadow: [
              BoxShadow(
                  color: Colors.black.withValues(alpha: 0.4),
                  blurRadius: 30,
                  offset: const Offset(0, 10))
            ],
          ),
          child: Row(
            children: [
              const SizedBox(width: 20),
              Icon(Icons.mic_none, color: Colors.white.withValues(alpha: 0.54)),
              const SizedBox(width: 12),
              Expanded(
                  child: Text(hintText,
                      style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.24),
                          fontSize: 14))),
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                      colors: [AxonColors.accent, AxonColors.accentPurple]),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.auto_awesome,
                    color: Colors.white, size: 20),
              ),
              const SizedBox(width: 16),
            ],
          ),
        ),
      ),
    );
  }
}

class RoseCurveBackground extends StatefulWidget {
  final Widget child;
  final double opacity;

  const RoseCurveBackground(
      {super.key, required this.child, this.opacity = 0.1});

  @override
  State<RoseCurveBackground> createState() => _RoseCurveBackgroundState();
}

class _RoseCurveBackgroundState extends State<RoseCurveBackground>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller =
        AnimationController(vsync: this, duration: const Duration(seconds: 30))
          ..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: AnimatedBuilder(
            animation: _controller,
            builder: (context, child) => CustomPaint(
                painter: RoseCurvePainter(
                    animation: _controller.value, opacity: widget.opacity)),
          ),
        ),
        widget.child,
      ],
    );
  }
}

class RoseCurvePainter extends CustomPainter {
  final double animation;
  final double opacity;

  RoseCurvePainter({required this.animation, this.opacity = 0.1});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AxonColors.accent.withValues(alpha: opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    final random = Random(42);
    for (int i = 0; i < 15; i++) {
      final centerX = size.width * (0.1 + random.nextDouble() * 0.8);
      final centerY = size.height * (0.1 + random.nextDouble() * 0.8);
      final radius = 50.0 + random.nextDouble() * 150;
      final n = 3 + random.nextInt(5);
      final d = 1 + random.nextInt(7);
      final phase = animation * 2 * pi + random.nextDouble() * pi;
      final path = Path();
      for (double angle = 0; angle <= 2 * pi; angle += 0.02) {
        final r = radius * cos(n / d * angle + phase);
        final x = centerX + r * cos(angle);
        final y = centerY + r * sin(angle);
        if (angle == 0) {
          path.moveTo(x, y);
        } else {
          path.lineTo(x, y);
        }
      }
      path.close();
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant RoseCurvePainter oldDelegate) =>
      oldDelegate.animation != animation;
}

class StudyPulseBackground extends StatefulWidget {
  final Widget child;
  final bool isActive;

  const StudyPulseBackground(
      {super.key, required this.child, this.isActive = false});

  @override
  State<StudyPulseBackground> createState() => _StudyPulseBackgroundState();
}

class _StudyPulseBackgroundState extends State<StudyPulseBackground>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _controller =
        AnimationController(vsync: this, duration: const Duration(seconds: 4));
    _pulseAnimation = Tween<double>(begin: 0.08, end: 0.15)
        .animate(CurvedAnimation(parent: _controller, curve: Curves.easeInOut));
    if (widget.isActive) {
      _controller.repeat(reverse: true);
    }
  }

  @override
  void didUpdateWidget(StudyPulseBackground oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isActive && !oldWidget.isActive) {
      _controller.repeat(reverse: true);
    } else if (!widget.isActive && oldWidget.isActive) {
      _controller.stop();
      _controller.value = 0;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isActive) return widget.child;
    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) => Container(
        decoration: BoxDecoration(
          gradient: RadialGradient(colors: [
            AxonColors.accent.withValues(alpha: _pulseAnimation.value),
            Colors.transparent
          ], radius: 1.5),
        ),
        child: widget.child,
      ),
    );
  }
}

class MinimalRosePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.02)
      ..strokeWidth = 0.5
      ..style = PaintingStyle.stroke;
    final center = Offset(size.width / 2, size.height * 0.7);
    final path = Path();
    const k = 7 / 2;
    for (double t = 0; t < 4 * pi; t += 0.01) {
      double r = 150 * cos(k * t);
      double x = center.dx + r * cos(t);
      double y = center.dy + r * sin(t);
      if (t == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class ContinueTaskCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final String timeRemaining;

  const ContinueTaskCard(
      {super.key,
      required this.title,
      required this.subtitle,
      required this.timeRemaining});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: const Color(0xFF111111),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.play_circle_outline,
                    size: 16, color: Colors.white54),
                const SizedBox(width: 8),
                Text("CONTINUE SESSION",
                    style: TextStyle(
                        color: Colors.white54,
                        fontSize: 10,
                        letterSpacing: 1.5,
                        fontWeight: FontWeight.bold)),
              ],
            ),
            const SizedBox(height: 16),
            Text(title,
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w500)),
            const SizedBox(height: 4),
            Row(
              children: [
                Expanded(
                  child: Text(subtitle,
                      style: TextStyle(color: Colors.white38, fontSize: 13),
                      overflow: TextOverflow.ellipsis),
                ),
                const SizedBox(width: 8),
                Text(timeRemaining,
                    style: TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                        fontFamily: 'Monospace')),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class StackedAnalyticsCards extends StatelessWidget {
  final String subjectName;
  final String subjectCode;
  final String estimatedGrade;
  final double coverage;

  const StackedAnalyticsCards(
      {super.key,
      required this.subjectName,
      required this.subjectCode,
      required this.estimatedGrade,
      required this.coverage});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 420,
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Stack(
        alignment: Alignment.topCenter,
        children: [
          Positioned(
              top: 48,
              child: _buildBlurredBackground(
                  scale: 0.88, opacity: 0.15, blur: 12)),
          Positioned(
              top: 24,
              child:
                  _buildBlurredBackground(scale: 0.94, opacity: 0.4, blur: 6)),
          Container(
            width: double.infinity,
            height: 320,
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: const Color(0xFF161616),
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text("SUBJECT ANALYTICS",
                        style: TextStyle(
                            color: Colors.white38,
                            fontSize: 10,
                            letterSpacing: 1.2)),
                    Text("${(coverage * 100).toInt()}% COVERED",
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.bold)),
                  ],
                ),
                const SizedBox(height: 12),
                Text("$subjectName $subjectCode",
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 20),
                Expanded(child: _GrowthLineChart()),
                const SizedBox(height: 20),
                Row(
                  children: [
                    const Icon(Icons.arrow_forward_ios,
                        size: 12, color: Colors.white24),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text("EST. $estimatedGrade",
                          style: TextStyle(color: Colors.white54, fontSize: 11),
                          overflow: TextOverflow.ellipsis),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBlurredBackground(
      {required double scale, required double opacity, required double blur}) {
    return Transform.scale(
      scale: scale,
      child: Opacity(
        opacity: opacity,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(24),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
            child: Container(
              width: 340,
              height: 320,
              decoration: BoxDecoration(
                color: const Color(0xFF161616),
                border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _GrowthLineChart extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return LineChart(
      LineChartData(
        gridData: FlGridData(show: false),
        titlesData: FlTitlesData(show: false),
        borderData: FlBorderData(show: false),
        lineBarsData: [
          LineChartBarData(
            spots: [
              FlSpot(0, 1),
              FlSpot(1, 1.5),
              FlSpot(2, 2.8),
              FlSpot(3, 3.5),
              FlSpot(4, 4.5)
            ],
            isCurved: true,
            color: Colors.white12,
            dashArray: [5, 5],
            barWidth: 2,
            dotData: FlDotData(show: false),
          ),
          LineChartBarData(
            spots: [
              FlSpot(0, 1),
              FlSpot(1, 1.2),
              FlSpot(2, 2.5),
              FlSpot(3, 3.2),
              FlSpot(4, 5.0)
            ],
            isCurved: true,
            color: Colors.white,
            barWidth: 3,
            belowBarData: BarAreaData(
                show: true, color: Colors.white.withValues(alpha: 0.03)),
            dotData: FlDotData(show: false),
          ),
        ],
      ),
    );
  }
}

// lib/widgets/native/conqueror_milestone.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// A cinematic, fully native 1-Hour Milestone overlay.
/// Trigger this by wrapping it in a generic transparent [PageRouteBuilder]
/// or placing it in an overlay stack when the timer hits 01:00:00.
class ConquerorMilestone extends StatefulWidget {
  final VoidCallback onComplete;

  const ConquerorMilestone({super.key, required this.onComplete});

  @override
  State<ConquerorMilestone> createState() => _ConquerorMilestoneState();
}

class _ConquerorMilestoneState extends State<ConquerorMilestone>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  // Timelines
  late final Animation<double> _bgBlur;
  late final Animation<double> _mountainReveal;
  late final Animation<double> _climberAscent;
  late final Animation<double> _textOpacity;
  late final Animation<double> _textTracking;
  late final Animation<double> _chromaticFringe;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration:
          const Duration(milliseconds: 6500), // 6.5 second cinematic sequence
    );

    // 1. Mist & Blur rolls in (0.0 -> 0.15)
    _bgBlur = Tween<double>(begin: 0.0, end: 15.0).animate(
      CurvedAnimation(
          parent: _controller,
          curve: const Interval(0.0, 0.15, curve: Curves.easeOut)),
    );

    // 2. Mountain Ridge slides up (0.1 -> 0.3)
    _mountainReveal = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(
          parent: _controller,
          curve: const Interval(0.1, 0.3, curve: Curves.easeOutCubic)),
    );

    // 3. Climber ascends the ridge (0.2 -> 0.5)
    _climberAscent = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
          parent: _controller,
          curve: const Interval(0.2, 0.6, curve: Curves.easeOutCubic)),
    );

    // 4. "CONQUEROR" Text burns in (0.4 -> 0.6)
    _textOpacity = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
          parent: _controller,
          curve: const Interval(0.4, 0.6, curve: Curves.easeIn)),
    );

    // 5. Text condenses from super-wide to structured (0.4 -> 0.7)
    _textTracking = Tween<double>(begin: 40.0, end: 4.0).animate(
      CurvedAnimation(
          parent: _controller,
          curve: const Interval(0.4, 0.8, curve: Curves.easeOut)),
    );

    // 6. Chromatic Aberration flares up then settles (0.45 -> 0.65)
    _chromaticFringe = TweenSequence<double>([
      TweenSequenceItem(
          tween: Tween(begin: 0.0, end: 4.0)
              .chain(CurveTween(curve: Curves.easeOut)),
          weight: 50),
      TweenSequenceItem(
          tween: Tween(begin: 4.0, end: 0.0)
              .chain(CurveTween(curve: Curves.easeIn)),
          weight: 50),
    ]).animate(
      CurvedAnimation(parent: _controller, curve: const Interval(0.45, 0.65)),
    );

    // Auto-play and dispose
    _controller.forward().then((_) {
      // Hold for a second, then reverse to fade out
      Future.delayed(const Duration(milliseconds: 800), () {
        if (mounted) {
          _controller.reverse().then((_) => widget.onComplete());
        }
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // 1. Deep Atmospheric Blur (The "Mist")
              if (_bgBlur.value > 0)
                BackdropFilter(
                  filter: ImageFilter.blur(
                      sigmaX: _bgBlur.value, sigmaY: _bgBlur.value),
                  child: Container(
                      color: const Color(0xFF080808).withValues(alpha: 0.4)),
                ),

              // 2. The Native Mountain & Climber
              Transform.translate(
                offset: Offset(0, size.height * _mountainReveal.value),
                child: CustomPaint(
                  painter: _MountainClimberPainter(
                    climberProgress: _climberAscent.value,
                    themeColor: const Color(0xFF3A86FF),
                  ),
                ),
              ),

              // 3. The Kinetic "CONQUEROR" Typography
              if (_textOpacity.value > 0)
                Center(
                  child: Transform.scale(
                    scaleY: 1.5, // The monolithic stretch
                    child: _ChromaticText(
                      text: "CONQUEROR",
                      opacity: _textOpacity.value,
                      tracking: _textTracking.value,
                      fringeOffset: _chromaticFringe.value,
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────
// NATIVE CANVAS RENDERER: The Ridge and The Silhouette
// ─────────────────────────────────────────────────────────
class _MountainClimberPainter extends CustomPainter {
  final double climberProgress;
  final Color themeColor;

  _MountainClimberPainter(
      {required this.climberProgress, required this.themeColor});

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Define the Ridge Path (Diagonal ascent)
    final double startX = 0;
    final double startY = size.height * 0.9;
    final double endX = size.width;
    final double endY = size.height * 0.3;

    final Path mountainPath = Path()
      ..moveTo(startX, startY)
      // Jagged edge logic can be expanded here. For sleekness, we use a sharp incline.
      ..lineTo(endX, endY)
      ..lineTo(size.width, size.height)
      ..lineTo(0, size.height)
      ..close();

    // 2. The Atmospheric Abyss Gradient (Inside the mountain)
    final Paint mountainPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          themeColor.withValues(alpha: 0.15),
          Colors.black,
        ],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

    canvas.drawPath(mountainPath, mountainPaint);

    // 3. The "Glass Rim" Light on the Ridge Edge
    final Path ridgeLine = Path()
      ..moveTo(startX, startY)
      ..lineTo(endX, endY);
    canvas.drawPath(
      ridgeLine,
      Paint()
        ..color = themeColor.withValues(alpha: 0.8)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..maskFilter = const MaskFilter.blur(BlurStyle.solid, 4),
    );

    // 4. The Climber Silhouette (Calculated along the vector of the slope)
    if (climberProgress > 0) {
      // Interpolate position along the line
      final double currentX =
          startX + (endX - startX) * (climberProgress * 0.8); // Stops at 80% up
      final double currentY =
          startY + (endY - startY) * (climberProgress * 0.8);

      // Draw minimal, abstract geometric climber
      _drawMinimalClimber(canvas, Offset(currentX, currentY));
    }
  }

  void _drawMinimalClimber(Canvas canvas, Offset anchor) {
    final Paint silhouettePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;

    // Angle of the mountain to tilt the climber
    canvas.save();
    canvas.translate(anchor.dx, anchor.dy);
    // Rotate slightly into the climb
    canvas.rotate(-0.3);

    // A sleek, minimal vector representation of a climbing figure
    // Head
    canvas.drawCircle(const Offset(-10, -35), 4, silhouettePaint);
    // Body Line
    canvas.drawPath(
        Path()
          ..moveTo(-10, -30)
          ..lineTo(-15, -10)
          ..lineTo(-5, 0),
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..strokeCap = StrokeCap.round);
    // Reaching Arm (The "Action" silhouette)
    canvas.drawPath(
        Path()
          ..moveTo(-10, -25)
          ..lineTo(5, -35),
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5
          ..strokeCap = StrokeCap.round);
    // Leg pushing off
    canvas.drawPath(
        Path()
          ..moveTo(-15, -10)
          ..lineTo(-25, 0),
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3
          ..strokeCap = StrokeCap.round);

    // Specular "Spark" at the hand's grip point
    canvas.drawCircle(
        const Offset(6, -36),
        2.0,
        Paint()
          ..color = Colors.white
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3));

    canvas.restore();
  }

  @override
  bool shouldRepaint(_MountainClimberPainter oldDelegate) {
    return oldDelegate.climberProgress != climberProgress;
  }
}

// ─────────────────────────────────────────────────────────
// HIGH-FIDELITY TEXT: Chromatic Aberration & Tracking
// ─────────────────────────────────────────────────────────
class _ChromaticText extends StatelessWidget {
  final String text;
  final double opacity;
  final double tracking;
  final double fringeOffset;

  const _ChromaticText({
    required this.text,
    required this.opacity,
    required this.tracking,
    required this.fringeOffset,
  });

  @override
  Widget build(BuildContext context) {
    final baseStyle = GoogleFonts.bebasNeue(
      fontSize: 80,
      color: Colors.white,
      letterSpacing: tracking,
      height: 0.9,
    );

    return Opacity(
      opacity: opacity,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Red Shift (Left)
          Transform.translate(
            offset: Offset(-fringeOffset, 0),
            child: Text(
              text,
              style: baseStyle.copyWith(
                color: const Color(0xFFFF0055).withValues(alpha: 0.6),
              ),
            ),
          ),
          // Blue Shift (Right)
          Transform.translate(
            offset: Offset(fringeOffset, 0),
            child: Text(
              text,
              style: baseStyle.copyWith(
                color: const Color(0xFF00E5FF).withValues(alpha: 0.6),
              ),
            ),
          ),
          // Core White Text
          Text(
            text,
            style: baseStyle.copyWith(
              // Adds an internal shadow to make it feel dense
              shadows: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.5),
                  blurRadius: 20,
                  offset: const Offset(0, 10),
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}

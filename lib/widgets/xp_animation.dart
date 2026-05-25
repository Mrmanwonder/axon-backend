import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

class XpAnimation extends StatefulWidget {
  final int xpAmount;
  final VoidCallback? onComplete;
  final Offset? startPosition;

  const XpAnimation({
    super.key,
    required this.xpAmount,
    this.onComplete,
    this.startPosition,
  });

  @override
  State<XpAnimation> createState() => _XpAnimationState();
}

class _XpAnimationState extends State<XpAnimation>
    with SingleTickerProviderStateMixin {
  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: _FloatingXp(
        xp: widget.xpAmount,
        onComplete: widget.onComplete,
      ),
    );
  }
}

class _FloatingXp extends StatelessWidget {
  final int xp;
  final VoidCallback? onComplete;

  const _FloatingXp({
    required this.xp,
    this.onComplete,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        _XpOrb(xp: xp),
        const SizedBox(height: 8),
        Text(
          '+$xp XP',
          style: GoogleFonts.orbitron(
            fontSize: 28,
            fontWeight: FontWeight.w800,
            color: Colors.amber,
            shadows: [
              Shadow(
                color: Colors.amber.withValues(alpha: 0.8),
                blurRadius: 20,
              ),
              Shadow(
                color: Colors.orange.withValues(alpha: 0.6),
                blurRadius: 40,
              ),
            ],
          ),
        )
            .animate()
            .fadeIn(duration: 200.ms)
            .slideY(
                begin: 0.5, end: 0, duration: 400.ms, curve: Curves.easeOutBack)
            .shimmer(
                duration: 1200.ms,
                delay: 400.ms,
                color: Colors.white.withValues(alpha: 0.5))
            .then()
            .fadeOut(delay: 800.ms, duration: 300.ms),
      ],
    )
        .animate(onComplete: (controller) => onComplete?.call())
        .scale(
          begin: const Offset(0.3, 0.3),
          end: const Offset(1, 1),
          duration: 500.ms,
          curve: Curves.easeOutBack,
        )
        .slideY(
            begin: 0,
            end: -0.8,
            delay: 600.ms,
            duration: 1500.ms,
            curve: Curves.easeInCubic);
  }
}

class _XpOrb extends StatelessWidget {
  final int xp;

  const _XpOrb({required this.xp});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 80,
      height: 80,
      child: Stack(
        alignment: Alignment.center,
        children: [
          _GlowingRing(),
          _PulsingCore(),
          _Sparkles(),
          Icon(
            Icons.bolt,
            size: 36,
            color: Colors.amber.shade300,
          )
              .animate(onPlay: (c) => c.repeat(reverse: true))
              .scale(
                  begin: const Offset(0.9, 0.9),
                  end: const Offset(1.1, 1.1),
                  duration: 600.ms)
              .rotate(begin: -0.05, end: 0.05, duration: 800.ms),
        ],
      ),
    );
  }
}

class _GlowingRing extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 80,
      height: 80,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [
            Colors.amber.withValues(alpha: 0.3),
            Colors.orange.withValues(alpha: 0.1),
            Colors.transparent,
          ],
        ),
      ),
    )
        .animate(onPlay: (c) => c.repeat())
        .scale(
          begin: const Offset(0.8, 0.8),
          end: const Offset(1.2, 1.2),
          duration: 1000.ms,
        )
        .fadeOut(begin: 0.6, duration: 1000.ms)
        .then()
        .fadeIn(duration: 1000.ms);
  }
}

class _PulsingCore extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 50,
      height: 50,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [
            Colors.white,
            Colors.amber.shade300,
            Colors.orange,
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.amber.withValues(alpha: 0.8),
            blurRadius: 20,
            spreadRadius: 2,
          ),
          BoxShadow(
            color: Colors.orange.withValues(alpha: 0.5),
            blurRadius: 40,
            spreadRadius: 5,
          ),
        ],
      ),
    )
        .animate(onPlay: (c) => c.repeat(reverse: true))
        .scale(
          begin: const Offset(0.9, 0.9),
          end: const Offset(1.1, 1.1),
          duration: 600.ms,
        )
        .then()
        .scale(
          begin: const Offset(1.1, 1.1),
          end: const Offset(0.9, 0.9),
          duration: 600.ms,
        );
  }
}

class _Sparkles extends StatelessWidget {
  final _random = math.Random();

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: List.generate(8, (index) {
        final angle = (index * 45) * (math.pi / 180);
        final distance = 35 + _random.nextDouble() * 10;
        final x = math.cos(angle) * distance;
        final y = math.sin(angle) * distance;
        final delay = _random.nextInt(300);

        return Transform.translate(
          offset: Offset(x, y),
          child: Container(
            width: 4 + _random.nextDouble() * 4,
            height: 4 + _random.nextDouble() * 4,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.amber.shade100,
              boxShadow: [
                BoxShadow(
                  color: Colors.amber.withValues(alpha: 0.8),
                  blurRadius: 8,
                ),
              ],
            ),
          )
              .animate(delay: Duration(milliseconds: delay))
              .fadeIn(duration: 150.ms)
              .scale(
                begin: const Offset(0, 0),
                end: const Offset(1, 1),
                duration: 200.ms,
                curve: Curves.easeOutBack,
              )
              .then()
              .fadeOut(duration: 300.ms),
        );
      }),
    );
  }
}

class XpNotification extends StatefulWidget {
  final int xp;
  final String message;
  final VoidCallback? onComplete;

  const XpNotification({
    super.key,
    required this.xp,
    required this.message,
    this.onComplete,
  });

  @override
  State<XpNotification> createState() => _XpNotificationState();
}

class _XpNotificationState extends State<XpNotification> {
  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Colors.black.withValues(alpha: 0.9),
              Colors.grey.shade900,
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: Colors.amber.withValues(alpha: 0.5),
            width: 2,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.amber.withValues(alpha: 0.3),
              blurRadius: 20,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const _XpOrbSmall(),
            const SizedBox(width: 16),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  '+${widget.xp} XP',
                  style: GoogleFonts.orbitron(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: Colors.amber,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  widget.message,
                  style: GoogleFonts.googleSans(
                    fontSize: 12,
                    color: Colors.white70,
                  ),
                ),
              ],
            ),
          ],
        ),
      )
          .animate(onComplete: (_) => widget.onComplete?.call())
          .fadeIn(duration: 300.ms)
          .slideY(
              begin: -0.5, end: 0, duration: 400.ms, curve: Curves.easeOutBack)
          .then(delay: 2000.ms)
          .fadeOut(duration: 300.ms)
          .slideY(begin: 0, end: -0.3, duration: 300.ms),
    );
  }
}

class _XpOrbSmall extends StatelessWidget {
  const _XpOrbSmall();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 44,
      height: 44,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [
            Colors.white,
            Colors.amber.shade300,
            Colors.orange,
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.amber.withValues(alpha: 0.6),
            blurRadius: 12,
          ),
        ],
      ),
      child: const Icon(
        Icons.bolt,
        color: Colors.white,
        size: 24,
      ),
    ).animate(onPlay: (c) => c.repeat(reverse: true)).scale(
          begin: const Offset(0.95, 0.95),
          end: const Offset(1.05, 1.05),
          duration: 800.ms,
        );
  }
}

class LevelUpAnimation extends StatefulWidget {
  final int newLevel;
  final String title;
  final VoidCallback? onComplete;

  const LevelUpAnimation({
    super.key,
    required this.newLevel,
    required this.title,
    this.onComplete,
  });

  @override
  State<LevelUpAnimation> createState() => _LevelUpAnimationState();
}

class _LevelUpAnimationState extends State<LevelUpAnimation> {
  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: Container(
        padding: const EdgeInsets.all(32),
        decoration: BoxDecoration(
          gradient: RadialGradient(
            colors: [
              Colors.purple.withValues(alpha: 0.3),
              Colors.black.withValues(alpha: 0.95),
            ],
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'LEVEL UP!',
              style: GoogleFonts.orbitron(
                fontSize: 32,
                fontWeight: FontWeight.w900,
                color: Colors.purple.shade300,
                letterSpacing: 4,
                shadows: [
                  Shadow(
                    color: Colors.purple.withValues(alpha: 0.8),
                    blurRadius: 30,
                  ),
                ],
              ),
            )
                .animate()
                .fadeIn(duration: 300.ms)
                .scale(
                    begin: const Offset(0.5, 0.5),
                    end: const Offset(1, 1),
                    duration: 500.ms,
                    curve: Curves.easeOutBack)
                .shimmer(
                    duration: 1500.ms,
                    color: Colors.white.withValues(alpha: 0.5)),
            const SizedBox(height: 24),
            _LevelCircle(level: widget.newLevel),
            const SizedBox(height: 16),
            Text(
              widget.title,
              style: GoogleFonts.googleSans(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Level ${widget.newLevel}',
              style: GoogleFonts.orbitron(
                fontSize: 28,
                fontWeight: FontWeight.w800,
                color: Colors.purple.shade300,
              ),
            ),
          ],
        ),
      )
          .animate(onComplete: (_) => widget.onComplete?.call())
          .fadeIn(duration: 400.ms)
          .then(delay: 2500.ms)
          .fadeOut(duration: 400.ms),
    );
  }
}

class _LevelCircle extends StatelessWidget {
  final int level;

  const _LevelCircle({required this.level});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 120,
      height: 120,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: RadialGradient(
          colors: [
            Colors.purple.shade300,
            Colors.purple.shade800,
            Colors.purple.shade900,
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.purple.withValues(alpha: 0.8),
            blurRadius: 40,
            spreadRadius: 10,
          ),
        ],
      ),
      child: Stack(
        alignment: Alignment.center,
        children: [
          _OrbitingParticles(),
          Text(
            '$level',
            style: GoogleFonts.orbitron(
              fontSize: 48,
              fontWeight: FontWeight.w900,
              color: Colors.white,
              shadows: [
                Shadow(
                  color: Colors.purple.withValues(alpha: 0.8),
                  blurRadius: 20,
                ),
              ],
            ),
          ),
        ],
      ),
    )
        .animate()
        .scale(
          begin: const Offset(0, 0),
          end: const Offset(1, 1),
          duration: 600.ms,
          curve: Curves.easeOutBack,
        )
        .then()
        .shimmer(duration: 2000.ms, color: Colors.white.withValues(alpha: 0.3));
  }
}

class _OrbitingParticles extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 140,
      height: 140,
      child: Stack(
        alignment: Alignment.center,
        children: List.generate(6, (index) {
          final angle = (index * 60) * (math.pi / 180);
          return TweenAnimationBuilder<double>(
            tween: Tween(begin: 0, end: 2 * math.pi),
            duration: const Duration(seconds: 3),
            builder: (context, value, child) {
              final x = math.cos(angle + value) * 55;
              final y = math.sin(angle + value) * 55;
              return Transform.translate(
                offset: Offset(x, y),
                child: Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.purple.shade200,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.purple.withValues(alpha: 0.8),
                        blurRadius: 8,
                      ),
                    ],
                  ),
                ),
              );
            },
          );
        }),
      ),
    );
  }
}

void showXpGain(BuildContext context, int xp,
    {String? message, Offset? position}) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;

  entry = OverlayEntry(
    builder: (context) => Positioned(
      top: position?.dy ?? 100,
      left: position?.dx,
      right: position == null ? 20 : null,
      child: Center(
        child: XpAnimation(
          xpAmount: xp,
          onComplete: () {
            entry.remove();
          },
        ),
      ),
    ),
  );

  overlay.insert(entry);
}

void showLevelUp(BuildContext context, int newLevel, String title) {
  final overlay = Overlay.of(context);
  late OverlayEntry entry;

  entry = OverlayEntry(
    builder: (context) => Positioned.fill(
      child: Container(
        color: Colors.black.withValues(alpha: 0.7),
        child: Center(
          child: LevelUpAnimation(
            newLevel: newLevel,
            title: title,
            onComplete: () {
              entry.remove();
            },
          ),
        ),
      ),
    ),
  );

  overlay.insert(entry);
}

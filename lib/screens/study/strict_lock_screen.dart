// lib/screens/study/strict_lock_screen.dart
import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/study_lock_service.dart';
import '../../services/overlay_permission_service.dart';
import '../../theme/app_theme.dart';

final selectedAppsProvider = StateProvider<Set<String>>((ref) => {});

class StrictLockScreen extends ConsumerStatefulWidget {
  final String? blockedPackage;

  const StrictLockScreen({super.key, this.blockedPackage});

  @override
  ConsumerState<StrictLockScreen> createState() => _StrictLockScreenState();
}

class _StrictLockScreenState extends ConsumerState<StrictLockScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _glowController;
  late AnimationController _shimmerController;
  late AnimationController _particleController;
  late AnimationController _breathController;

  Timer? _progressTimer;
  int _loggedMinutes = 0;
  int _requiredMinutes = 60;
  bool _isActive = false;

  @override
  void initState() {
    super.initState();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat(reverse: true);

    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2500),
    )..repeat(reverse: true);

    _shimmerController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 3000),
    )..repeat();

    _particleController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 5000),
    )..repeat();

    _breathController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 4000),
    )..repeat(reverse: true);

    _loadStatus();
    _startProgressTimer();
  }

  Future<void> _loadStatus() async {
    final status = await StudyLockService.instance.getStatus();
    if (mounted) {
      setState(() {
        _loggedMinutes = status['loggedMinutes'] ?? 0;
        _requiredMinutes = status['requiredMinutes'] ?? 60;
        _isActive = status['isActive'] ?? false;
      });
    }
  }

  void _startProgressTimer() {
    _progressTimer?.cancel();
    // Efficient battery-conscious polling: only update every 30s when app is visible
    // The actual lock continues via native service - this is just for UI updates
    _progressTimer = Timer.periodic(const Duration(seconds: 30), (_) {
      if (mounted) {
        _loadStatus();
      }
    });
  }

  @override
  void dispose() {
    _progressTimer?.cancel();
    _pulseController.dispose();
    _glowController.dispose();
    _shimmerController.dispose();
    _particleController.dispose();
    _breathController.dispose();
    super.dispose();
  }

  int get _remaining => (_requiredMinutes - _loggedMinutes).clamp(0, _requiredMinutes);
  double get _progress => _requiredMinutes > 0 ? (_loggedMinutes / _requiredMinutes).clamp(0.0, 1.0) : 0.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              const Color(0xFF050608),
              const Color(0xFF090B10),
              const Color(0xFF020304),
            ],
          ),
        ),
        child: Stack(
          children: [
            // Animated particle background
            AnimatedBuilder(
              animation: _particleController,
              builder: (context, child) {
                return CustomPaint(
                  painter: _LockParticlePainter(
                    particleValue: _particleController.value,
                    breathValue: _breathController.value,
                  ),
                  size: Size.infinite,
                );
              },
            ),

            SafeArea(
              child: !_isActive || _remaining <= 0
                  ? _buildUnlockedContent()
                  : _buildLockedContent(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLockedContent() {
    return Column(
      children: [
        const Spacer(flex: 1),

        // Animated lock icon with glow rings
        _AnimatedLockWidget(
          pulseController: _pulseController,
          glowController: _glowController,
          shimmerController: _shimmerController,
          breathController: _breathController,
        ),

        const SizedBox(height: 32),

        Text(
          'Focus Lock',
          style: GoogleFonts.googleSans(
            fontSize: 34,
            fontWeight: FontWeight.w700,
            color: Colors.white,
            letterSpacing: 0,
          ),
        )
            .animate()
            .fadeIn(duration: 600.ms)
            .slideY(begin: 0.08, end: 0, duration: 600.ms),

        const SizedBox(height: 12),

        Text(
          'Blocked apps reopen when this session is complete.',
          style: GoogleFonts.googleSans(
            fontSize: 15,
            color: Colors.white.withValues(alpha: 0.62),
            height: 1.4,
          ),
          textAlign: TextAlign.center,
        )
            .animate()
            .fadeIn(delay: 200.ms, duration: 500.ms),

        const SizedBox(height: 44),

        // Progress ring with orbit animation
        _ProgressRingWidget(
          progress: _progress,
          logged: _loggedMinutes,
          required: _requiredMinutes,
          pulseController: _pulseController,
          glowController: _glowController,
          particleController: _particleController,
        ),

        const SizedBox(height: 32),

        // Remaining time
        Text(
          '$_remaining minutes remaining',
          style: GoogleFonts.googleSans(
            fontSize: 17,
            fontWeight: FontWeight.w600,
            color: Colors.white.withValues(alpha: 0.92),
          ),
        )
            .animate()
            .fadeIn(delay: 400.ms, duration: 500.ms)
            .slideY(begin: 0.2, end: 0, delay: 400.ms, duration: 500.ms),

        const Spacer(flex: 2),

        // Motivational text with typing effect
        _MotivationalWidget(),

        const SizedBox(height: 48),
      ],
    );
  }

  Widget _buildUnlockedContent() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Spacer(flex: 1),

        // Unlocked icon with celebration animation
        _UnlockedIcon(breathController: _breathController),

        const SizedBox(height: 36),

        Text(
          'Session Complete',
          style: GoogleFonts.googleSans(
            fontSize: 30,
            fontWeight: FontWeight.w700,
            color: Colors.white,
            letterSpacing: 0,
          ),
        )
            .animate()
            .fadeIn(duration: 600.ms)
            .scale(
              begin: const Offset(0.8, 0.8),
              end: const Offset(1, 1),
              duration: 600.ms,
              curve: Curves.easeOutBack,
            ),

        const SizedBox(height: 16),

        Text(
          'Great job staying focused!',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            color: Colors.grey[400],
          ),
        )
            .animate()
            .fadeIn(delay: 300.ms, duration: 500.ms),

        const Spacer(flex: 2),

        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: ElevatedButton(
            onPressed: () => Navigator.of(context).pop(),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF3A86FF),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 48, vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
            child: Text(
              'Continue Studying',
              style: GoogleFonts.googleSans(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        )
            .animate()
            .fadeIn(delay: 600.ms, duration: 500.ms)
            .slideY(begin: 0.3, end: 0, delay: 600.ms, duration: 500.ms),

        const SizedBox(height: 48),
      ],
    );
  }
}

class _AnimatedLockWidget extends StatelessWidget {
  final AnimationController pulseController;
  final AnimationController glowController;
  final AnimationController shimmerController;
  final AnimationController breathController;

  const _AnimatedLockWidget({
    required this.pulseController,
    required this.glowController,
    required this.shimmerController,
    required this.breathController,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        ...List.generate(3, (index) {
          return AnimatedBuilder(
            animation: glowController,
            builder: (context, child) {
              return Container(
                width: 140 + (index * 30),
                height: 140 + (index * 30),
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: const Color(0xFF3A86FF).withValues(
                      alpha: 0.15 - (index * 0.04),
                    ),
                    width: 1.5,
                  ),
                ),
              )
                  .animate(onPlay: (c) => c.repeat(reverse: true))
                  .scale(
                    begin: const Offset(0.97, 0.97),
                    end: const Offset(1.03, 1.03),
                    duration: Duration(milliseconds: 2000 + (index * 300)),
                  );
            },
          );
        }),
        AnimatedBuilder(
          animation: pulseController,
          builder: (context, child) {
            return Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    const Color(0xFF3A86FF).withValues(alpha: 0.4 * pulseController.value),
                    const Color(0xFF3A86FF).withValues(alpha: 0.1),
                  ],
                ),
              ),
            );
          },
        ),
        AnimatedBuilder(
          animation: breathController,
          builder: (context, child) {
            final scale = 0.95 + (breathController.value * 0.05);
            return Transform.scale(
              scale: scale,
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(32),
                  gradient: const LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Color(0x663A86FF),
                      Color(0x220B0D0E),
                    ],
                  ),
                  border: Border.all(
                    color: const Color(0xFF3A86FF).withValues(alpha: 0.6),
                    width: 1,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: const Color(0xFF3A86FF).withValues(alpha: 0.28),
                      blurRadius: 36,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: Icon(
                  Icons.lock_rounded,
                  size: 48,
                  color: Colors.white,
                ),
              ),
            );
          },
        )
            .animate()
            .fadeIn(duration: 800.ms)
            .scale(
              begin: const Offset(0.5, 0.5),
              end: const Offset(1, 1),
              duration: 800.ms,
              curve: Curves.easeOutBack,
            ),
        AnimatedBuilder(
          animation: shimmerController,
          builder: (context, child) {
            return Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(32),
                gradient: LinearGradient(
                  begin: Alignment(-1 + shimmerController.value * 2, -1),
                  end: Alignment(1 - shimmerController.value * 2, 1),
                  colors: [
                    Colors.transparent,
                    Colors.white.withValues(alpha: 0.1),
                    Colors.transparent,
                  ],
                ),
              ),
            );
          },
        ),
      ],
    )
        .animate()
        .fadeIn(duration: 1000.ms)
        .scale(
          begin: const Offset(0.7, 0.7),
          end: const Offset(1, 1),
          duration: 1000.ms,
          curve: Curves.easeOutBack,
        );
  }
}

class _ProgressRingWidget extends StatelessWidget {
  final double progress;
  final int logged;
  final int required;
  final AnimationController pulseController;
  final AnimationController glowController;
  final AnimationController particleController;

  const _ProgressRingWidget({
    required this.progress,
    required this.logged,
    required this.required,
    required this.pulseController,
    required this.glowController,
    required this.particleController,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 180,
      height: 180,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: const Size(180, 180),
            painter: _ProgressRingPainter(
              progress: 1.0,
              color: Colors.grey[850]!,
              strokeWidth: 10,
            ),
          ),
          AnimatedBuilder(
            animation: Listenable.merge([pulseController, glowController]),
            builder: (context, child) {
              return CustomPaint(
                size: const Size(180, 180),
                painter: _ProgressRingPainter(
                  progress: progress,
                  color: Color.lerp(
                    const Color(0xFFFF4444),
                    const Color(0xFF3A86FF),
                    progress,
                  ) ?? const Color(0xFF3A86FF),
                  strokeWidth: 10,
                  glowRadius: pulseController.value * 3,
                ),
              );
            },
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '$logged',
                style: GoogleFonts.orbitron(
                  fontSize: 38,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),
              Text(
                '/ $required min',
                style: GoogleFonts.googleSans(
                  fontSize: 14,
                  color: Colors.grey[500],
                ),
              ),
            ],
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: 300.ms, duration: 600.ms)
        .scale(
          begin: const Offset(0.8, 0.8),
          end: const Offset(1, 1),
          delay: 300.ms,
          duration: 600.ms,
          curve: Curves.easeOutBack,
        );
  }
}

class _ProgressRingPainter extends CustomPainter {
  final double progress;
  final Color color;
  final double strokeWidth;
  final double glowRadius;

  _ProgressRingPainter({
    required this.progress,
    required this.color,
    required this.strokeWidth,
    this.glowRadius = 0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;

    if (glowRadius > 0) {
      final glowPaint = Paint()
        ..color = color.withValues(alpha: 0.3)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth + glowRadius
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, glowRadius);
      canvas.drawCircle(center, radius, glowPaint);
    }

    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    final startAngle = -math.pi / 2;
    final sweepAngle = 2 * math.pi * progress;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _ProgressRingPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.color != color ||
        oldDelegate.glowRadius != glowRadius;
  }
}

class _LockParticlePainter extends CustomPainter {
  final double particleValue;
  final double breathValue;

  _LockParticlePainter({
    required this.particleValue,
    required this.breathValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.35);

    for (int i = 0; i < 15; i++) {
      final baseAngle = (i / 15) * 2 * math.pi;
      final drift = math.sin(particleValue * 2 * math.pi + i) * 15;
      final radius = size.width * (0.35 + (i % 4) * 0.08) + drift;
      final angle = baseAngle + particleValue * 0.3;

      final x = center.dx + radius * math.cos(angle);
      final y = center.dy + radius * math.sin(angle) * breathValue;

      final particlePaint = Paint()
        ..color = const Color(0xFF3A86FF).withValues(alpha: 0.15 + (i % 5) * 0.05)
        ..style = PaintingStyle.fill;

      canvas.drawCircle(Offset(x, y), 2 + (i % 3), particlePaint);
    }
  }

  @override
  bool shouldRepaint(covariant _LockParticlePainter oldDelegate) {
    return oldDelegate.particleValue != particleValue ||
        oldDelegate.breathValue != breathValue;
  }
}

class _UnlockedIcon extends StatelessWidget {
  final AnimationController breathController;

  const _UnlockedIcon({required this.breathController});

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: breathController,
      builder: (context, child) {
        return Transform.scale(
          scale: 0.9 + (breathController.value * 0.1),
          child: Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: RadialGradient(
                colors: [
                  Colors.green.withValues(alpha: 0.4),
                  Colors.green.withValues(alpha: 0.15),
                ],
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.green.withValues(alpha: 0.3),
                  blurRadius: 40,
                  spreadRadius: 5,
                ),
              ],
            ),
            child: const Icon(
              Icons.lock_open_rounded,
              size: 60,
              color: Colors.green,
            ),
          ),
        );
      },
    )
        .animate()
        .scale(
          begin: const Offset(0.5, 0.5),
          end: const Offset(1, 1),
          duration: 600.ms,
          curve: Curves.easeOutBack,
        )
        .then()
        .shimmer(duration: 1500.ms, color: Colors.green.withValues(alpha: 0.2));
  }
}

class _MotivationalWidget extends StatelessWidget {
  static const _quotes = [
    'Focus is the key to success.',
    'Every minute counts.',
    "You're building your future.",
    'Discipline creates champions.',
    'Stay committed to your goals.',
    'The harder you work, the luckier you get.',
    'Small progress is still progress.',
  ];

  @override
  Widget build(BuildContext context) {
    final quote = _quotes[DateTime.now().minute % _quotes.length];

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 40),
      child: Text(
        quote,
        style: GoogleFonts.googleSans(
          fontSize: 14,
          color: Colors.grey[500],
          fontStyle: FontStyle.italic,
        ),
        textAlign: TextAlign.center,
      ),
    )
        .animate()
        .fadeIn(delay: 600.ms, duration: 800.ms)
        .then()
        .shimmer(
          delay: 2500.ms,
          duration: 1500.ms,
          color: const Color(0xFF3A86FF).withValues(alpha: 0.1),
        );
  }
}

class _AppCategoryWithProvider extends ConsumerWidget {
  const _AppCategoryWithProvider();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final socialApps = CommonBlockedApps.socialMedia;
    final videoApps = CommonBlockedApps.video;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Block Distractions',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          'Select apps to block during study sessions',
          style: GoogleFonts.googleSans(
            fontSize: 12,
            color: Colors.grey[500],
          ),
        ),
        const SizedBox(height: 16),
        _AppCategory('Social Media', socialApps),
        const SizedBox(height: 16),
        _AppCategory('Video & Streaming', videoApps),
      ],
    );
  }
}

class StudyLockSetupScreen extends ConsumerStatefulWidget {
  const StudyLockSetupScreen({super.key});

  @override
  ConsumerState<StudyLockSetupScreen> createState() =>
      _StudyLockSetupScreenState();
}

class _StudyLockSetupScreenState extends ConsumerState<StudyLockSetupScreen> {
  int _selectedMinutes = 60;

  @override
  Widget build(BuildContext context) {
    final selectedApps = ref.watch(selectedAppsProvider);

    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: AxonGradients.backgroundGradient,
        ),
        child: SafeArea(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.all(20),
                  children: [
                    _buildTimeSelector(),
                    const SizedBox(height: 24),
                    const _AppCategoryWithProvider(),
                    const SizedBox(height: 32),
                    _buildStartButton(selectedApps),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_rounded, color: Colors.white),
            onPressed: () => Navigator.pop(context),
          ),
          Expanded(
            child: Text(
              'Study Lock',
              style: GoogleFonts.googleSans(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(width: 48),
        ],
      ),
    );
  }

  Widget _buildTimeSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Study Duration',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [15, 30, 45, 60, 90, 120].map((mins) {
            final isSelected = _selectedMinutes == mins;
            return GestureDetector(
              onTap: () => setState(() => _selectedMinutes = mins),
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                decoration: BoxDecoration(
                  color: isSelected
                      ? const Color(0xFF3A86FF)
                      : const Color(0xFF1A1A1A),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: isSelected
                        ? const Color(0xFF3A86FF)
                        : Colors.grey[800]!,
                  ),
                ),
                child: Text(
                  '$mins min',
                  style: GoogleFonts.googleSans(
                    fontWeight: FontWeight.w600,
                    color: isSelected ? Colors.white : Colors.grey[400],
                  ),
                ),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _buildStartButton(Set<String> selectedApps) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: selectedApps.isEmpty ? null : _startStudyLock,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF3A86FF),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
          ),
          disabledBackgroundColor: Colors.grey[800],
        ),
        child: Text(
          selectedApps.isEmpty
              ? 'Select apps to block'
              : 'Start $_selectedMinutes min Session',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }

  void _startStudyLock() async {
    final selectedApps = ref.read(selectedAppsProvider);
    final apps = selectedApps.toList();
    if (apps.isEmpty) return;

    final hasOverlay = await overlayPermissionService.checkAndRequestOverlayPermission();
    final hasAccessibility = await overlayPermissionService.isAccessibilityServiceEnabled();

    if (!hasOverlay || !hasAccessibility) {
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          backgroundColor: AxonColors.surfaceElevated,
          title: const Text(
            'Permissions Required',
            style: TextStyle(color: Colors.white),
          ),
          content: Text(
            hasOverlay
                ? 'Enable the AXON accessibility service so Study Lock can detect blocked apps.'
                : 'Allow AXON to display over other apps so Study Lock can show the lock screen.',
            style: const TextStyle(color: Colors.white70),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                if (hasOverlay) {
                  overlayPermissionService.openAccessibilitySettings();
                } else {
                  overlayPermissionService.openOverlaySettings();
                }
              },
              child: const Text('Open Settings'),
            ),
          ],
        ),
      );
      return;
    }

    await StudyLockService.instance.setDistractionApps(apps, _selectedMinutes);
    if (mounted) {
      Navigator.pop(context);
    }
  }
}

class _AppCategory extends ConsumerWidget {
  final String title;
  final List<AppBlockerService> apps;

  const _AppCategory(this.title, this.apps);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final selectedApps = ref.watch(selectedAppsProvider);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.googleSans(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.grey[400],
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: apps.map((app) {
              final isSelected = selectedApps.contains(app.packageName);
              return FilterChip(
                label: Text(app.appName),
                selected: isSelected,
                onSelected: (selected) {
                  final current = Set<String>.from(selectedApps);
                  if (selected) {
                    current.add(app.packageName);
                  } else {
                    current.remove(app.packageName);
                  }
                  ref.read(selectedAppsProvider.notifier).state = current;
                },
                selectedColor: const Color(0xFF3A86FF).withValues(alpha: 0.3),
                checkmarkColor: const Color(0xFF3A86FF),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

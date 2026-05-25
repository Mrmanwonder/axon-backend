// lib/widgets/ask_axon_logo.dart

import 'dart:async';

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme/app_theme.dart';

enum MorphState { sphere, morphing, icon }

class AskAxonLogo extends StatefulWidget {
  final double size;
  final bool isWorking;
  final double morphProgress;
  final bool playBuildUp;
  final int replaySeed;
  final bool hapticEnabled;
  final bool isSending;
  final Color plasmaColor;
  final Color arrowColor;
  final MorphState morphState;
  final bool showFullRoseCurve;
  final bool longPressMode;

  const AskAxonLogo({
    super.key,
    required this.size,
    this.isWorking = false,
    this.morphProgress = 0.0,
    this.playBuildUp = false,
    this.replaySeed = 0,
    this.hapticEnabled = false,
    this.isSending = false,
    this.plasmaColor = const Color(0xFF3A86FF),
    this.arrowColor = Colors.white,
    this.morphState = MorphState.sphere,
    this.showFullRoseCurve = false,
    this.longPressMode = false,
  });

  @override
  State<AskAxonLogo> createState() => _AskAxonLogoState();
}

class _AskAxonLogoState extends State<AskAxonLogo>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    );
    if (widget.playBuildUp) {
      _controller.forward();
    } else if (widget.isWorking || widget.isSending) {
      _controller.repeat();
    }
  }

  @override
  void didUpdateWidget(covariant AskAxonLogo oldWidget) {
    super.didUpdateWidget(oldWidget);
    final shouldAnimate =
        widget.isWorking || widget.playBuildUp || widget.isSending;
    final wasAnimating =
        oldWidget.isWorking || oldWidget.playBuildUp || oldWidget.isSending;

    if (widget.playBuildUp && !oldWidget.playBuildUp) {
      _controller.forward(from: 0.0);
    } else if (shouldAnimate && !wasAnimating) {
      _controller.repeat();
    } else if (!shouldAnimate && wasAnimating) {
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
    final shouldAnimate =
        widget.isWorking || widget.playBuildUp || widget.isSending;
    return SizedBox.square(
      dimension: widget.size,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, _) {
          final rotation =
              shouldAnimate ? _controller.value * math.pi * 2 : 0.0;
          final glow = shouldAnimate
              ? 0.5 + (math.sin(_controller.value * math.pi * 2) + 1) * 0.18
              : 0.24;
          final roseProgress = widget.playBuildUp ? _controller.value : 1.0;
          // For longPressMode: don't rotate the rose, only rotate gradient inside
          final Widget paintedRose = CustomPaint(
            painter: _RoseCurvePainter(
              color: widget.plasmaColor,
              secondaryColor: widget.arrowColor,
              phase: _controller.value,
              glowStrength: glow,
              showLoaderArc: widget.isWorking,
              roseProgress: roseProgress,
              rotateGradientOnly: widget.longPressMode,
            ),
          );
          // Don't rotate the rose path itself, only gradient rotates
          return widget.longPressMode
              ? paintedRose
              : Transform.rotate(
                  angle: rotation * 0.08,
                  child: paintedRose,
                );
        },
      ),
    );
  }
}

class _RoseCurvePainter extends CustomPainter {
  final Color color;
  final Color secondaryColor;
  final double phase;
  final double glowStrength;
  final bool showLoaderArc;
  final double roseProgress;
  final bool rotateGradientOnly;

  const _RoseCurvePainter({
    required this.color,
    required this.secondaryColor,
    required this.phase,
    required this.glowStrength,
    required this.showLoaderArc,
    this.roseProgress = 1.0,
    this.rotateGradientOnly = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.shortestSide * 0.34;
    final rosePath = Path();

    for (var i = 0; i <= (360 * roseProgress).round(); i++) {
      final t = math.pi * i / 180;
      final r = radius * math.sin(3 * t);
      final point = Offset(
        center.dx + r * math.cos(t),
        center.dy + r * math.sin(t),
      );
      if (i == 0) {
        rosePath.moveTo(point.dx, point.dy);
      } else {
        rosePath.lineTo(point.dx, point.dy);
      }
    }

    // Outer ring - removed for long press

    final glowPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = size.shortestSide * 0.09
      ..color = Colors.transparent
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
    canvas.drawPath(rosePath, glowPaint);

    // Rose curve with gradient - rotate gradient only, not the path itself
    final rosePaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = size.shortestSide * 0.055
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..shader = LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [
          secondaryColor.withValues(alpha: 0.92),
          color,
          color.withValues(alpha: 0.75),
        ],
        // Rotate gradient but not the rose path
        transform: GradientRotation(phase * math.pi * 2),
      ).createShader(Offset.zero & size);
    canvas.drawPath(rosePath, rosePaint);

    // Center circle - removed for long press

    if (showLoaderArc) {
      final loaderRect =
          Rect.fromCircle(center: center, radius: size.shortestSide * 0.45);
      final loaderPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round
        ..strokeWidth = size.shortestSide * 0.05
        ..color = color.withValues(alpha: 0.95);
      canvas.drawArc(
        loaderRect,
        (phase * math.pi * 2) - math.pi / 3,
        math.pi / 1.6,
        false,
        loaderPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _RoseCurvePainter oldDelegate) {
    return color != oldDelegate.color ||
        secondaryColor != oldDelegate.secondaryColor ||
        phase != oldDelegate.phase ||
        glowStrength != oldDelegate.glowStrength ||
        showLoaderArc != oldDelegate.showLoaderArc;
  }
}

class AskAxonOrbButton extends StatefulWidget {
  final VoidCallback onTap;
  final bool isWorking;
  final bool enableLongPressBuildUp;
  final bool enabled;
  final double size;
  final Color backgroundColor;
  final Color borderColor;
  final Color? arrowColor;
  final EdgeInsets padding;
  final List<BoxShadow>? boxShadow;
  final VoidCallback? onSendingComplete;
  final bool usePaperPlane;
  final bool useInstantSend;
  final bool isExpandable;
  final TextEditingController? textController;
  final VoidCallback? onExpand;
  final VoidCallback? onCollapse;

  const AskAxonOrbButton({
    super.key,
    required this.onTap,
    this.isWorking = false,
    this.enableLongPressBuildUp = false,
    this.enabled = true,
    this.size = 56,
    this.backgroundColor = const Color(0xFF0D0D0D),
    this.borderColor = const Color(0x2E3A86FF),
    this.arrowColor,
    this.padding = const EdgeInsets.all(8),
    this.boxShadow,
    this.onSendingComplete,
    this.usePaperPlane = false,
    this.useInstantSend = false,
    this.isExpandable = false,
    this.textController,
    this.onExpand,
    this.onCollapse,
  });

  @override
  State<AskAxonOrbButton> createState() => _AskAxonOrbButtonState();
}

class _AskAxonOrbButtonState extends State<AskAxonOrbButton>
    with TickerProviderStateMixin {
  late final AnimationController _morphController;
  late final AnimationController _expandController;
  Timer? _buildUpTimer;
  int _replaySeed = 0;
  bool _playBuildUp = false;
  bool _hasPlayedBuildUp = false;
  bool _isSending = false;
  bool _isExpanded = false;
  MorphState _morphState = MorphState.sphere;

  @override
  void initState() {
    super.initState();
    _morphController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
      reverseDuration: const Duration(milliseconds: 300),
    );
    _expandController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
      reverseDuration: const Duration(milliseconds: 250),
    );

    if (widget.useInstantSend) {
      _morphController.value = 1.0;
      _morphState = MorphState.icon;
    }
  }

  @override
  void didUpdateWidget(AskAxonOrbButton oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.useInstantSend != oldWidget.useInstantSend) {
      if (widget.useInstantSend) {
        _morphController.value = 1.0;
        _morphState = MorphState.icon;
      } else {
        _morphController.value = 0.0;
        _morphState = MorphState.sphere;
      }
    }
  }

  @override
  void dispose() {
    _buildUpTimer?.cancel();
    _morphController.dispose();
    _expandController.dispose();
    super.dispose();
  }

  void _toggleExpand() {
    setState(() {
      _isExpanded = !_isExpanded;
    });
    if (_isExpanded) {
      _expandController.forward();
      widget.onExpand?.call();
    } else {
      _expandController.reverse();
      widget.onCollapse?.call();
    }
  }

  Future<void> _handleTap() async {
    if (!widget.enabled) return;

    if (widget.useInstantSend) {
      HapticFeedback.lightImpact();
      widget.onTap();
      return;
    }

    if (_morphController.isAnimating) return;

    setState(() {
      _isSending = true;
      _morphState = MorphState.morphing;
    });

    await _morphController.forward();

    setState(() => _morphState = MorphState.icon);
    HapticFeedback.mediumImpact();

    widget.onTap();

    Future.delayed(const Duration(milliseconds: 600), () {
      if (!mounted) return;
      widget.onSendingComplete?.call();
      setState(() {
        _isSending = false;
        _morphState = MorphState.sphere;
      });
      _morphController.reverse();
    });
  }

  void _handleLongPress() {
    if (!widget.enableLongPressBuildUp || _hasPlayedBuildUp) return;
    debugPrint('AskAxonOrbButton: Long press triggered');
    HapticFeedback.mediumImpact();
    _buildUpTimer?.cancel();
    setState(() {
      _replaySeed += 1;
      _playBuildUp = true;
      _hasPlayedBuildUp = true;
    });
    _buildUpTimer = Timer(const Duration(milliseconds: 2800), () {
      if (!mounted) return;
      debugPrint('AskAxonOrbButton: Timer fired, calling onTap');
      widget.onTap();
      setState(() {
        _playBuildUp = false;
        _hasPlayedBuildUp = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    final shadows = widget.boxShadow ??
        [
          BoxShadow(
            color: const Color(0xFF3A86FF).withValues(alpha: 0.16),
            blurRadius: 24,
            offset: const Offset(0, 8),
          ),
        ];

    Widget buildLogoButton() {
      return Container(
        width: widget.size,
        height: widget.size,
        padding: widget.padding,
        decoration: BoxDecoration(
          color: widget.backgroundColor,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: widget.borderColor),
          boxShadow: shadows,
        ),
        child: AskAxonLogo(
          size: widget.size - (widget.padding.horizontal / 2),
          isWorking: widget.isWorking,
          morphProgress: _morphController.value,
          playBuildUp: _playBuildUp,
          replaySeed: _replaySeed,
          hapticEnabled: widget.enableLongPressBuildUp,
          isSending: _isSending,
          plasmaColor: const Color(0xFF3A86FF),
          arrowColor: widget.arrowColor ?? Colors.white,
          morphState: _morphState,
          longPressMode: widget.enableLongPressBuildUp,
        ),
      );
    }

    if (widget.isExpandable) {
      return AnimatedBuilder(
        animation: _expandController,
        builder: (context, _) {
          final expandWidth = 200 + (_expandController.value * 80);
          return Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: _isExpanded ? expandWidth : 0,
                height: 48,
                decoration: BoxDecoration(
                  color: AxonColors.surfaceElevated,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: _isExpanded
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(24),
                        child: TextField(
                          controller: widget.textController,
                          style: const TextStyle(
                              color: Colors.white, fontSize: 14),
                          decoration: const InputDecoration(
                            hintText: 'Ask Axon...',
                            hintStyle: TextStyle(color: Color(0xFF666666)),
                            border: InputBorder.none,
                            contentPadding:
                                EdgeInsets.symmetric(horizontal: 16),
                          ),
                          onSubmitted: (_) => _handleTap(),
                        ),
                      )
                    : const SizedBox.shrink(),
              ),
              const SizedBox(width: 12),
              GestureDetector(
                onTap: widget.enabled
                    ? () {
                        if (_isExpanded &&
                            widget.textController?.text.trim().isNotEmpty ==
                                true) {
                          _handleTap();
                        } else {
                          _toggleExpand();
                        }
                      }
                    : null,
                child: buildLogoButton(),
              ),
            ],
          );
        },
      );
    }

    return GestureDetector(
      onTap: widget.enabled ? _handleTap : null,
      onLongPress: widget.enabled && widget.enableLongPressBuildUp
          ? _handleLongPress
          : null,
      child: AnimatedBuilder(
        animation: _morphController,
        builder: (context, _) => buildLogoButton(),
      ),
    );
  }
}

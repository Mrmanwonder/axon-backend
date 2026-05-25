import 'dart:ui';
import 'package:flutter/material.dart';

/// Global tracker for animations that should only play once per app session.
/// This persists as long as the app process is alive.
final Set<String> _sessionAnimationTracker = {};

class BlurRevealWidget extends StatefulWidget {
  final Widget child;
  final Duration duration;
  final Duration delay;
  final double blurAmount;
  final bool autoPlay;
  final String? sessionKey; // If provided, animation only plays once per session

  const BlurRevealWidget({
    super.key,
    required this.child,
    this.duration = const Duration(milliseconds: 600),
    this.delay = Duration.zero,
    this.blurAmount = 20.0,
    this.autoPlay = true,
    this.sessionKey,
  });

  @override
  State<BlurRevealWidget> createState() => _BlurRevealWidgetState();
}

class _BlurRevealWidgetState extends State<BlurRevealWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _blurAnimation;
  late Animation<double> _opacityAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.duration,
    );

    _blurAnimation = Tween<double>(
      begin: widget.blurAmount,
      end: 0.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));

    _opacityAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));

    final shouldPlay = _shouldAnimate();

    if (shouldPlay && widget.autoPlay) {
      if (widget.sessionKey != null) {
        _sessionAnimationTracker.add(widget.sessionKey!);
      }
      Future.delayed(widget.delay, () {
        if (mounted) {
          _controller.forward();
        }
      });
    } else {
      _controller.value = 1.0;
    }
  }

  bool _shouldAnimate() {
    if (widget.sessionKey == null) return true;
    return !_sessionAnimationTracker.contains(widget.sessionKey);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return ImageFiltered(
          imageFilter: ImageFilter.blur(
            sigmaX: _blurAnimation.value,
            sigmaY: _blurAnimation.value,
          ),
          child: Opacity(
            opacity: _opacityAnimation.value,
            child: widget.child,
          ),
        );
      },
    );
  }
}

class BlurFadeSlideWidget extends StatefulWidget {
  final Widget child;
  final Duration duration;
  final Duration delay;
  final Offset beginOffset;
  final double blurAmount;
  final bool autoPlay;
  final String? sessionKey; // If provided, animation only plays once per session

  const BlurFadeSlideWidget({
    super.key,
    required this.child,
    this.duration = const Duration(milliseconds: 500),
    this.delay = Duration.zero,
    this.beginOffset = const Offset(0, 0.05),
    this.blurAmount = 15.0,
    this.autoPlay = true,
    this.sessionKey,
  });

  @override
  State<BlurFadeSlideWidget> createState() => _BlurFadeSlideWidgetState();
}

class _BlurFadeSlideWidgetState extends State<BlurFadeSlideWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _blurAnimation;
  late Animation<double> _opacityAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: widget.duration,
    );

    _blurAnimation = Tween<double>(
      begin: widget.blurAmount,
      end: 0.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));

    _opacityAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));

    _slideAnimation = Tween<Offset>(
      begin: widget.beginOffset,
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    ));

    final shouldPlay = _shouldAnimate();

    if (shouldPlay && widget.autoPlay) {
      if (widget.sessionKey != null) {
        _sessionAnimationTracker.add(widget.sessionKey!);
      }
      Future.delayed(widget.delay, () {
        if (mounted) {
          _controller.forward();
        }
      });
    } else {
      _controller.value = 1.0;
    }
  }

  bool _shouldAnimate() {
    if (widget.sessionKey == null) return true;
    return !_sessionAnimationTracker.contains(widget.sessionKey);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Transform.translate(
          offset: Offset(
            _slideAnimation.value.dx * MediaQuery.of(context).size.width,
            _slideAnimation.value.dy * MediaQuery.of(context).size.height,
          ),
          child: ImageFiltered(
            imageFilter: ImageFilter.blur(
              sigmaX: _blurAnimation.value,
              sigmaY: _blurAnimation.value,
            ),
            child: Opacity(
              opacity: _opacityAnimation.value,
              child: widget.child,
            ),
          ),
        );
      },
    );
  }
}

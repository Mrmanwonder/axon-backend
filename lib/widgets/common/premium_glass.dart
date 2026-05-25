// lib/widgets/common/premium_glass.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

/// Premium Glass Card - Glassmorphic component with blur effect
class AxonGlassCard extends StatelessWidget {
  final Widget child;
  final double borderRadius;
  final double blur;
  final EdgeInsets? padding;
  final Color? glowColor;

  const AxonGlassCard({
    super.key,
    required this.child,
    this.borderRadius = 24,
    this.blur = 15,
    this.padding,
    this.glowColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(borderRadius),
        boxShadow: glowColor != null
            ? [
                BoxShadow(
                  color: glowColor!.withValues(alpha: 0.3),
                  blurRadius: 20,
                  spreadRadius: 0,
                ),
              ]
            : null,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(borderRadius),
        child: BackdropFilter(
          filter: ImageFilter.blur(
            sigmaX: blur < 28 ? 28 : blur,
            sigmaY: blur < 28 ? 28 : blur,
          ),
          child: Container(
            padding: padding,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(borderRadius),
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  SpatialColors.glassHighlight,
                  SpatialColors.glassWhite,
                  SpatialColors.glassLowlight,
                ],
                stops: const [0.0, 0.44, 1.0],
              ),
              border: Border.all(
                color: SpatialColors.glassBorder,
                width: 1.1,
              ),
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}

/// Premium Glass Card with multi-layer shadows and gradient
class AxonPremiumGlassCard extends StatelessWidget {
  final Widget child;
  final double borderRadius;
  final EdgeInsets? padding;
  final Color? accentColor;

  const AxonPremiumGlassCard({
    super.key,
    required this.child,
    this.borderRadius = 28,
    this.padding,
    this.accentColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(borderRadius),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.15),
            blurRadius: 40,
            offset: const Offset(0, 20),
          ),
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.1),
            blurRadius: 1,
            offset: const Offset(0, 1),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(borderRadius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 32, sigmaY: 32),
          child: Container(
            padding: padding,
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  SpatialColors.glassHighlight,
                  SpatialColors.glassWhite,
                  SpatialColors.glassLowlight,
                ],
                stops: const [0.0, 0.44, 1.0],
              ),
              borderRadius: BorderRadius.circular(borderRadius),
              border: Border.all(
                width: 1.1,
                color: SpatialColors.glassBorder,
              ),
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}

/// Glowing Glass Card with animated effects
class AxonGlowingGlassCard extends StatelessWidget {
  final Widget child;
  final Color glowColor;
  final double borderRadius;
  final bool isActive;
  final EdgeInsets? padding;

  const AxonGlowingGlassCard({
    super.key,
    required this.child,
    required this.glowColor,
    this.borderRadius = 24,
    this.isActive = true,
    this.padding,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutCubic,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(borderRadius),
        boxShadow: isActive
            ? [
                BoxShadow(
                  color: glowColor.withValues(alpha: 0.4),
                  blurRadius: 24,
                  spreadRadius: 0,
                ),
                BoxShadow(
                  color: glowColor.withValues(alpha: 0.2),
                  blurRadius: 60,
                  spreadRadius: -10,
                ),
              ]
            : [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.1),
                  blurRadius: 20,
                  offset: const Offset(0, 8),
                ),
              ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(borderRadius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 30, sigmaY: 30),
          child: Container(
            padding: padding,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(borderRadius),
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: isActive
                    ? [
                        glowColor.withValues(alpha: 0.22),
                        SpatialColors.glassWhite,
                        SpatialColors.glassLowlight,
                      ]
                    : [
                        SpatialColors.glassHighlight,
                        SpatialColors.glassWhite,
                        SpatialColors.glassLowlight,
                      ],
                stops: const [0.0, 0.46, 1.0],
              ),
              border: Border.all(
                width: 1.1,
                color: isActive
                    ? glowColor.withValues(alpha: 0.5)
                    : SpatialColors.glassBorder,
              ),
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}

import 'dart:ui';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../services/haptics_service.dart';
import '../services/app_state.dart';
import '../theme/app_theme.dart';

class FloatingGlassNavbar extends ConsumerStatefulWidget {
  final int currentIndex;

  const FloatingGlassNavbar({
    super.key,
    required this.currentIndex,
  });

  @override
  ConsumerState<FloatingGlassNavbar> createState() =>
      _FloatingGlassNavbarState();
}

class _FloatingGlassNavbarState extends ConsumerState<FloatingGlassNavbar>
    with SingleTickerProviderStateMixin {
  late AnimationController _slideController;
  int _previousIndex = 0;

  static const List<String> _appRoutes = [
    '/home',
    '/analysis',
    '/pdf',
    '/study',
    '/settings',
  ];

  static const List<IconData> _appIcons = [
    Icons.grid_view_rounded,
    Icons.analytics_outlined,
    Icons.event_note_rounded,
    Icons.auto_stories_outlined,
    Icons.settings_outlined,
  ];

  static const List<IconData> _examPlannerIcons = [
    Icons.insights_rounded,
    Icons.flag_rounded,
    Icons.folder_rounded,
    Icons.task_alt_rounded,
    Icons.functions_rounded,
  ];

  static const List<String> _examPlannerLabels = [
    'DATA',
    'GOALS',
    'VAULT',
    'TASKS',
    'MATH',
  ];

  @override
  void initState() {
    super.initState();
    _previousIndex = widget.currentIndex;
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
  }

  @override
  void didUpdateWidget(FloatingGlassNavbar oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.currentIndex != widget.currentIndex) {
      _previousIndex = oldWidget.currentIndex;
      _slideController.forward(from: 0);
    }
  }

  @override
  void dispose() {
    _slideController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final flipValue = ref.watch(examPlannerFlipProvider);

    return SafeArea(
      bottom: true,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(24, 0, 24, 16),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(24),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 32, sigmaY: 32),
            child: Container(
              height: 72,
              decoration: BoxDecoration(
                color: SpatialColors.charcoalLight.withValues(alpha: 0.86),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(
                  color: Colors.white.withValues(alpha: 0.08),
                  width: 1.1,
                ),
                boxShadow: SpatialGlow.glassDock,
              ),
              child: Stack(
                children: [
                  AnimatedBuilder(
                    animation: _slideController,
                    builder: (context, child) {
                      return _SlidingHighlight(
                        fromIndex: _previousIndex,
                        toIndex: widget.currentIndex,
                        progress: Curves.easeOutCubic
                            .transform(_slideController.value),
                        flipValue: flipValue,
                        child: child!,
                      );
                    },
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: List.generate(_appRoutes.length, (index) {
                        final isSelected = widget.currentIndex == index;
                        return _FlippingNavbarItem(
                          appIcon: _appIcons[index],
                          examPlannerIcon: _examPlannerIcons[index],
                          examPlannerLabel: _examPlannerLabels[index],
                          isSelected: isSelected,
                          flipValue: flipValue,
                          staggerIndex: index,
                          onTap: () {
                            AxonHaptics.mediumImpact();
                            context.go(_appRoutes[index]);
                          },
                          onLongPress: index == 3
                              ? () {
                                  AxonHaptics.heavyImpact();
                                  ref
                                      .read(dailyPlanServiceProvider)
                                      .runDailyBuild();
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text('Starting Daily Plan...'),
                                      behavior: SnackBarBehavior.floating,
                                    ),
                                  );
                                }
                              : null,
                        );
                      }),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _SlidingHighlight extends StatelessWidget {
  final int fromIndex;
  final int toIndex;
  final double progress;
  final double flipValue;
  final Widget child;

  const _SlidingHighlight({
    required this.fromIndex,
    required this.toIndex,
    required this.progress,
    required this.flipValue,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final horizontalPadding = 24.0;
    final availableWidth = screenWidth - (horizontalPadding * 2);
    final itemCount = 5;
    final itemWidth = availableWidth / itemCount;
    final highlightWidth = itemWidth * 0.9;
    final spacingWidth = itemWidth - highlightWidth;
    final fromHighlightLeft = (fromIndex * itemWidth) + (spacingWidth / 2);
    final toHighlightLeft = (toIndex * itemWidth) + (spacingWidth / 2);
    final currentHighlightLeft =
        fromHighlightLeft + (toHighlightLeft - fromHighlightLeft) * progress;

    return Stack(
      children: [
        child,
        Positioned(
          left: currentHighlightLeft,
          top: 8,
          child: AnimatedContainer(
            duration: Duration.zero,
            width: highlightWidth,
            height: 56,
            decoration: BoxDecoration(
              color: AxonThemeMode.isDark
                  ? Colors.white.withValues(
                      alpha: (0.08 + (0.04 * flipValue)).clamp(0.0, 1.0))
                  : AxonColors.accent.withValues(
                      alpha: (0.08 + (0.08 * flipValue)).clamp(0.0, 1.0)),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: AxonThemeMode.isDark
                    ? Colors.white.withValues(
                        alpha: (0.1 + (0.05 * flipValue)).clamp(0.0, 1.0))
                    : AxonColors.accent.withValues(
                        alpha: (0.25 + (0.15 * flipValue)).clamp(0.0, 1.0)),
                width: 0.5,
              ),
              boxShadow: [
                BoxShadow(
                  color: AxonColors.accent
                      .withValues(alpha: (0.1 * flipValue).clamp(0.0, 1.0)),
                  blurRadius: 8 * flipValue,
                  offset: const Offset(0, -2),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _FlippingNavbarItem extends StatelessWidget {
  final IconData appIcon;
  final IconData examPlannerIcon;
  final String examPlannerLabel;
  final bool isSelected;
  final double flipValue;
  final int staggerIndex;
  final VoidCallback onTap;
  final VoidCallback? onLongPress;

  const _FlippingNavbarItem({
    required this.appIcon,
    required this.examPlannerIcon,
    required this.examPlannerLabel,
    required this.isSelected,
    required this.flipValue,
    required this.staggerIndex,
    required this.onTap,
    this.onLongPress,
  });

  static const _perspective = 0.003;

  double _getEffectiveFlip() {
    return flipValue.clamp(0.0, 1.0);
  }

  @override
  Widget build(BuildContext context) {
    final effectiveFlip = _getEffectiveFlip();
    final showNewIcon = effectiveFlip > 0.5;
    final angle = effectiveFlip * math.pi;

    final labelOpacity =
        ((effectiveFlip - 0.3).clamp(0.0, 1.0) * 2.0).clamp(0.0, 1.0);

    return Expanded(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onTap,
        onLongPress: onLongPress,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Transform(
              alignment: Alignment.center,
              transform: Matrix4.identity()
                ..setEntry(3, 2, _perspective)
                ..rotateY(angle),
              child: Icon(
                showNewIcon ? examPlannerIcon : appIcon,
                size: 22,
                color: _getIconColor(showNewIcon),
              ),
            ),
            if (labelOpacity > 0)
              Opacity(
                opacity: labelOpacity,
                child: Container(
                  margin: const EdgeInsets.only(top: 4),
                  child: Text(
                    examPlannerLabel,
                    style: TextStyle(
                      fontSize: 8,
                      fontWeight:
                          isSelected ? FontWeight.w700 : FontWeight.w500,
                      color: isSelected
                          ? AxonColors.accent
                          : Colors.white.withValues(alpha: 0.5),
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Color _getIconColor(bool showNewIcon) {
    if (AxonThemeMode.isDark) {
      if (isSelected) {
        return showNewIcon ? AxonColors.accent : Colors.white;
      }
      return Colors.white38;
    } else {
      if (isSelected) {
        return showNewIcon ? AxonColors.accent : AxonColors.accent;
      }
      return AxonColors.textTertiary;
    }
  }
}

import 'package:flutter/material.dart';

class SyllabusMasteryWidget extends StatefulWidget {
  final List<int> activityLevels;
  final int days;
  final Function(int)? onNodeTap;
  final VoidCallback? onTap;

  const SyllabusMasteryWidget({
    super.key,
    required this.activityLevels,
    this.days = 35,
    this.onNodeTap,
    this.onTap,
  });

  @override
  State<SyllabusMasteryWidget> createState() => _SyllabusMasteryWidgetState();
}

class _SyllabusMasteryWidgetState extends State<SyllabusMasteryWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 2500),
      vsync: this,
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.7, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final levels = widget.activityLevels.length >= widget.days
        ? widget.activityLevels
        : List.generate(widget.days, (i) => 0);

    return GestureDetector(
      onTap: widget.onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF000000),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: const Color(0xFF1A1A1A)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'SYLLABUS NODES',
                  style: TextStyle(
                    color: Color(0xFF8B949E),
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 2,
                  ),
                ),
                const Spacer(),
                _Legend(),
              ],
            ),
            const SizedBox(height: 12),
            _MasteryGrid(
              levels: levels,
              pulseAnimation: _pulseAnimation,
              onNodeTap: widget.onNodeTap,
            ),
          ],
        ),
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _LegendItem(color: const Color(0xFF161B22), label: '0'),
        const SizedBox(width: 4),
        _LegendItem(color: const Color(0xFF1E3A5F), label: '1'),
        const SizedBox(width: 4),
        _LegendItem(color: const Color(0xFF3A86FF), label: '2'),
        const SizedBox(width: 4),
        _LegendItem(
            color: const Color(0xFF60A5FA), label: '3', isGlowing: true),
      ],
    );
  }
}

class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;
  final bool isGlowing;

  const _LegendItem({
    required this.color,
    required this.label,
    this.isGlowing = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 12,
      height: 12,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(2),
        boxShadow: isGlowing
            ? [
                BoxShadow(
                  color: color.withValues(alpha: 0.5),
                  blurRadius: 6,
                  spreadRadius: 1,
                ),
              ]
            : null,
      ),
    );
  }
}

class _MasteryGrid extends StatelessWidget {
  final List<int> levels;
  final Animation<double> pulseAnimation;
  final Function(int)? onNodeTap;

  const _MasteryGrid({
    required this.levels,
    required this.pulseAnimation,
    this.onNodeTap,
  });

  @override
  Widget build(BuildContext context) {
    final columns = 7;
    final rows = (levels.length / columns).ceil();

    return Column(
      children: [
        Row(
          children: ['', 'M', '', 'W', '', 'F', '']
              .map((day) => SizedBox(
                    width: 18,
                    child: Text(
                      day,
                      style: const TextStyle(
                        color: Color(0xFF666666),
                        fontSize: 8,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ))
              .toList(),
        ),
        const SizedBox(height: 4),
        ...List.generate(rows, (row) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 3),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: List.generate(columns, (col) {
                final index = row * columns + col;
                if (index >= levels.length) {
                  return const SizedBox(width: 18, height: 18);
                }
                return _MasteryNode(
                  level: levels[index],
                  isToday: index == levels.length - 1,
                  pulseAnimation: pulseAnimation,
                  onTap: () => onNodeTap?.call(index),
                );
              }),
            ),
          );
        }),
      ],
    );
  }
}

class _MasteryNode extends StatelessWidget {
  final int level;
  final bool isToday;
  final Animation<double> pulseAnimation;
  final VoidCallback? onTap;

  const _MasteryNode({
    required this.level,
    required this.isToday,
    required this.pulseAnimation,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final colors = [
      const Color(0xFF161B22), // Level 0 - Not Started
      const Color(0xFF1E3A5F), // Level 1 - Read
      const Color(0xFF3A86FF), // Level 2 - Practice Done
      const Color(0xFF60A5FA), // Level 3 - Mastered
    ];

    final color = colors[level.clamp(0, 3)];
    final hasGlow = level >= 3;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedBuilder(
        animation: pulseAnimation,
        builder: (context, child) {
          return Container(
            width: 18,
            height: 18,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(3),
              boxShadow: isToday && level > 0
                  ? [
                      BoxShadow(
                        color:
                            color.withValues(alpha: 0.6 * pulseAnimation.value),
                        blurRadius: 8 * pulseAnimation.value,
                        spreadRadius: 2 * pulseAnimation.value,
                      ),
                    ]
                  : hasGlow
                      ? [
                          BoxShadow(
                            color: color.withValues(alpha: 0.4),
                            blurRadius: 6,
                            spreadRadius: 1,
                          ),
                        ]
                      : null,
            ),
            child: isToday && level == 0
                ? Center(
                    child: Container(
                      width: 6,
                      height: 6,
                      decoration: BoxDecoration(
                        color: const Color(0xFF3A86FF),
                        shape: BoxShape.circle,
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFF3A86FF)
                                .withValues(alpha: pulseAnimation.value * 0.8),
                            blurRadius: 4,
                          ),
                        ],
                      ),
                    ),
                  )
                : null,
          );
        },
      ),
    );
  }
}

class CompactMasteryWidget extends StatelessWidget {
  final List<int> activityLevels;
  final double masteryPercentage;
  final VoidCallback? onTap;

  const CompactMasteryWidget({
    super.key,
    required this.activityLevels,
    required this.masteryPercentage,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final recentLevels = activityLevels.length >= 7
        ? activityLevels.sublist(activityLevels.length - 7)
        : List.generate(7, (i) => 0);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF000000),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFF1A1A1A)),
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'MASTERY',
                    style: TextStyle(
                      color: Color(0xFF8B949E),
                      fontSize: 8,
                      letterSpacing: 1,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${(masteryPercentage * 100).round()}%',
                    style: const TextStyle(
                      color: Color(0xFFFFFFFF),
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
            Row(
              children:
                  recentLevels.map((level) => _MiniNode(level: level)).toList(),
            ),
          ],
        ),
      ),
    );
  }
}

class _MiniNode extends StatelessWidget {
  final int level;

  const _MiniNode({required this.level});

  @override
  Widget build(BuildContext context) {
    final colors = [
      const Color(0xFF161B22),
      const Color(0xFF1E3A5F),
      const Color(0xFF3A86FF),
      const Color(0xFF60A5FA),
    ];

    return Container(
      width: 8,
      height: 8,
      margin: const EdgeInsets.symmetric(horizontal: 1),
      decoration: BoxDecoration(
        color: colors[level.clamp(0, 3)],
        borderRadius: BorderRadius.circular(2),
      ),
    );
  }
}

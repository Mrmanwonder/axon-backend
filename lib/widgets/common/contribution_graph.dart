import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../services/study_activity_service.dart';
import 'rose_loader.dart';
import '../../models/study_activity.dart';

class ContributionGraph extends StatelessWidget {
  final List<int> activityLevels;
  final int rows;
  final int columns;
  final double cellSize;
  final double spacing;
  final bool showLabels;
  final VoidCallback? onTap;

  const ContributionGraph({
    super.key,
    required this.activityLevels,
    this.rows = 7,
    this.columns = 5,
    this.cellSize = 12,
    this.spacing = 3,
    this.showLabels = true,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final levels = activityLevels.length >= rows * columns
        ? activityLevels
        : List.generate(rows * columns, (i) => 0);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF000000),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (showLabels) ...[
              Column(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildDayLabel(''),
                  _buildDayLabel('M'),
                  _buildDayLabel(''),
                  _buildDayLabel('W'),
                  _buildDayLabel(''),
                  _buildDayLabel('F'),
                  _buildDayLabel(''),
                ],
              ),
              const SizedBox(width: 8),
            ],
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: List.generate(rows, (row) {
                  return Padding(
                    padding:
                        EdgeInsets.only(bottom: row < rows - 1 ? spacing : 0),
                    child: Row(
                      children: List.generate(columns, (col) {
                        final index = col * rows + row;
                        final level = index < levels.length ? levels[index] : 0;
                        return Padding(
                          padding: EdgeInsets.only(
                              right: col < columns - 1 ? spacing : 0),
                          child: _ActivityCell(level: level, size: cellSize),
                        );
                      }),
                    ),
                  );
                }),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDayLabel(String text) {
    return SizedBox(
      height: cellSize,
      child: Text(
        text,
        style: const TextStyle(
          color: Color(0xFF8B949E),
          fontSize: 10,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
}

class _ActivityCell extends StatelessWidget {
  final int level;
  final double size;

  const _ActivityCell({required this.level, required this.size});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: Color(ActivityColors.colorList[level.clamp(0, 3)]),
        borderRadius: BorderRadius.circular(2),
      ),
    );
  }
}

class ContributionGraphWidget extends StatefulWidget {
  final bool showAnimation;
  final VoidCallback? onTap;

  const ContributionGraphWidget({
    super.key,
    this.showAnimation = true,
    this.onTap,
  });

  @override
  State<ContributionGraphWidget> createState() =>
      _ContributionGraphWidgetState();
}

class _ContributionGraphWidgetState extends State<ContributionGraphWidget> {
  List<int> _activityLevels = [];
  bool _isLoading = true;
  final _service = StudyActivityService();

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    await _service.initialize();
    setState(() {
      _activityLevels = _service.getRecentActivityLevels(days: 35);
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF000000),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Center(
          child: RoseLoader(size: 24, color: Color(0xFF3A86FF)),
        ),
      );
    }

    return GestureDetector(
      onTap: () {
        HapticFeedback.selectionClick();
        widget.onTap?.call();
      },
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF000000),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: const Color(0xFF1A1A1A)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'CONTRIBUTION',
                  style: TextStyle(
                    color: Color(0xFF8B949E),
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 2,
                  ),
                ),
                const Spacer(),
                _buildLegend(),
              ],
            ),
            const SizedBox(height: 8),
            _ContributionGrid(
              levels: _activityLevels,
              animate: widget.showAnimation,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegend() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text(
          'L',
          style: TextStyle(color: Color(0xFF444444), fontSize: 8),
        ),
        const SizedBox(width: 2),
        ...List.generate(
          4,
          (i) => Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.symmetric(horizontal: 1),
            decoration: BoxDecoration(
              color: Color(ActivityColors.colorList[i]),
              borderRadius: BorderRadius.circular(2),
            ),
          ),
        ),
        const SizedBox(width: 2),
        const Text(
          'H',
          style: TextStyle(color: Color(0xFF444444), fontSize: 8),
        ),
      ],
    );
  }
}

class _ContributionGrid extends StatelessWidget {
  final List<int> levels;
  final bool animate;

  const _ContributionGrid({required this.levels, this.animate = true});

  @override
  Widget build(BuildContext context) {
    const columns = 7;
    const rows = 5;
    final gridLevels = levels.length >= columns * rows
        ? levels
        : List.generate(columns * rows, (i) => 0);

    return Column(
      children: [
        Row(
          children: ['', 'M', '', 'W', '', 'F', '']
              .map(
                (day) => SizedBox(
                  width: 14,
                  child: Text(
                    day,
                    style: const TextStyle(
                      color: Color(0xFF444444),
                      fontSize: 8,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              )
              .toList(),
        ),
        const SizedBox(height: 2),
        ...List.generate(rows, (row) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 2),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: List.generate(columns, (col) {
                final index = row * columns + col;
                final level = index < gridLevels.length ? gridLevels[index] : 0;
                return _ActivityCell(level: level, size: 14);
              }),
            ),
          );
        }),
      ],
    );
  }
}

class AnimatedContributionGraph extends StatefulWidget {
  final List<int>? activityLevels;
  final Duration animationDuration;
  final VoidCallback? onTap;

  const AnimatedContributionGraph({
    super.key,
    this.activityLevels,
    this.animationDuration = const Duration(milliseconds: 800),
    this.onTap,
  });

  @override
  State<AnimatedContributionGraph> createState() =>
      _AnimatedContributionGraphState();
}

class _AnimatedContributionGraphState extends State<AnimatedContributionGraph>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;
  List<int> _levels = [];
  bool _isLoading = true;
  final _service = StudyActivityService();

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.animationDuration,
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _loadData();
  }

  Future<void> _loadData() async {
    await _service.initialize();
    setState(() {
      _levels =
          widget.activityLevels ?? _service.getRecentActivityLevels(days: 35);
      _isLoading = false;
    });
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const SizedBox(
        height: 80,
        child: Center(
          child: RoseLoader(size: 24, color: Color(0xFF3A86FF)),
        ),
      );
    }

    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Transform.scale(
          scale: 0.8 + (0.2 * _animation.value),
          child: Opacity(
            opacity: _animation.value,
            child: GestureDetector(
              onTap: () {
                HapticFeedback.selectionClick();
                widget.onTap?.call();
              },
              child: ContributionGraph(
                activityLevels: _levels,
                rows: 5,
                columns: 7,
                cellSize: 14,
                spacing: 2,
              ),
            ),
          ),
        );
      },
    );
  }
}

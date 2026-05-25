import 'package:flutter/material.dart';

class AxonSkeletonLoader extends StatefulWidget {
  final double width;
  final double height;
  final double borderRadius;
  final Color baseColor;

  const AxonSkeletonLoader({
    super.key,
    this.width = double.infinity,
    this.height = 16,
    this.borderRadius = 8,
    this.baseColor = const Color(0xFF141414),
  });

  @override
  State<AxonSkeletonLoader> createState() => _AxonSkeletonLoaderState();
}

class _AxonSkeletonLoaderState extends State<AxonSkeletonLoader>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
    _animation = Tween<double>(begin: -2, end: 2).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOutSine),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final highlightColor = const Color(0xFF2A2A2A);
    final baseColor = widget.baseColor;

    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Container(
          width: widget.width,
          height: widget.height,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(widget.borderRadius),
            gradient: LinearGradient(
              begin: Alignment(_animation.value, 0),
              end: Alignment(_animation.value + 1, 0),
              colors: [
                baseColor,
                highlightColor,
                baseColor,
              ],
              stops: const [0.0, 0.5, 1.0],
            ),
          ),
        );
      },
    );
  }
}

class AxonSkeletonText extends StatelessWidget {
  final double width;
  final int lines;
  final double lineHeight;
  final double spacing;
  final double borderRadius;

  const AxonSkeletonText({
    super.key,
    this.width = double.infinity,
    this.lines = 3,
    this.lineHeight = 14,
    this.spacing = 8,
    this.borderRadius = 6,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: List.generate(lines, (index) {
        final isLast = index == lines - 1;
        return Padding(
          padding: EdgeInsets.only(bottom: isLast ? 0 : spacing),
          child: AxonSkeletonLoader(
            width: isLast ? width * 0.6 : width,
            height: lineHeight,
            borderRadius: borderRadius,
          ),
        );
      }),
    );
  }
}

class AxonSkeletonCard extends StatelessWidget {
  final double? width;
  final double height;
  final double padding;
  final double borderRadius;

  const AxonSkeletonCard({
    super.key,
    this.width,
    this.height = 120,
    this.padding = 16,
    this.borderRadius = 12,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      padding: EdgeInsets.all(padding),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: BorderRadius.circular(borderRadius),
        border: Border.all(color: const Color(0xFF2A2A2A), width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const AxonSkeletonLoader(
            width: 80,
            height: 20,
            borderRadius: 4,
          ),
          const SizedBox(height: 12),
          const AxonSkeletonText(lines: 2, lineHeight: 12, spacing: 6),
          const Spacer(),
          Row(
            children: [
              const AxonSkeletonLoader(width: 60, height: 24, borderRadius: 12),
              const Spacer(),
              const AxonSkeletonLoader(width: 40, height: 24, borderRadius: 12),
            ],
          ),
        ],
      ),
    );
  }
}

class AxonSkeletonListTile extends StatelessWidget {
  final double height;
  final bool showAvatar;
  final double avatarSize;

  const AxonSkeletonListTile({
    super.key,
    this.height = 72,
    this.showAvatar = true,
    this.avatarSize = 48,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          if (showAvatar) ...[
            const AxonSkeletonLoader(
              width: 48,
              height: 48,
              borderRadius: 24,
            ),
            const SizedBox(width: 12),
          ],
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                AxonSkeletonLoader(height: 16, borderRadius: 4),
                SizedBox(height: 8),
                AxonSkeletonLoader(
                  width: 120,
                  height: 12,
                  borderRadius: 4,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class AxonSkeletonCircle extends StatelessWidget {
  final double size;

  const AxonSkeletonCircle({
    super.key,
    this.size = 40,
  });

  @override
  Widget build(BuildContext context) {
    return AxonSkeletonLoader(
      width: size,
      height: size,
      borderRadius: size / 2,
    );
  }
}

class AxonSkeletonGrid extends StatelessWidget {
  final int itemCount;
  final int crossAxisCount;
  final double itemHeight;
  final double spacing;

  const AxonSkeletonGrid({
    super.key,
    this.itemCount = 6,
    this.crossAxisCount = 2,
    this.itemHeight = 160,
    this.spacing = 12,
  });

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount,
        mainAxisSpacing: spacing,
        crossAxisSpacing: spacing,
        childAspectRatio: 1,
      ),
      itemCount: itemCount,
      itemBuilder: (context, index) => const AxonSkeletonCard(
        height: 160,
        borderRadius: 12,
      ),
    );
  }
}

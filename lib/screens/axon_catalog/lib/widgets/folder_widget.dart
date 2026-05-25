import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../controllers/catalog_controller.dart';
import '../models/paper_series.dart';
import '../painters/folder_painter.dart';

// ── Public layout constants (used by catalog_screen too) ──────────────────

const double kFolderWidth  = 204.0;
const double kFolderHeight = 134.0;

// ─────────────────────────────────────────────────────────────────────────────
//  FolderWidget
// ─────────────────────────────────────────────────────────────────────────────

class FolderWidget extends StatefulWidget {
  final PaperSeries series;
  final int index;
  final bool isCenter;
  final bool isOpen;
  final double depth;

  const FolderWidget({
    super.key,
    required this.series,
    required this.index,
    required this.isCenter,
    required this.isOpen,
    required this.depth,
  });

  @override
  State<FolderWidget> createState() => _FolderWidgetState();
}

class _FolderWidgetState extends State<FolderWidget>
    with SingleTickerProviderStateMixin {
  // ── Bounce animation ───────────────────────────────────────────────────────
  late final AnimationController _bounceCtrl;
  late final Animation<double> _bounceAnim;

  // ── Global key for screen-position lookup ──────────────────────────────────
  final GlobalKey _widgetKey = GlobalKey();

  @override
  void initState() {
    super.initState();

    _bounceCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 580),
    );

    // Compress → overshoot → tiny undershoot → settle
    _bounceAnim = TweenSequence<double>([
      TweenSequenceItem(
        tween: Tween(begin: 1.0, end: 0.89)
            .chain(CurveTween(curve: Curves.easeIn)),
        weight: 13,
      ),
      TweenSequenceItem(
        tween: Tween(begin: 0.89, end: 1.09)
            .chain(CurveTween(curve: Curves.easeOut)),
        weight: 27,
      ),
      TweenSequenceItem(
        tween: Tween(begin: 1.09, end: 0.97)
            .chain(CurveTween(curve: Curves.easeInOut)),
        weight: 20,
      ),
      TweenSequenceItem(
        tween: Tween(begin: 0.97, end: 1.0)
            .chain(CurveTween(curve: Curves.easeOut)),
        weight: 40,
      ),
    ]).animate(_bounceCtrl);
  }

  @override
  void dispose() {
    _bounceCtrl.dispose();
    super.dispose();
  }

  // ── Tap handling ───────────────────────────────────────────────────────────
  void _onTap() {
    _bounceCtrl.reset();
    _bounceCtrl.forward();

    final box = _widgetKey.currentContext?.findRenderObject() as RenderBox?;
    if (box == null) return;

    final screenCenter = box.localToGlobal(
      Offset(box.size.width / 2, box.size.height / 2),
    );

    context.read<CatalogController>().openFolder(widget.index, screenCenter);
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedBuilder(
        animation: _bounceAnim,
        builder: (context, child) => Transform.scale(
          scale: _bounceAnim.value,
          child: child,
        ),
        child: SizedBox(
          key: _widgetKey,
          width: kFolderWidth,
          height: kFolderHeight,
          child: CustomPaint(
            painter: FolderPainter(
              series: widget.series,
              isCenter: widget.isCenter,
              isOpen: widget.isOpen,
              depth: widget.depth,
            ),
          ),
        ),
      ),
    );
  }
}

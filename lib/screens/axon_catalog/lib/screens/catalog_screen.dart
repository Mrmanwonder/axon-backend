import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../controllers/catalog_controller.dart';
import '../models/paper.dart';
import '../models/paper_series.dart';
import '../widgets/animated_file_card.dart';
import '../widgets/folder_widget.dart';
import '../widgets/paper_detail_sheet.dart';

// ─────────────────────────────────────────────────────────────────────────────
//  CatalogScreen
// ─────────────────────────────────────────────────────────────────────────────

class CatalogScreen extends StatefulWidget {
  const CatalogScreen({super.key});
  @override
  State<CatalogScreen> createState() => _CatalogScreenState();
}

class _CatalogScreenState extends State<CatalogScreen>
    with TickerProviderStateMixin {
  CatalogController? _controller;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _controller = context.read<CatalogController>();
        _controller?.initialize(this);
      }
    });
  }

  @override
  void dispose() {
    _controller?.reset();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: Consumer<CatalogController>(
        builder: (context, ctrl, _) {
          return Stack(
            children: [
              // ── 1. Ambient background ────────────────────────────────────
              const _BackgroundGradient(),

              // ── 2. Scene (folders on diagonal) ───────────────────────────
              _SceneLayer(ctrl: ctrl),

              // ── 3. HUD overlay ───────────────────────────────────────────
              if (ctrl.initialized) _HUDLayer(ctrl: ctrl),

              // ── 4. File card explosion ────────────────────────────────────
              if (ctrl.openFolderIndex != null)
                _FileCardLayer(ctrl: ctrl),

              // ── 5. Paper detail modal ────────────────────────────────────
              if (ctrl.selectedPaper != null)
                _PaperDetailLayer(ctrl: ctrl),
            ],
          );
        },
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  1. Background
// ─────────────────────────────────────────────────────────────────────────────

class _BackgroundGradient extends StatelessWidget {
  const _BackgroundGradient();
  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: RadialGradient(
            center: const Alignment(0.15, -0.4),
            radius: 1.6,
            colors: const [Color(0xFF1C1830), Color(0xFF09090E)],
            stops: const [0.0, 0.7],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  2. Scene — perspective-transformed diagonal folder row
// ─────────────────────────────────────────────────────────────────────────────

class _SceneLayer extends StatelessWidget {
  final CatalogController ctrl;
  const _SceneLayer({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      behavior: HitTestBehavior.translucent,
      onHorizontalDragUpdate: (d) {
        if (ctrl.openFolderIndex == null) ctrl.panBy(d.delta.dx);
      },
      onHorizontalDragEnd: (_) {
        if (ctrl.openFolderIndex == null) ctrl.snapToNearest();
      },
      onTap: () {
        if (ctrl.openFolderIndex != null) ctrl.closeFolder();
      },
      child: ctrl.initialized
          ? AnimatedBuilder(
              animation: ctrl.animationListenable,
              builder: (context, _) {
                final size = MediaQuery.of(context).size;
                final zoom = ctrl.zoomValue;
                final opacity = ctrl.opacityValue.clamp(0.0, 1.0);

                return Opacity(
                  opacity: opacity,
                  child: Transform(
                    alignment: Alignment.center,
                    transform: Matrix4.identity()
                      ..setEntry(3, 2, 0.00035)
                      ..rotateX(0.22)
                      ..scale(zoom),
                    child: _DiagonalFolderRow(ctrl: ctrl, screenSize: size),
                  ),
                );
              },
            )
          : const SizedBox.expand(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Diagonal folder row
// ─────────────────────────────────────────────────────────────────────────────

class _DiagonalFolderRow extends StatelessWidget {
  final CatalogController ctrl;
  final Size screenSize;

  // Diagonal direction: right = up (matching inspo bottom-left→top-right)
  static const double kSpacingX = 218.0;
  static const double kSpacingY = -72.0;

  const _DiagonalFolderRow({required this.ctrl, required this.screenSize});

  @override
  Widget build(BuildContext context) {
    final cx = screenSize.width / 2;
    final cy = screenSize.height / 2;
    final viewOffset = ctrl.viewOffset;
    final seriesList = ctrl.allSeries;
    final breathVal = ctrl.breathValue;
    final startupDone = ctrl.startupComplete;

    return SizedBox(
      width: screenSize.width,
      height: screenSize.height,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          for (int i = 0; i < seriesList.length; i++)
            _buildFolderSlot(
              series: seriesList[i],
              index: i,
              cx: cx,
              cy: cy,
              viewOffset: viewOffset,
              breathVal: breathVal,
              startupDone: startupDone,
            ),
        ],
      ),
    );
  }

  Widget _buildFolderSlot({
    required PaperSeries series,
    required int index,
    required double cx,
    required double cy,
    required double viewOffset,
    required double breathVal,
    required bool startupDone,
  }) {
    final relIdx = index.toDouble() - viewOffset;

    // Position on diagonal
    final posX = cx + relIdx * kSpacingX - kFolderWidth / 2;
    final posY = cy + relIdx * kSpacingY - kFolderHeight / 2;

    // Depth: 1.0 at centre, min 0.38 at extremes
    final dist = relIdx.abs();
    final depth = (1.0 - dist * 0.082).clamp(0.38, 1.0);
    final folderOpacity = (0.22 + depth * 0.78).clamp(0.0, 1.0);

    final isCenter = dist < 0.65;
    final isOpen = ctrl.openFolderIndex == index;

    // Gentle idle breath only on the centre folder after startup
    final effectiveBreath = (isCenter && startupDone) ? breathVal : 1.0;

    return Positioned(
      left: posX,
      top: posY,
      child: Opacity(
        opacity: folderOpacity,
        child: Transform.scale(
          scale: depth * effectiveBreath,
          alignment: const Alignment(0, 0.08),
          child: FolderWidget(
            series: series,
            index: index,
            isCenter: isCenter,
            isOpen: isOpen,
            depth: depth,
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  3. HUD — branding, series label, subject filter, hint
// ─────────────────────────────────────────────────────────────────────────────

class _HUDLayer extends StatelessWidget {
  final CatalogController ctrl;
  const _HUDLayer({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
          child: Stack(
            children: [
              // Top-left: series info (non-interactive)
              Positioned(
                left: 0, top: 0,
                child: IgnorePointer(child: _TopLeftInfo(ctrl: ctrl)),
              ),

              // Top-center: branding (non-interactive)
              Positioned(
                top: 0, left: 0, right: 0,
                child: IgnorePointer(
                  child: Center(child: _BrandLabel()),
                ),
              ),

              // Bottom-right: subject filter
              Positioned(
                right: 0, bottom: 0,
                child: _SubjectFilter(ctrl: ctrl),
              ),

              // Bottom-left: hint
              Positioned(
                left: 0, bottom: 0,
                child: IgnorePointer(child: _HintText(ctrl: ctrl)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BrandLabel extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Text(
      'axon  ·  catalog',
      style: GoogleFonts.jetBrainsMono(
        fontSize: 11,
        color: Colors.white.withOpacity(0.20),
        letterSpacing: 3.2,
        fontWeight: FontWeight.w400,
      ),
    ).animate(delay: 3200.ms).fadeIn(duration: 900.ms);
  }
}

class _TopLeftInfo extends StatelessWidget {
  final CatalogController ctrl;
  const _TopLeftInfo({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    final open = ctrl.openSeries;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          open?.displayLabel ?? 'Cambridge  CAIE',
          style: GoogleFonts.inter(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: Colors.white.withOpacity(0.68),
            letterSpacing: 0.15,
          ),
        ),
        const SizedBox(height: 3),
        Text(
          open != null
              ? '${open.paperCount} papers  ·  '
                '${open.subjectGroups.length} subjects'
              : 'A Level  ·  AS Level  ·  2021 – 2026',
          style: GoogleFonts.jetBrainsMono(
            fontSize: 9.5,
            color: Colors.white.withOpacity(0.28),
            letterSpacing: 0.6,
          ),
        ),
      ],
    ).animate(delay: 3200.ms).fadeIn(duration: 700.ms);
  }
}

class _HintText extends StatelessWidget {
  final CatalogController ctrl;
  const _HintText({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    if (ctrl.openFolderIndex != null) return const SizedBox();
    return Text(
      'swipe  ·  tap folder to open',
      style: GoogleFonts.jetBrainsMono(
        fontSize: 9,
        color: Colors.white.withOpacity(0.16),
        letterSpacing: 0.9,
      ),
    ).animate(delay: 3800.ms).fadeIn(duration: 1000.ms);
  }
}

class _SubjectFilter extends StatelessWidget {
  final CatalogController ctrl;
  const _SubjectFilter({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    final openSeries = ctrl.openSeries;
    final isVisible = openSeries != null;

    return AnimatedOpacity(
      opacity: isVisible ? 1.0 : 0.0,
      duration: const Duration(milliseconds: 280),
      child: IgnorePointer(
        ignoring: !isVisible,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          mainAxisSize: MainAxisSize.min,
          children: [
            _FilterChip(
              label: 'all',
              count: openSeries?.paperCount,
              isActive: ctrl.filterSubject == null,
              color: const Color(0xFFC9AA71),
              onTap: () => ctrl.setFilter(null),
            ),
            ...SubjectGroup.values.map((g) {
              final count = openSeries?.papers
                  .where((p) => p.subjectGroup == g)
                  .length;
              if (count == null || count == 0) return const SizedBox();
              return Padding(
                padding: const EdgeInsets.only(top: 1),
                child: _FilterChip(
                  label: g.label.toLowerCase(),
                  count: count,
                  isActive: ctrl.filterSubject == g,
                  color: g.color,
                  onTap: () =>
                      ctrl.setFilter(ctrl.filterSubject == g ? null : g),
                ),
              );
            }),
          ],
        ),
      ),
    );
  }
}

class _FilterChip extends StatelessWidget {
  final String label;
  final int? count;
  final bool isActive;
  final Color color;
  final VoidCallback onTap;

  const _FilterChip({
    required this.label,
    required this.isActive,
    required this.color,
    required this.onTap,
    this.count,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 3),
        decoration: BoxDecoration(
          color: isActive ? color.withOpacity(0.13) : Colors.transparent,
          borderRadius: BorderRadius.circular(5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: GoogleFonts.inter(
                fontSize: 11,
                fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
                color: isActive ? color : Colors.white.withOpacity(0.30),
              ),
            ),
            if (count != null) ...[
              const SizedBox(width: 3),
              Text(
                '$count',
                style: GoogleFonts.jetBrainsMono(
                  fontSize: 8.5,
                  color: Colors.white.withOpacity(0.20),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  4. File card explosion layer
// ─────────────────────────────────────────────────────────────────────────────

class _FileCardLayer extends StatelessWidget {
  final CatalogController ctrl;
  const _FileCardLayer({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    final papers = ctrl.displayedPapers;
    final folderPos = ctrl.openFolderScreenPos ?? Offset.zero;
    final size = MediaQuery.of(context).size;
    final positions = _computePositions(folderPos, papers.length, size);

    return Stack(
      children: [
        // Dim backdrop — tap to close
        Positioned.fill(
          child: GestureDetector(
            onTap: ctrl.closeFolder,
            child: Container(color: Colors.black.withOpacity(0.72))
                .animate()
                .fadeIn(duration: 280.ms),
          ),
        ),

        // File cards
        for (int i = 0; i < papers.length; i++)
          AnimatedFileCard(
            key: ValueKey(papers[i].id),
            paper: papers[i],
            index: i,
            position: i < positions.length ? positions[i] : folderPos,
            onTap: () => ctrl.selectPaper(papers[i]),
          ),

        // Hint at bottom
        Positioned(
          left: 0, right: 0, bottom: 22,
          child: Center(
            child: IgnorePointer(
              child: Text(
                'tap a paper to view details  ·  tap background to close',
                style: GoogleFonts.jetBrainsMono(
                  fontSize: 9,
                  color: Colors.white.withOpacity(0.20),
                  letterSpacing: 0.8,
                ),
              ).animate(delay: 700.ms).fadeIn(duration: 500.ms),
            ),
          ),
        ),
      ],
    );
  }

  List<Offset> _computePositions(Offset folderCenter, int count, Size screen) {
    if (count == 0) return [];

    const cardW = AnimatedFileCard.kCardW;
    const cardH = AnimatedFileCard.kCardH;
    const gapX  = 11.0;
    const gapY  = 13.0;
    const maxCols = 9;

    final cols = count.clamp(1, maxCols);
    final rows = (count / cols).ceil();

    final totalW = cols * cardW + (cols - 1) * gapX;
    final totalH = rows * cardH + (rows - 1) * gapY;

    // Centre horizontally on screen, sit in the upper region
    double startX = (screen.width  - totalW) / 2;
    double startY = screen.height * 0.09;

    startX = startX.clamp(14.0, screen.width  - totalW - 14.0);
    startY = startY.clamp(18.0, screen.height - totalH - 18.0);

    return List.generate(count, (i) {
      final col = i % cols;
      final row = i ~/ cols;
      return Offset(
        startX + col * (cardW + gapX) + cardW / 2,
        startY + row * (cardH + gapY) + cardH / 2,
      );
    });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  5. Paper detail modal layer
// ─────────────────────────────────────────────────────────────────────────────

class _PaperDetailLayer extends StatelessWidget {
  final CatalogController ctrl;
  const _PaperDetailLayer({required this.ctrl});

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: GestureDetector(
        onTap: ctrl.deselectPaper,
        behavior: HitTestBehavior.opaque,
        child: Container(
          color: Colors.black.withOpacity(0.45),
          child: PaperDetailSheet(paper: ctrl.selectedPaper!),
        ),
      ),
    );
  }
}

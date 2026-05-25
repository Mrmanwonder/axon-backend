import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/paper.dart';

class AnimatedFileCard extends StatelessWidget {
  final Paper paper;
  final int index;
  final Offset position; // center point in screen coordinates
  final VoidCallback onTap;

  static const double kCardW = 168.0;
  static const double kCardH = 95.0;

  const AnimatedFileCard({
    super.key,
    required this.paper,
    required this.index,
    required this.position,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    // Alternating subtle rotation
    final rotAngle = ((index % 3) - 1) * 0.022;

    return Positioned(
      left: position.dx - kCardW / 2,
      top: position.dy - kCardH / 2,
      child: GestureDetector(
        onTap: onTap,
        child: Transform.rotate(
          angle: rotAngle,
          child: _CardContent(paper: paper),
        )
            .animate(
              delay: Duration(milliseconds: 60 + index * 32),
            )
            .fadeIn(
              duration: 280.ms,
              curve: Curves.easeOut,
            )
            .scale(
              begin: const Offset(0.08, 0.08),
              end: const Offset(1.0, 1.0),
              duration: 480.ms,
              curve: Curves.elasticOut,
            )
            .slideY(
              begin: 0.55,
              end: 0.0,
              duration: 380.ms,
              curve: Curves.easeOutCubic,
            ),
      ),
    );
  }
}

// ─── Card content widget ───────────────────────────────────────────────────

class _CardContent extends StatefulWidget {
  final Paper paper;
  const _CardContent({required this.paper});

  @override
  State<_CardContent> createState() => _CardContentState();
}

class _CardContentState extends State<_CardContent> {
  bool _hovered = false;

  @override
  Widget build(BuildContext context) {
    final color = widget.paper.subjectGroup.color;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        width: AnimatedFileCard.kCardW,
        height: AnimatedFileCard.kCardH,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(9),
          color: _hovered
              ? const Color(0xFF252538)
              : const Color(0xFF1F1F2E),
          border: Border.all(
            color: _hovered
                ? color.withOpacity(0.55)
                : color.withOpacity(0.22),
            width: 1.2,
          ),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(_hovered ? 0.30 : 0.15),
              blurRadius: _hovered ? 18 : 10,
              spreadRadius: _hovered ? 1 : 0,
              offset: const Offset(0, 4),
            ),
            const BoxShadow(
              color: Color(0x55000000),
              blurRadius: 10,
              offset: Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Accent strip ──────────────────────────────────────────
            Container(
              height: 3.5,
              decoration: BoxDecoration(
                color: color,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(9),
                ),
              ),
            ),
            // ── Card body ─────────────────────────────────────────────
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(11, 7, 11, 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Subject chip + marks
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.14),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            widget.paper.subjectGroup.shortLabel,
                            style: GoogleFonts.jetBrainsMono(
                              fontSize: 8.5,
                              fontWeight: FontWeight.w700,
                              color: color,
                              letterSpacing: 0.6,
                            ),
                          ),
                        ),
                        const Spacer(),
                        Text(
                          '${widget.paper.totalMarks} mk',
                          style: GoogleFonts.jetBrainsMono(
                            fontSize: 8.5,
                            color: Colors.white.withOpacity(0.35),
                            letterSpacing: 0.3,
                          ),
                        ),
                      ],
                    ),
                    // Paper code — bold mono
                    Text(
                      widget.paper.paperCode,
                      style: GoogleFonts.jetBrainsMono(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: Colors.white.withOpacity(0.92),
                        letterSpacing: 1.1,
                        height: 1,
                      ),
                    ),
                    // Component name
                    Text(
                      widget.paper.component,
                      style: GoogleFonts.inter(
                        fontSize: 9,
                        color: Colors.white.withOpacity(0.42),
                        height: 1.25,
                        letterSpacing: 0.1,
                      ),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

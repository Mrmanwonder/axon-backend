import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../controllers/catalog_controller.dart';
import '../models/paper.dart';

class PaperDetailSheet extends StatelessWidget {
  final Paper paper;

  const PaperDetailSheet({super.key, required this.paper});

  @override
  Widget build(BuildContext context) {
    final color = paper.subjectGroup.color;
    final screenSize = MediaQuery.of(context).size;

    return Center(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: screenSize.width * 0.42,
          maxHeight: screenSize.height * 0.82,
        ),
        child: Container(
          margin: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: const Color(0xFF18182A),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(
              color: color.withOpacity(0.38),
              width: 1.5,
            ),
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.18),
                blurRadius: 40,
                spreadRadius: 6,
              ),
              const BoxShadow(
                color: Color(0xDD000000),
                blurRadius: 50,
                offset: Offset(0, 12),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Header ──────────────────────────────────────────────────
              _Header(paper: paper, color: color),
              // ── Divider ─────────────────────────────────────────────────
              Divider(
                height: 0,
                color: Colors.white.withOpacity(0.06),
              ),
              // ── Details ─────────────────────────────────────────────────
              Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _DetailRow(
                      icon: Icons.tag_rounded,
                      label: 'Paper Code',
                      value: paper.paperCode,
                      color: color,
                    ),
                    const SizedBox(height: 12),
                    _DetailRow(
                      icon: Icons.link_rounded,
                      label: 'Full Reference',
                      value: paper.fullReference,
                      color: color,
                    ),
                    const SizedBox(height: 12),
                    _DetailRow(
                      icon: Icons.calendar_today_rounded,
                      label: 'Series',
                      value: paper.seriesId.toUpperCase(),
                      color: color,
                    ),
                    const SizedBox(height: 12),
                    _DetailRow(
                      icon: Icons.layers_rounded,
                      label: 'Paper / Variant',
                      value:
                          'Paper ${paper.paperNumber}  ·  Variant ${paper.variant}',
                      color: color,
                    ),
                    const SizedBox(height: 12),
                    _DetailRow(
                      icon: Icons.score_rounded,
                      label: 'Total Marks',
                      value: '${paper.totalMarks} marks',
                      color: color,
                    ),
                    const SizedBox(height: 20),
                    // ── Action button ──────────────────────────────────────
                    _ActionButton(color: color),
                  ],
                ),
              ),
            ],
          ),
        )
            .animate()
            .fadeIn(duration: 220.ms, curve: Curves.easeOut)
            .scale(
              begin: const Offset(0.88, 0.88),
              curve: Curves.easeOutCubic,
              duration: 300.ms,
            ),
      ),
    );
  }
}

// ─── Header ───────────────────────────────────────────────────────────────

class _Header extends StatelessWidget {
  final Paper paper;
  final Color color;
  const _Header({required this.paper, required this.color});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 22, 20, 18),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Color accent bar
          Container(
            width: 4,
            height: 46,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(2),
              boxShadow: [
                BoxShadow(
                  color: color.withOpacity(0.4),
                  blurRadius: 10,
                  spreadRadius: 1,
                ),
              ],
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  paper.subject,
                  style: GoogleFonts.inter(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                    letterSpacing: 0.1,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  paper.component,
                  style: GoogleFonts.inter(
                    fontSize: 11.5,
                    color: Colors.white.withOpacity(0.48),
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
          // Close button
          GestureDetector(
            onTap: () =>
                context.read<CatalogController>().deselectPaper(),
            child: Container(
              width: 28,
              height: 28,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.06),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                Icons.close_rounded,
                color: Colors.white.withOpacity(0.45),
                size: 16,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Detail row ────────────────────────────────────────────────────────────

class _DetailRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _DetailRow({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, size: 13, color: color.withOpacity(0.55)),
        const SizedBox(width: 8),
        Text(
          label,
          style: GoogleFonts.jetBrainsMono(
            fontSize: 9.5,
            color: Colors.white.withOpacity(0.32),
            letterSpacing: 0.5,
          ),
        ),
        const Spacer(),
        Text(
          value,
          style: GoogleFonts.jetBrainsMono(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: Colors.white.withOpacity(0.85),
            letterSpacing: 0.4,
          ),
        ),
      ],
    );
  }
}

// ─── Action button ─────────────────────────────────────────────────────────

class _ActionButton extends StatefulWidget {
  final Color color;
  const _ActionButton({required this.color});

  @override
  State<_ActionButton> createState() => _ActionButtonState();
}

class _ActionButtonState extends State<_ActionButton> {
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _pressed = true),
      onTapUp: (_) => setState(() => _pressed = false),
      onTapCancel: () => setState(() => _pressed = false),
      onTap: () {},
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 110),
        width: double.infinity,
        height: 40,
        decoration: BoxDecoration(
          color: _pressed
              ? widget.color.withOpacity(0.22)
              : widget.color.withOpacity(0.10),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: widget.color.withOpacity(_pressed ? 0.55 : 0.3),
            width: 1.2,
          ),
        ),
        child: Center(
          child: Text(
            'View  Paper  →',
            style: GoogleFonts.jetBrainsMono(
              fontSize: 11.5,
              fontWeight: FontWeight.w700,
              color: widget.color,
              letterSpacing: 1.2,
            ),
          ),
        ),
      ),
    );
  }
}

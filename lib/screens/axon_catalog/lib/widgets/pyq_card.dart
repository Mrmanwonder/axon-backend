import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/paper.dart';

class PYQCard extends StatefulWidget {
  final Paper paper;
  final VoidCallback onTap;
  final bool isListView;

  const PYQCard({
    super.key,
    required this.paper,
    required this.onTap,
    this.isListView = false,
  });

  @override
  State<PYQCard> createState() => _PYQCardState();
}

class _PYQCardState extends State<PYQCard> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          decoration: BoxDecoration(
            color: _isHovered 
                ? Colors.white.withOpacity(0.1) 
                : Colors.white.withOpacity(0.05),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: _isHovered 
                  ? widget.paper.subjectGroup.color.withOpacity(0.4) 
                  : Colors.white.withOpacity(0.1),
              width: 1,
            ),
            boxShadow: _isHovered ? [
              BoxShadow(
                color: widget.paper.subjectGroup.color.withOpacity(0.2),
                blurRadius: 12,
                spreadRadius: 2,
              )
            ] : [],
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: widget.isListView ? _buildListLayout() : _buildGridLayout(),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildGridLayout() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: widget.paper.subjectGroup.color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                widget.paper.subjectGroup.shortLabel,
                style: GoogleFonts.inter(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: widget.paper.subjectGroup.color,
                ),
              ),
            ),
            Icon(Icons.bookmark_border, color: Colors.white.withOpacity(0.5), size: 16),
          ],
        ),
        const Spacer(),
        Text(
          widget.paper.fullReference,
          style: GoogleFonts.jetBrainsMono(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          widget.paper.component,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: GoogleFonts.inter(
            fontSize: 12,
            color: Colors.white.withOpacity(0.7),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              '${widget.paper.totalMarks} Marks',
              style: GoogleFonts.inter(
                fontSize: 11,
                color: Colors.white.withOpacity(0.5),
              ),
            ),
            Icon(Icons.arrow_forward_ios, size: 12, color: Colors.white.withOpacity(0.4)),
          ],
        ),
      ],
    );
  }

  Widget _buildListLayout() {
    return Row(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: widget.paper.subjectGroup.color.withOpacity(0.2),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Center(
            child: Text(
              widget.paper.subjectGroup.shortLabel,
              style: GoogleFonts.inter(
                fontSize: 12,
                fontWeight: FontWeight.w800,
                color: widget.paper.subjectGroup.color,
              ),
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                widget.paper.fullReference,
                style: GoogleFonts.jetBrainsMono(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                widget.paper.component,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.inter(
                  fontSize: 13,
                  color: Colors.white.withOpacity(0.7),
                ),
              ),
            ],
          ),
        ),
        Column(
          crossAxisAlignment: CrossAxisAlignment.end,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '${widget.paper.totalMarks} Marks',
              style: GoogleFonts.inter(
                fontSize: 12,
                color: Colors.white.withOpacity(0.5),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Icon(Icons.download, size: 16, color: Colors.white.withOpacity(0.6)),
                const SizedBox(width: 12),
                Icon(Icons.bookmark_border, size: 16, color: Colors.white.withOpacity(0.6)),
              ],
            )
          ],
        )
      ],
    );
  }
}

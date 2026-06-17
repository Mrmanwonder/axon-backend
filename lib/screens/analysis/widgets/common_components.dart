import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme/app_theme.dart';

class Panel extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  final Color? borderColor;

  const Panel({
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(14),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 16, sigmaY: 16),
        child: Container(
          width: double.infinity,
          padding: padding,
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated.withValues(alpha: 0.72),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
              color: borderColor ?? AxonColors.divider.withValues(alpha: 0.45),
              width: 0.5,
            ),
            boxShadow: [
              BoxShadow(
                color: const Color(0x47000000),
                blurRadius: 24,
                offset: const Offset(0, 8),
              ),
              BoxShadow(
                color: (borderColor ?? AxonColors.accent).withValues(alpha: 0.05),
                blurRadius: 40,
                offset: const Offset(0, 16),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );
  }
}

class GlassSheet extends StatelessWidget {
  final Widget child;

  const GlassSheet({required this.child});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(18)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated.withValues(alpha: 0.88),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(18)),
            border: Border(
              top: BorderSide(
                color: AxonColors.divider.withValues(alpha: 0.4),
                width: 0.5,
              ),
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}

class IconTile extends StatelessWidget {
  final IconData icon;
  final Color color;

  const IconTile({
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 38,
      height: 38,
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.18)),
      ),
      child: Icon(icon, color: color, size: 20),
    );
  }
}

class SectionTitle extends StatelessWidget {
  final String title;
  final String? trailing;

  const SectionTitle({
    required this.title,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: Text(
            title,
            style: GoogleFonts.inter(
              color: AxonColors.textPrimary,
              fontSize: 15,
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        if (trailing != null)
          Text(
            trailing!,
            style: GoogleFonts.inter(
              color: AxonColors.textTertiary,
              fontSize: 11,
              fontWeight: FontWeight.w700,
            ),
          ),
      ],
    );
  }
}

class StatusChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;

  const StatusChip({
    required this.icon,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 15),
          const SizedBox(width: 6),
          Text(
            label,
            style: GoogleFonts.inter(
              color: AxonColors.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

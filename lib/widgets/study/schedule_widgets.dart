import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme/app_theme.dart';

class AnchorDateCountdownCard extends StatefulWidget {
  final DateTime targetDate;
  final String examName;
  final String subject;
  final bool showGlow;

  const AnchorDateCountdownCard({
    super.key,
    required this.targetDate,
    required this.examName,
    required this.subject,
    this.showGlow = true,
  });

  @override
  State<AnchorDateCountdownCard> createState() =>
      _AnchorDateCountdownCardState();
}

class _AnchorDateCountdownCardState extends State<AnchorDateCountdownCard> {
  late Duration _remaining;
  bool _isUrgent = false;

  @override
  void initState() {
    super.initState();
    _updateRemaining();
  }

  void _updateRemaining() {
    final now = DateTime.now();
    final remaining = widget.targetDate.difference(now);
    setState(() {
      _remaining = remaining;
      _isUrgent = remaining.inHours < 48;
    });
  }

  @override
  Widget build(BuildContext context) {
    final days = _remaining.inDays;
    final hours = _remaining.inHours.remainder(24);
    final minutes = _remaining.inMinutes.remainder(60);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF121212),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _isUrgent
              ? Colors.redAccent.withValues(alpha: 0.5)
              : Colors.white.withValues(alpha: 0.1),
          width: 1,
        ),
        boxShadow: _isUrgent && widget.showGlow
            ? [
                BoxShadow(
                  color: Colors.redAccent.withValues(alpha: 0.3),
                  blurRadius: 16,
                  spreadRadius: 0,
                ),
              ]
            : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.flag,
                color: _isUrgent ? Colors.redAccent : AxonColors.electricCyan,
                size: 18,
              ),
              const SizedBox(width: 8),
              Text(
                'First Paper',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.5),
                  fontSize: 12,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            widget.examName,
            style: GoogleFonts.inter(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          Text(
            widget.subject,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _TimeUnit(value: days, label: 'DAYS', isUrgent: _isUrgent),
              _TimeUnit(value: hours, label: 'HRS', isUrgent: _isUrgent),
              _TimeUnit(value: minutes, label: 'MIN', isUrgent: _isUrgent),
            ],
          ),
        ],
      ),
    );
  }
}

class _TimeUnit extends StatelessWidget {
  final int value;
  final String label;
  final bool isUrgent;

  const _TimeUnit({
    required this.value,
    required this.label,
    required this.isUrgent,
  });

  @override
  Widget build(BuildContext context) {
    final color = isUrgent ? Colors.redAccent : Colors.white;

    return Column(
      children: [
        Text(
          value.toString().padLeft(2, '0'),
          style: GoogleFonts.jetBrainsMono(
            color: color,
            fontSize: 32,
            fontWeight: FontWeight.w700,
            letterSpacing: 2,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            color: color.withValues(alpha: 0.5),
            fontSize: 10,
            fontWeight: FontWeight.w500,
            letterSpacing: 1,
          ),
        ),
      ],
    );
  }
}

class SmartGapSuggestionChip extends StatelessWidget {
  final String subject;
  final String subjectCode;
  final int durationMinutes;
  final String activityType;
  final VoidCallback? onTap;

  const SmartGapSuggestionChip({
    super.key,
    required this.subject,
    required this.subjectCode,
    required this.durationMinutes,
    required this.activityType,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        HapticFeedback.selectionClick();
        onTap?.call();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.transparent,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.2),
            width: 1,
            style: BorderStyle.solid,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.auto_awesome,
              color: AxonColors.electricCyan,
              size: 16,
            ),
            const SizedBox(width: 8),
            Text(
              'Recommended: ${durationMinutes}m $subject ($subjectCode) $activityType',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

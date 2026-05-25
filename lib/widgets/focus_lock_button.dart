// lib/widgets/focus_lock_button.dart
// Focus Lock Button - triggers Study Lock screen

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';
import 'common/rose_loader.dart';

class FocusLockButton extends StatefulWidget {
  final VoidCallback? onLockEnabled;
  final VoidCallback? onLockDisabled;

  const FocusLockButton({
    super.key,
    this.onLockEnabled,
    this.onLockDisabled,
  });

  @override
  State<FocusLockButton> createState() => _FocusLockButtonState();
}

class _FocusLockButtonState extends State<FocusLockButton> {
  bool _isLoading = false;

  void _openStudyLock() {
    if (_isLoading) return;
    setState(() => _isLoading = true);
    context.push('/study/lock');
    Future.delayed(const Duration(milliseconds: 500), () {
      if (mounted) setState(() => _isLoading = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AxonColors.accent.withValues(alpha: 0.4),
            blurRadius: 20,
            spreadRadius: 2,
          ),
        ],
      ),
      child: ElevatedButton.icon(
        onPressed: _isLoading ? null : _openStudyLock,
        style: ElevatedButton.styleFrom(
          backgroundColor: AxonColors.accent,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 20),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        icon: _isLoading
            ? const RoseLoader(size: 20, color: Colors.white)
            : const Icon(
                Icons.lock_person,
                size: 24,
              ),
        label: Text(
          'IGNITE DEEP FOCUS',
          style: GoogleFonts.googleSans(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            letterSpacing: 1,
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme/app_theme.dart';

class AuthFrame extends StatelessWidget {
  final String eyebrow;
  final String title;
  final String subtitle;
  final Widget child;
  final Widget? footer;
  final Widget? floatingButton;
  final VoidCallback? onBack;

  const AuthFrame({
    super.key,
    required this.eyebrow,
    required this.title,
    required this.subtitle,
    required this.child,
    this.footer,
    this.floatingButton,
    this.onBack,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  AxonColors.oxfordBlueDark,
                  AxonColors.oxfordBlueDark.withValues(alpha: 0.85),
                ],
              ),
            ),
          ),
          SafeArea(
            bottom: false,
            child: SingleChildScrollView(
              physics: const ClampingScrollPhysics(),
              padding: const EdgeInsets.fromLTRB(20, 40, 20, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (onBack != null)
                    GestureDetector(
                      onTap: onBack,
                      child: Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: AxonColors.surfaceElevated,
                          shape: BoxShape.circle,
                          border: AxonDecor.axonBorder,
                        ),
                        child: Icon(
                          Icons.arrow_back_rounded,
                          color: AxonColors.textPrimary,
                          size: 20,
                        ),
                      ),
                    ),
                  if (onBack != null) const SizedBox(height: 20),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: AxonColors.accent.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(
                      eyebrow,
                      style: GoogleFonts.googleSans(
                        color: AxonColors.accent,
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.2,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    title,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 34,
                      fontWeight: FontWeight.w800,
                      height: 1.1,
                      letterSpacing: -1.2,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    subtitle,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textSecondary,
                      fontSize: 14,
                      height: 1.5,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 24),
                  child,
                  if (footer != null) ...[
                    const SizedBox(height: 20),
                    footer!,
                  ],
                ],
              ),
            ),
          ),
          if (floatingButton != null)
            Positioned(
              left: 0,
              right: 0,
              bottom: MediaQuery.of(context).padding.bottom + 16,
              child: Center(child: floatingButton!),
            ),
        ],
      ),
    );
  }
}

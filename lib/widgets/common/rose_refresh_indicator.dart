import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'rose_loader.dart';

class RoseRefreshIndicator extends StatelessWidget {
  final Widget child;
  final Future<void> Function() onRefresh;
  final Color color;

  const RoseRefreshIndicator({
    super.key,
    required this.child,
    required this.onRefresh,
    this.color = Colors.white,
  });

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      color: color,
      backgroundColor: Colors.transparent,
      onRefresh: onRefresh,
      child: child,
    );
  }
}

class RoseLoadingOverlay extends StatelessWidget {
  final Widget? child;
  final bool isLoading;
  final String? message;

  const RoseLoadingOverlay({
    super.key,
    this.child,
    required this.isLoading,
    this.message,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        if (child != null) child!,
        if (isLoading)
          Positioned.fill(
            child: Container(
              color: Colors.black.withValues(alpha: 0.6),
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const RoseLoader(size: 48),
                    if (message != null) ...[
                      const SizedBox(height: 16),
                      Text(
                        message!,
                        style: GoogleFonts.googleSans(
                          color: Colors.white,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}

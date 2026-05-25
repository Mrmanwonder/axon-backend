// lib/services/alert_service.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

enum AlertType { success, error, warning, info }

class AlertService {
  static OverlayEntry? _currentOverlay;

  static void show({
    required BuildContext context,
    required String title,
    required String message,
    AlertType type = AlertType.info,
    Duration duration = const Duration(seconds: 3),
    VoidCallback? onTap,
    String? actionLabel,
    VoidCallback? onAction,
  }) {
    _currentOverlay?.remove();

    final overlay = OverlayEntry(
      builder: (context) => _PillAlertWidget(
        title: title,
        message: message,
        type: type,
        duration: duration,
        onTap: onTap,
        actionLabel: actionLabel,
        onAction: onAction,
        onDismiss: () {
          _currentOverlay?.remove();
          _currentOverlay = null;
        },
      ),
    );

    _currentOverlay = overlay;
    Overlay.of(context).insert(overlay);

    Future.delayed(duration, () {
      if (_currentOverlay?.mounted == true) {
        _currentOverlay?.remove();
        _currentOverlay = null;
      }
    });
  }

  static void showSuccess(BuildContext context, String title, String message) =>
      show(
          context: context,
          title: title,
          message: message,
          type: AlertType.success);

  static void showError(BuildContext context, String title, String message) =>
      show(
          context: context,
          title: title,
          message: message,
          type: AlertType.error);

  static void showWarning(BuildContext context, String title, String message) =>
      show(
          context: context,
          title: title,
          message: message,
          type: AlertType.warning);

  static void showInfo(BuildContext context, String title, String message) =>
      show(
          context: context,
          title: title,
          message: message,
          type: AlertType.info);

  static void dismiss() {
    _currentOverlay?.remove();
    _currentOverlay = null;
  }
}

class _PillAlertWidget extends StatefulWidget {
  final String title;
  final String message;
  final AlertType type;
  final Duration duration;
  final VoidCallback? onTap;
  final String? actionLabel;
  final VoidCallback? onAction;
  final VoidCallback onDismiss;

  const _PillAlertWidget({
    required this.title,
    required this.message,
    required this.type,
    required this.duration,
    this.onTap,
    this.actionLabel,
    this.onAction,
    required this.onDismiss,
  });

  @override
  State<_PillAlertWidget> createState() => _PillAlertWidgetState();
}

class _PillAlertWidgetState extends State<_PillAlertWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 300));
    _fadeAnimation =
        CurvedAnimation(parent: _controller, curve: Curves.easeOut);
    _slideAnimation =
        Tween<Offset>(begin: const Offset(0, -1), end: Offset.zero).animate(
            CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));
    _controller.forward();
    HapticFeedback.lightImpact();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Color get _accentColor {
    switch (widget.type) {
      case AlertType.success:
        return const Color(0xFF10B981);
      case AlertType.error:
        return const Color(0xFFEF4444);
      case AlertType.warning:
        return const Color(0xFFF59E0B);
      case AlertType.info:
        return const Color(0xFF3A86FF);
    }
  }

  IconData get _icon {
    switch (widget.type) {
      case AlertType.success:
        return Icons.check_circle_rounded;
      case AlertType.error:
        return Icons.error_rounded;
      case AlertType.warning:
        return Icons.warning_rounded;
      case AlertType.info:
        return Icons.info_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: MediaQuery.of(context).padding.top + 8,
      left: 16,
      right: 16,
      child: SlideTransition(
        position: _slideAnimation,
        child: FadeTransition(
          opacity: _fadeAnimation,
          child: GestureDetector(
            onTap: () {
              widget.onTap?.call();
              widget.onDismiss();
            },
            child: ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 32, sigmaY: 32),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        Color(0x33FFFFFF),
                        Color(0xD90B0D0E),
                        Color(0xA6000000),
                      ],
                      stops: [0.0, 0.45, 1.0],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.22),
                      width: 1.1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.55),
                        blurRadius: 34,
                        offset: const Offset(0, 18),
                      ),
                      BoxShadow(
                        color: _accentColor.withValues(alpha: 0.18),
                        blurRadius: 22,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                            color: _accentColor.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(12)),
                        child: Icon(_icon, color: _accentColor, size: 22),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(widget.title,
                                style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600)),
                            const SizedBox(height: 2),
                            Text(widget.message,
                                style: TextStyle(
                                    color: Colors.white.withValues(alpha: 0.7),
                                    fontSize: 12),
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis),
                          ],
                        ),
                      ),
                      if (widget.actionLabel != null) ...[
                        const SizedBox(width: 8),
                        GestureDetector(
                          onTap: () {
                            widget.onAction?.call();
                            widget.onDismiss();
                          },
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                                color: _accentColor,
                                borderRadius: BorderRadius.circular(8)),
                            child: Text(widget.actionLabel!,
                                style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600)),
                          ),
                        ),
                      ],
                      const SizedBox(width: 4),
                      GestureDetector(
                          onTap: widget.onDismiss,
                          child: Icon(Icons.close_rounded,
                              color: Colors.white.withValues(alpha: 0.5),
                              size: 20)),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

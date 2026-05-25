import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme/app_theme.dart';

class AxonDialog {
  static Future<T?> show<T>({
    required BuildContext context,
    required String title,
    Widget? content,
    List<Widget>? actions,
    bool dismissible = true,
    Color? backgroundColor,
  }) {
    return showDialog<T>(
      context: context,
      barrierDismissible: dismissible,
      builder: (context) => Dismissible(
        key: UniqueKey(),
        direction:
            dismissible ? DismissDirection.vertical : DismissDirection.none,
        onDismissed: (_) => Navigator.pop(context),
        child: AlertDialog(
          backgroundColor: backgroundColor ?? AxonColors.surfaceElevated,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Text(
            title,
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontWeight: FontWeight.w700,
              fontSize: 18,
            ),
          ),
          content: content,
          actions: actions,
        ),
      ),
    );
  }

  static Future<String?> showTextInput({
    required BuildContext context,
    required String title,
    required String label,
    required String hint,
    String? initialValue,
  }) async {
    final controller = TextEditingController(text: initialValue);
    return showDialog<String>(
      context: context,
      barrierDismissible: true,
      builder: (context) => Dismissible(
        key: UniqueKey(),
        direction: DismissDirection.vertical,
        onDismissed: (_) => Navigator.pop(context),
        child: AlertDialog(
          backgroundColor: AxonColors.surfaceElevated,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Text(
            title,
            style: GoogleFonts.googleSans(color: Colors.white),
          ),
          content: TextField(
            controller: controller,
            decoration: InputDecoration(
              labelText: label,
              hintText: hint,
              hintStyle: const TextStyle(color: Color(0xFF666666)),
              labelStyle: const TextStyle(color: Color(0xFF8B949E)),
              enabledBorder: const UnderlineInputBorder(
                borderSide: BorderSide(color: Color(0xFF3A86FF)),
              ),
              focusedBorder: const UnderlineInputBorder(
                borderSide: BorderSide(color: Color(0xFF3A86FF), width: 2),
              ),
            ),
            style: const TextStyle(color: Colors.white),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text(
                'Cancel',
                style: TextStyle(color: Color(0xFF8B949E)),
              ),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, controller.text.trim()),
              style: FilledButton.styleFrom(
                backgroundColor: const Color(0xFF3A86FF),
              ),
              child: const Text('OK'),
            ),
          ],
        ),
      ),
    );
  }

  static Future<bool> showConfirmation({
    required BuildContext context,
    required String title,
    required String message,
    String confirmText = 'Confirm',
    String cancelText = 'Cancel',
    Color? confirmColor,
    bool isDestructive = false,
  }) async {
    final result = await showDialog<bool>(
      context: context,
      barrierDismissible: true,
      builder: (context) => Dismissible(
        key: UniqueKey(),
        direction: DismissDirection.vertical,
        onDismissed: (_) => Navigator.pop(context, false),
        child: AlertDialog(
          backgroundColor: AxonColors.surfaceElevated,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Text(
            title,
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontWeight: FontWeight.w700,
              fontSize: 18,
            ),
          ),
          content: Text(
            message,
            style: GoogleFonts.googleSans(
              color: const Color(0xFF8B949E),
              fontSize: 14,
              height: 1.5,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: Text(
                cancelText,
                style: GoogleFonts.googleSans(
                  color: const Color(0xFF8B949E),
                ),
              ),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, true),
              style: FilledButton.styleFrom(
                backgroundColor: confirmColor ??
                    (isDestructive ? Colors.red : const Color(0xFF3A86FF)),
              ),
              child: Text(confirmText),
            ),
          ],
        ),
      ),
    );
    return result ?? false;
  }

  static Future<T?> showCustom<T>({
    required BuildContext context,
    required String title,
    required Widget content,
    List<Widget>? actions,
    bool dismissible = true,
    Color? backgroundColor,
  }) {
    return showDialog<T>(
      context: context,
      barrierDismissible: dismissible,
      builder: (context) => Dismissible(
        key: UniqueKey(),
        direction:
            dismissible ? DismissDirection.vertical : DismissDirection.none,
        onDismissed: (_) => Navigator.pop(context),
        child: AlertDialog(
          backgroundColor: backgroundColor ?? AxonColors.surfaceElevated,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Text(
            title,
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontWeight: FontWeight.w700,
              fontSize: 18,
            ),
          ),
          content: content,
          actions: actions,
        ),
      ),
    );
  }
}

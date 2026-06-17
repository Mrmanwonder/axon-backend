// lib/providers/navigation_provider.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'ui_provider.dart';

Future<void> safeNavigate(BuildContext context, String path) async {
  final container = ProviderScope.containerOf(context);
  final lock = container.read(navigationLockProvider);
  if (lock) return;

  container.read(navigationLockProvider.notifier).state = true;
  try {
    context.go(path);
  } finally {
    await Future.delayed(const Duration(milliseconds: 500));
    if (context.mounted) {
      container.read(navigationLockProvider.notifier).state = false;
    }
  }
}

Future<void> safePop(BuildContext context) async {
  final container = ProviderScope.containerOf(context);
  final lock = container.read(navigationLockProvider);
  if (lock) return;

  container.read(navigationLockProvider.notifier).state = true;
  try {
    if (Navigator.canPop(context)) {
      Navigator.of(context).pop();
    } else {
      context.go('/home');
    }
  } finally {
    await Future.delayed(const Duration(milliseconds: 300));
    if (context.mounted) {
      container.read(navigationLockProvider.notifier).state = false;
    }
  }
}

// lib/providers/ui_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

final navbarVisibleProvider = StateProvider<bool>((ref) => true);
final examPlannerFlipProvider = StateProvider<double>((ref) => 0.0);
final navigationLockProvider = StateProvider<bool>((ref) => false);

void ensureNavbarVisible(WidgetRef ref) {
  ref.read(navigationLockProvider.notifier).state = false;
  Future.microtask(() {
    ref.read(navbarVisibleProvider.notifier).state = true;
  });
}

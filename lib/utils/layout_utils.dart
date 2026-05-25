import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

const double kFloatingNavbarHeight = 72;
const double kFloatingNavbarBottomMargin = 16;

double bottomDockClearance(
  BuildContext context, {
  double extra = 20,
}) {
  return kFloatingNavbarHeight +
      kFloatingNavbarBottomMargin +
      MediaQuery.of(context).padding.bottom +
      extra;
}

class PageIntroService {
  static final Set<String> _seenPages = {};
  static bool shouldAnimate(String pageId) {
    if (_seenPages.contains(pageId)) return false;
    _seenPages.add(pageId);
    return true;
  }
}

extension AnimateIfExtension on Widget {
  Widget animateIf(bool condition, List<Animate Function(Animate)> effects) {
    if (!condition) return this;
    Animate anim = animate();
    for (var effect in effects) {
      anim = effect(anim);
    }
    return anim;
  }
}

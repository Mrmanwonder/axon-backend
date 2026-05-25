import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

// ═══════════════════════════════════════════════════════════════════════════
// CORE BRAND COLORS
// ═══════════════════════════════════════════════════════════════════════════
class AxonBrandColors {
  static const Color electricCyan =
      Color(0xFF3A86FF); // Primary actions, CTA buttons, key highlights
  static const Color deepMidnight =
      Color(0xFF003B95); // Dark variant for hover/pressed states
  static const Color softSky =
      Color(0xFFEBF2FF); // Subtle backgrounds for badges
}

class AxonThemeMode {
  static final ValueNotifier<ThemeMode> notifier =
      ValueNotifier(ThemeMode.system);
  static ThemeMode get mode => notifier.value;
  static bool get isDark {
    if (mode == ThemeMode.dark) return true;
    if (mode == ThemeMode.light) return false;
    final brightness =
        WidgetsBinding.instance.platformDispatcher.platformBrightness;
    return brightness == Brightness.dark;
  }
}

class AxonAccentScheme {
  final Color accent;
  final Color accentBlue;
  final Color accentPurple;
  final Color warning;

  const AxonAccentScheme({
    required this.accent,
    required this.accentBlue,
    required this.accentPurple,
    required this.warning,
  });

  static const Color _accent = AxonBrandColors.electricCyan;

  static const calm = AxonAccentScheme(
    accent: _accent,
    accentBlue: Color(0xFF6BA3FF),
    accentPurple: AxonBrandColors.deepMidnight,
    warning: Color(0xFF6B7280),
  );

  static const focused = AxonAccentScheme(
    accent: _accent,
    accentBlue: Color(0xFF6BA3FF),
    accentPurple: AxonBrandColors.deepMidnight,
    warning: Color(0xFF9CA3AF),
  );

  static const urgent = AxonAccentScheme(
    accent: _accent,
    accentBlue: Color(0xFF6BA3FF),
    accentPurple: AxonBrandColors.deepMidnight,
    warning: Color(0xFF374151),
  );

  static AxonAccentScheme forDaysUntilExam(int? days) {
    if (days == null) return focused;
    if (days <= 7) return urgent;
    if (days <= 14) return focused;
    return calm;
  }

  @override
  bool operator ==(Object other) {
    return other is AxonAccentScheme &&
        other.accent == accent &&
        other.accentBlue == accentBlue &&
        other.accentPurple == accentPurple &&
        other.warning == warning;
  }

  @override
  int get hashCode => Object.hash(accent, accentBlue, accentPurple, warning);
}

class AxonAccentPalette {
  static final ValueNotifier<AxonAccentScheme> notifier =
      ValueNotifier(AxonAccentScheme.focused);

  static AxonAccentScheme get scheme => notifier.value;

  static void setScheme(AxonAccentScheme next) {
    if (notifier.value == next) return;
    notifier.value = next;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// SPATIAL DESIGN SYSTEM - Rule of 8s
// All spacing must be a multiple of 8 to create mathematical rhythm
// ═══════════════════════════════════════════════════════════════════════════
class AppSpacing {
  static const double xs = 8.0;
  static const double s = 16.0;
  static const double m = 24.0;
  static const double l = 32.0;
  static const double xl = 40.0;
  static const double xxl = 48.0;

  // Semantic aliases for common patterns
  static const double cardPadding = s;
  static const double screenPadding = s;
  static const double sectionGap = m;
  static const double itemGap = s;
  static const double iconTextGap = s;
}

// Legacy alias for backward compatibility
class AxonSpacing {
  static const double xs = 4;
  static const double sm = 8;
  static const double md = 16;
  static const double lg = 24;
  static const double xl = 32;
  static const double xxl = 48;
}

// Design Tokens - Border Radius
class AxonRadius {
  static const double sm = 8;
  static const double md = 12;
  static const double lg = 16;
  static const double xl = 24;
  static const double full = 100;
}

// ═══════════════════════════════════════════════════════════════════════════
// SPATIAL DESIGN SYSTEM - Premium Aesthetics
// Deep Charcoal background with Glow System for floating 3D cards
// ═══════════════════════════════════════════════════════════════════════════

class SpatialColors {
  static bool get _dark => AxonThemeMode.isDark;

  // Background - switches with theme (Onyx / Gallery profiles)
  static Color get background =>
      _dark ? const Color(0xFF050505) : const Color(0xFFF8FAFC);
  static Color get charcoal =>
      _dark ? const Color(0xFF050505) : const Color(0xFFF8FAFC);
  static Color get charcoalLight =>
      _dark ? const Color(0xFF121214) : const Color(0xFFE2E8F0);
  static Color get charcoalElevated =>
      _dark ? const Color(0xFF1C1C1E) : const Color(0xFFFFFFFF);

  // Glass effect colors - dark physical glass, high blur, visible rim.
  static Color get glassWhite =>
      _dark ? const Color(0xA60B0D0E) : const Color(0xCFFFFFFF);
  static Color get glassBorder =>
      _dark ? const Color(0x33FFFFFF) : const Color(0x33FFFFFF);
  static Color get glassHighlight =>
      _dark ? const Color(0x24FFFFFF) : const Color(0x66FFFFFF);
  static Color get glassLowlight =>
      _dark ? const Color(0x66000000) : const Color(0x1A000000);

  // Glow intensity levels - Premium shadow (no pure black)
  static Color get glowSubtle =>
      _dark ? const Color(0x0F0F172A) : const Color(0x0A000000);
  static Color get glowMedium =>
      _dark ? const Color(0x140F172A) : const Color(0x14000000);
  static Color get glowStrong =>
      _dark ? const Color(0x1F0F172A) : const Color(0x1F000000);

  // Primary accent glow (Electric Cyan) - Subtle gradient feel
  static Color get accentGlow =>
      AxonBrandColors.electricCyan.withValues(alpha: 0.25);
  static Color get accentGlowStrong =>
      AxonBrandColors.electricCyan.withValues(alpha: 0.4);

  // Warning/Alert glow - Polished colors
  static Color get warningGlow =>
      const Color(0xFFF59E0B).withValues(alpha: 0.25);

  // Success glow
  static Color get successGlow =>
      const Color(0xFF10B981).withValues(alpha: 0.25);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPATIAL DESIGN SYSTEM - Glow BoxShadow Factory
// High blur + low opacity = floating 3D effect
// ═══════════════════════════════════════════════════════════════════════════

class SpatialGlow {
  // Subtle glow for cards (lowest elevation)
  static List<BoxShadow> subtle(Color color) => [
        BoxShadow(
          color: color.withValues(alpha: 0.08),
          blurRadius: 16,
          offset: const Offset(0, 4),
        ),
      ];

  // Medium glow for active/focused cards
  static List<BoxShadow> medium(Color color) => [
        BoxShadow(
          color: color.withValues(alpha: 0.12),
          blurRadius: 24,
          offset: const Offset(0, 8),
        ),
        BoxShadow(
          color: color.withValues(alpha: 0.06),
          blurRadius: 40,
          offset: const Offset(0, 16),
        ),
      ];

  // Strong glow for premium/featured cards (floating effect)
  static List<BoxShadow> strong(Color color) => [
        BoxShadow(
          color: color.withValues(alpha: 0.18),
          blurRadius: 30,
          offset: const Offset(0, 10),
        ),
        BoxShadow(
          color: color.withValues(alpha: 0.10),
          blurRadius: 50,
          offset: const Offset(0, 20),
        ),
      ];

  // Extra strong for hero/premium elements
  static List<BoxShadow> hero(Color color) => [
        BoxShadow(
          color: color.withValues(alpha: 0.22),
          blurRadius: 40,
          offset: const Offset(0, 14),
        ),
        BoxShadow(
          color: color.withValues(alpha: 0.14),
          blurRadius: 60,
          offset: const Offset(0, 28),
        ),
      ];

  // Glass morphism navbar shadow
  static List<BoxShadow> get glassDock => [
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.black.withValues(alpha: 0.55)
              : Colors.black.withValues(alpha: 0.18),
          blurRadius: 34,
          offset: const Offset(0, 18),
        ),
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.white.withValues(alpha: 0.05)
              : Colors.white.withValues(alpha: 0.45),
          blurRadius: 1,
          offset: const Offset(0, -1),
        ),
      ];
}

// ═══════════════════════════════════════════════════════════════════════════
// CONTENT-FIRST CONTAINER HELPERS
// Use instead of fixed heights for responsive layouts
// ═══════════════════════════════════════════════════════════════════════════

class SpatialContainer {
  // Expands to fit content with optional constraints
  static Widget wrap({required Widget child, double? maxWidth}) {
    return ConstrainedBox(
      constraints: BoxConstraints(
        maxWidth: maxWidth ?? double.infinity,
      ),
      child: IntrinsicHeight(child: child),
    );
  }

  // Card container with glow effect
  static Widget card({
    required Widget child,
    Color? glowColor,
    double glowIntensity = 1.0,
    EdgeInsets padding = const EdgeInsets.all(AppSpacing.s),
    double borderRadius = 24,
  }) {
    final color = glowColor ?? Colors.white;
    final shadows = glowIntensity <= 0.5
        ? SpatialGlow.subtle(color)
        : glowIntensity <= 1.0
            ? SpatialGlow.medium(color)
            : SpatialGlow.strong(color);

    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: SpatialColors.charcoalLight,
        borderRadius: BorderRadius.circular(borderRadius),
        border: Border.all(
          color: SpatialColors.glassBorder,
          width: 0.5,
        ),
        boxShadow: shadows,
      ),
      child: child,
    );
  }

  // Glass card with backdrop blur
  static Widget glassCard({
    required Widget child,
    EdgeInsets padding = const EdgeInsets.all(AppSpacing.s),
    double borderRadius = 24,
    double blurSigma = 30,
  }) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(borderRadius),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: blurSigma, sigmaY: blurSigma),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: SpatialColors.charcoalLight.withValues(alpha: 0.86),
            borderRadius: BorderRadius.circular(borderRadius),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.08),
              width: 1.1,
            ),
            boxShadow: SpatialGlow.glassDock,
          ),
          child: child,
        ),
      ),
    );
  }
}

// Design Tokens - Animation Durations
class AxonDuration {
  static const Duration fast = Duration(milliseconds: 150);
  static const Duration normal = Duration(milliseconds: 300);
  static const Duration slow = Duration(milliseconds: 500);
}

// Premium Light Theme - Gallery Profile
class _AxonLight {
  static const oxfordBlue = Color(0xFFF8FAFC); // Clean, airy base
  static const oxfordBlueDark = Color(0xFFF1F5F9); // Slightly darker background
  static const oxfordBlueLight = Color(0xFFE2E8F0); // Surface border
  static const electricCyan = AxonBrandColors.electricCyan;
  static const electricCyanGlow = Color(0x203A86FF);
  static const electricCyanDim = Color(0xFF2563EB);
  static const slateGray = Color(0xFF475569); // Professional gray
  static const slateGrayLight = Color(0xFF94A3B8); // Muted gray
  static const poor = Color(0xFFE11D48); // Rose-tinted red
  static const poorGlow = Color(0x20E11D48);
  static const belowAverage = Color(0xFFF59E0B);
  static const average = Color(0xFFFBBF24);
  static const good = Color(0xFF10B981);
  static const excellent = AxonBrandColors.electricCyan;
  static const excellentGlow = Color(0x203A86FF);
  static const surface = Color(0xFFFFFFFF); // Clean white cards
  static const surfaceElevated = Color(0xFFF8FAFC); // Elevated cards
  static const surfaceHighlight = Color(0xFFF1F5F9); // Active states
  static const textPrimary = Color(0xFF0F172A); // Deep navy-black
  static const textSecondary = Color(0xFF475569); // Professional gray
  static const textTertiary = Color(0xFF94A3B8); // Hints, placeholders
  static const divider = Color(0xFFE2E8F0); // Ultra-thin borders
}

// Premium Dark Theme - Onyx Profile
class _AxonDark {
  static const oxfordBlue = Color(0xFF050505); // Nearly black, retains detail
  static const oxfordBlueDark = Color(0xFF050505); // Background
  static const oxfordBlueLight = Color(0xFF121214); // Primary card layer
  static const electricCyan = AxonBrandColors.electricCyan;
  static const electricCyanGlow = Color(0x333A86FF);
  static const electricCyanDim = Color(0xFF2563EB);
  static const slateGray = Color(0xFF94A3B8); // Muted slate for body text
  static const slateGrayLight = Color(0xFF64748B); // Tertiary text
  static const poor = Color(0xFFE11D48); // Rose-tinted red
  static const poorGlow = Color(0x20E11D48);
  static const belowAverage = Color(0xFFF59E0B); // Amber
  static const average = Color(0xFFFBBF24); // Yellow
  static const good = Color(0xFF10B981); // Emerald green
  static const excellent = AxonBrandColors.electricCyan;
  static const excellentGlow = Color(0x203A86FF);
  static const surface = Color(0xFF050505); // Background
  static const surfaceElevated = Color(0xFF121214); // Primary cards
  static const surfaceHighlight =
      Color(0xFF1C1C1E); // Modals, floating elements
  static const textPrimary = Color(0xFFFFFFFF); // Crisp readability
  static const textSecondary =
      Color(0xFF94A3B8); // Muted slate (reduces eye strain)
  static const textTertiary = Color(0xFF64748B);
  static const divider = Color(0xFF1E1E22); // Subtle divider
}

class AxonColors {
  static bool get _dark => AxonThemeMode.isDark;
  static Color get background =>
      _dark ? _AxonDark.oxfordBlue : _AxonLight.oxfordBlue;
  static const Color vibrantBlue = AxonBrandColors.electricCyan;

  static Color get oxfordBlue =>
      _dark ? _AxonDark.oxfordBlue : _AxonLight.oxfordBlue;
  static Color get oxfordBlueDark =>
      _dark ? _AxonDark.oxfordBlueDark : _AxonLight.oxfordBlueDark;
  static Color get oxfordBlueLight =>
      _dark ? _AxonDark.oxfordBlueLight : _AxonLight.oxfordBlueLight;
  static Color get electricCyan =>
      _dark ? _AxonDark.electricCyan : _AxonLight.electricCyan;
  static Color get electricCyanGlow =>
      _dark ? _AxonDark.electricCyanGlow : _AxonLight.electricCyanGlow;
  static Color get electricCyanDim =>
      _dark ? _AxonDark.electricCyanDim : _AxonLight.electricCyanDim;
  static Color get slateGray =>
      _dark ? _AxonDark.slateGray : _AxonLight.slateGray;
  static Color get slateGrayLight =>
      _dark ? _AxonDark.slateGrayLight : _AxonLight.slateGrayLight;

  static Color get poor => _dark ? _AxonDark.poor : _AxonLight.poor;
  static Color get poorGlow => _dark ? _AxonDark.poorGlow : _AxonLight.poorGlow;
  static Color get belowAverage =>
      _dark ? _AxonDark.belowAverage : _AxonLight.belowAverage;
  static Color get average => _dark ? _AxonDark.average : _AxonLight.average;
  static Color get good => _dark ? _AxonDark.good : _AxonLight.good;
  static Color get excellent =>
      _dark ? _AxonDark.excellent : _AxonLight.excellent;
  static Color get excellentGlow =>
      _dark ? _AxonDark.excellentGlow : _AxonLight.excellentGlow;

  static Color get surface => _dark ? _AxonDark.surface : _AxonLight.surface;
  static Color get surfaceSecondary =>
      _dark ? _AxonDark.surfaceElevated : _AxonLight.surfaceElevated;
  static Color get surfaceElevated =>
      _dark ? _AxonDark.surfaceElevated : _AxonLight.surfaceElevated;
  static Color get surfaceHighlight =>
      _dark ? _AxonDark.surfaceHighlight : _AxonLight.surfaceHighlight;
  static Color get textPrimary =>
      _dark ? _AxonDark.textPrimary : _AxonLight.textPrimary;
  static Color get textSecondary =>
      _dark ? _AxonDark.textSecondary : _AxonLight.textSecondary;
  static Color get textTertiary =>
      _dark ? _AxonDark.textTertiary : _AxonLight.textTertiary;
  static Color get divider => _dark ? _AxonDark.divider : _AxonLight.divider;

  // Solid surface colors
  static Color get cardSurface =>
      _dark ? _AxonDark.surfaceElevated : _AxonLight.surfaceElevated;

  // Legacy aliases
  static Color get backgroundPrimary => background;
  static Color get accentPrimary => accent;

  // Enhanced vibrant accent colors
  static Color get accent =>
      _dark ? vibrantBlue : AxonAccentPalette.scheme.accent;
  static Color get accentBlue => AxonAccentPalette.scheme.accentBlue;
  static Color get accentPurple => AxonAccentPalette.scheme.accentPurple;
  static const Color accentPink = Color(0xFFF43F5E);
  static Color get warning => AxonAccentPalette.scheme.warning;

  // Polished Functional Colors
  static const Color error = Color(0xFFE11D48); // Rose-tinted red (modern)
  static const Color success =
      Color(0xFF10B981); // Emerald green (sophisticated)
  static const Color info = AxonBrandColors.electricCyan; // Brand primary

  // Glow colors
  static Color get accentGlow => accent.withValues(alpha: 0.25);
  static Color get warningGlow => warning.withValues(alpha: 0.25);
  static Color get errorGlow => error.withValues(alpha: 0.25);

  static Color performanceColor(double score) {
    if (score < 0.3) return error;
    if (score < 0.5) return belowAverage;
    if (score < 0.7) return average;
    if (score < 0.85) return good;
    return accent;
  }

  static Color performanceGlow(double score) {
    return performanceColor(score).withValues(alpha: 0.2);
  }

  static String performanceLabel(double score) {
    if (score < 0.3) return 'Critical';
    if (score < 0.5) return 'Sub-optimal';
    if (score < 0.7) return 'Stable';
    if (score < 0.85) return 'Optimal';
    return 'Peak Performance';
  }
}

class AxonTheme {
  static ThemeData get light {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: _AxonLight.oxfordBlueDark,
      colorScheme: ColorScheme.light(
        primary: AxonColors.accent,
        onPrimary: Colors.white,
        secondary: _AxonLight.poor,
        surface: _AxonLight.surface,
        onSurface: _AxonLight.textPrimary,
        error: _AxonLight.poor,
        outline: AxonColors.divider,
      ),
      textTheme: _buildTextTheme(),
      appBarTheme: AppBarTheme(
        backgroundColor: _AxonLight.oxfordBlueDark,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.inter(
          color: AxonColors.textPrimary,
          fontSize: 20,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
        ),
        iconTheme: IconThemeData(color: AxonColors.textPrimary),
      ),
      cardTheme: CardThemeData(
        color: _AxonLight.surfaceElevated,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: AxonColors.divider, width: 1),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _AxonLight.surface,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: AxonColors.divider),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: AxonColors.divider),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(8),
          borderSide: BorderSide(color: AxonColors.accent, width: 1.5),
        ),
        labelStyle: GoogleFonts.inter(color: AxonColors.textSecondary),
        hintStyle: GoogleFonts.inter(color: AxonColors.textTertiary),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AxonColors.accent,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w700,
            fontSize: 16,
            letterSpacing: 0,
          ),
          minimumSize: const Size(0, 48),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AxonColors.textPrimary,
          side: BorderSide(color: AxonColors.divider, width: 1),
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w600,
            fontSize: 16,
            letterSpacing: 0,
          ),
          minimumSize: const Size(0, 48),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 16),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w700,
            fontSize: 16,
            letterSpacing: 0,
          ),
          minimumSize: const Size(0, 48),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w600,
            fontSize: 16,
            letterSpacing: 0,
          ),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: AxonColors.divider,
        thickness: 1,
      ),
      canvasColor: _AxonLight.oxfordBlueDark,
      dialogTheme: DialogThemeData(
        backgroundColor: _AxonLight.surfaceElevated,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: _AxonLight.divider),
        ),
      ),
      bottomSheetTheme: const BottomSheetThemeData(
        backgroundColor: _AxonLight.surfaceElevated,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        ),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: _AxonLight.surface,
        selectedItemColor: AxonColors.accent,
        unselectedItemColor: AxonColors.textTertiary,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      sliderTheme: SliderThemeData(
        thumbColor: AxonColors.accent,
        activeTrackColor: AxonColors.accent,
        inactiveTrackColor: AxonColors.divider,
        trackHeight: 4,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 12),
        overlayShape: RoundSliderOverlayShape(overlayRadius: 20),
        tickMarkShape: const RoundSliderTickMarkShape(tickMarkRadius: 2),
      ),
    );
  }

  static TextTheme _buildTextTheme() {
    return TextTheme(
      displayLarge: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 32,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
        height: 1.1,
      ),
      displayMedium: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 28,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
        height: 1.15,
      ),
      displaySmall: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 24,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.2,
        height: 1.2,
      ),
      headlineLarge: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 32,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
        height: 1.25,
      ),
      headlineMedium: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 28,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
        height: 1.3,
      ),
      headlineSmall: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 24,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.2,
        height: 1.35,
      ),
      titleLarge: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 20,
        fontWeight: FontWeight.w700,
        letterSpacing: 0,
      ),
      titleMedium: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 16,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.1,
      ),
      titleSmall: GoogleFonts.inter(
        color: AxonColors.textSecondary,
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.1,
      ),
      bodyLarge: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 16,
        fontWeight: FontWeight.w400,
        height: 1.5,
        letterSpacing: 0,
      ),
      bodyMedium: GoogleFonts.inter(
        color: AxonColors.textSecondary,
        fontSize: 14,
        fontWeight: FontWeight.w400,
        height: 1.5,
        letterSpacing: 0.1,
      ),
      bodySmall: GoogleFonts.inter(
        color: AxonColors.textTertiary,
        fontSize: 12,
        fontWeight: FontWeight.w400,
        height: 1.4,
      ),
      labelLarge: GoogleFonts.inter(
        color: AxonColors.textPrimary,
        fontSize: 14,
        fontWeight: FontWeight.w700,
        letterSpacing: 0.2,
      ),
      labelMedium: GoogleFonts.inter(
        color: AxonColors.textTertiary,
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.4,
      ),
      labelSmall: GoogleFonts.inter(
        color: AxonColors.textTertiary,
        fontSize: 11,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.3,
      ),
    );
  }

  static ThemeData get dark {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AxonColors.background,
      colorScheme: ColorScheme.dark(
        primary: AxonColors.electricCyan,
        onPrimary: Colors.black,
        secondary: _AxonDark.poor,
        surface: _AxonDark.surface,
        onSurface: _AxonDark.textPrimary,
        error: _AxonDark.poor,
        outline: AxonColors.divider,
      ),
      textTheme: _buildTextTheme(),
      appBarTheme: AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.inter(
          color: _AxonDark.textPrimary,
          fontSize: 22,
          fontWeight: FontWeight.w600,
          letterSpacing: -0.7,
        ),
        iconTheme: IconThemeData(color: _AxonDark.textPrimary),
      ),
      cardTheme: CardThemeData(
        color: AxonColors.surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: AxonColors.divider, width: 0.5),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _AxonDark.surface,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(20),
          borderSide: BorderSide(color: _AxonDark.divider),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(20),
          borderSide: BorderSide(color: _AxonDark.divider),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(20),
          borderSide: BorderSide(color: AxonColors.accent, width: 1.5),
        ),
        labelStyle: GoogleFonts.inter(color: _AxonDark.textSecondary),
        hintStyle: GoogleFonts.inter(color: _AxonDark.textTertiary),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AxonColors.accent,
          foregroundColor: Colors.white,
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w700,
            fontSize: 16,
            letterSpacing: -0.1,
            height: 1.0,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _AxonDark.textPrimary,
          side: BorderSide(color: _AxonDark.divider, width: 1.5),
          padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          textStyle: GoogleFonts.inter(
            fontWeight: FontWeight.w600,
            fontSize: 16,
            letterSpacing: -0.1,
            height: 1.0,
          ),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: _AxonDark.divider,
        thickness: 1,
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: _AxonDark.surfaceElevated,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(24),
          side: BorderSide(color: _AxonDark.divider),
        ),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: _AxonDark.surface,
        selectedItemColor: AxonColors.accent,
        unselectedItemColor: _AxonDark.textTertiary,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      sliderTheme: SliderThemeData(
        thumbColor: AxonColors.accent,
        activeTrackColor: AxonColors.accent,
        inactiveTrackColor: AxonColors.divider,
        trackHeight: 4,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 12),
        overlayShape: RoundSliderOverlayShape(overlayRadius: 20),
        tickMarkShape: const RoundSliderTickMarkShape(tickMarkRadius: 2),
      ),
    );
  }
}

class AxonDecor {
  static List<BoxShadow> get axonShadow => [
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.black.withValues(alpha: 0.3)
              : Colors.black.withValues(alpha: 0.04),
          offset: const Offset(0, 2),
          blurRadius: 6,
          spreadRadius: -2,
        ),
      ];

  static List<BoxShadow> get subtleShadow => [
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.black.withValues(alpha: 0.4)
              : Colors.black.withValues(alpha: 0.03),
          offset: const Offset(0, 1),
          blurRadius: 4,
        ),
      ];

  static List<BoxShadow> glowShadow(Color color) => [
        BoxShadow(
          color: color.withValues(alpha: 0.28),
          blurRadius: 20,
          offset: const Offset(0, 0),
          spreadRadius: 2,
        ),
      ];

  static Border get axonBorder => Border.all(
        color: AxonColors.divider,
        width: 1.2,
      );
}

// Gradient utilities
class AxonGradients {
  static LinearGradient get backgroundGradient =>
      backgroundGradientFor(AxonThemeMode.isDark);

  static LinearGradient backgroundGradientFor(bool isDark) => LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: isDark
            ? [
                const Color(0xFF050505),
                const Color(0xFF050505),
                const Color(0xFF121214),
              ]
            : [
                const Color(0xFFF8FAFC),
                const Color(0xFFF1F5F9),
                const Color(0xFFE2E8F0),
              ],
        stops: const [0.0, 0.5, 1.0],
      );

  // Subtle gradient from #3A86FF to #2563EB (adds "weight" to buttons)
  static LinearGradient get cyanGradient => LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          AxonBrandColors.electricCyan,
          const Color(0xFF2563EB),
        ],
      );

  static LinearGradient get accentGradient => LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          AxonBrandColors.electricCyan,
          const Color(0xFF2563EB),
          AxonBrandColors.deepMidnight,
        ],
      );

  // Polished functional gradients
  static LinearGradient get successGradient => LinearGradient(
        colors: [const Color(0xFF10B981), const Color(0xFF059669)],
      );

  static LinearGradient get errorGradient => LinearGradient(
        colors: [const Color(0xFFE11D48), const Color(0xFFBE123C)],
      );

  static LinearGradient get warningGradient => LinearGradient(
        colors: [const Color(0xFFF59E0B), const Color(0xFFB45309)],
      );

  static LinearGradient get excellentGradient => LinearGradient(
        colors: [AxonBrandColors.electricCyan, const Color(0xFF2563EB)],
      );

  static LinearGradient get purpleGradient => LinearGradient(
        colors: [AxonBrandColors.electricCyan, AxonBrandColors.deepMidnight],
      );

  static LinearGradient performanceGradient(double score) {
    if (score < 0.3) return errorGradient;
    if (score < 0.5) return warningGradient;
    if (score < 0.8) return excellentGradient;
    return accentGradient;
  }

  static LinearGradient get poorGradient => errorGradient;

  static LinearGradient get surfaceGradient => LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [
          AxonColors.surface,
          AxonColors.surfaceElevated,
        ],
      );
}

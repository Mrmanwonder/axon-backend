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

  // Neon fluorescent brand colors for Obsidian-Neon (dark mode)
  static const Color neonCyan = Color(0xFF00F0FF);
  static const Color neonPurple = Color(0xFFBD00FF);
  static const Color neonPink = Color(0xFFFF007F);
  static const Color neonGreen = Color(0xFF39FF14);
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
  final Color _accent;
  final Color _accentBlue;
  final Color _accentPurple;
  final Color _warning;

  final Color? _darkAccent;
  final Color? _darkAccentBlue;
  final Color? _darkAccentPurple;
  final Color? _darkWarning;

  const AxonAccentScheme({
    required Color accent,
    required Color accentBlue,
    required Color accentPurple,
    required Color warning,
    Color? darkAccent,
    Color? darkAccentBlue,
    Color? darkAccentPurple,
    Color? darkWarning,
  })  : _accent = accent,
        _accentBlue = accentBlue,
        _accentPurple = accentPurple,
        _warning = warning,
        _darkAccent = darkAccent,
        _darkAccentBlue = darkAccentBlue,
        _darkAccentPurple = darkAccentPurple,
        _darkWarning = darkWarning;

  Color get accent => AxonThemeMode.isDark ? (_darkAccent ?? _accent) : _accent;
  Color get accentBlue =>
      AxonThemeMode.isDark ? (_darkAccentBlue ?? _accentBlue) : _accentBlue;
  Color get accentPurple =>
      AxonThemeMode.isDark ? (_darkAccentPurple ?? _accentPurple) : _accentPurple;
  Color get warning =>
      AxonThemeMode.isDark ? (_darkWarning ?? _warning) : _warning;

  static const calm = AxonAccentScheme(
    accent: AxonBrandColors.electricCyan,
    accentBlue: Color(0xFF6BA3FF),
    accentPurple: AxonBrandColors.deepMidnight,
    warning: Color(0xFF6B7280),
    darkAccent: AxonBrandColors.neonGreen,
    darkAccentBlue: AxonBrandColors.neonCyan,
    darkAccentPurple: AxonBrandColors.neonPurple,
    darkWarning: Color(0xFF6B7280),
  );

  static const focused = AxonAccentScheme(
    accent: AxonBrandColors.electricCyan,
    accentBlue: Color(0xFF6BA3FF),
    accentPurple: AxonBrandColors.deepMidnight,
    warning: Color(0xFF9CA3AF),
    darkAccent: AxonBrandColors.neonCyan,
    darkAccentBlue: AxonBrandColors.neonPurple,
    darkAccentPurple: AxonBrandColors.neonPink,
    darkWarning: Color(0xFF9CA3AF),
  );

  static const urgent = AxonAccentScheme(
    accent: AxonBrandColors.electricCyan,
    accentBlue: Color(0xFF6BA3FF),
    accentPurple: AxonBrandColors.deepMidnight,
    warning: Color(0xFF374151),
    darkAccent: AxonBrandColors.neonPink,
    darkAccentBlue: AxonBrandColors.neonPurple,
    darkAccentPurple: AxonBrandColors.neonCyan,
    darkWarning: Color(0xFF374151),
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
      _dark ? const Color(0xFF020204) : const Color(0xFFEEF2FF);
  static Color get charcoal =>
      _dark ? const Color(0xFF020204) : const Color(0xFFEEF2FF);
  static Color get charcoalLight =>
      _dark ? const Color(0xFF0A0A10) : const Color(0x59FFFFFF); // 35% white in light mode
  static Color get charcoalElevated =>
      _dark ? const Color(0xFF161622) : const Color(0x80FFFFFF); // 50% white in light mode

  // Glass effect colors - dark physical glass, high blur, visible rim.
  static Color get glassWhite =>
      _dark ? const Color(0xA60B0D0E) : const Color(0x59FFFFFF); // 35% white background for glass in light mode
  static Color get glassBorder =>
      _dark ? const Color(0x33FFFFFF) : const Color(0x9CFFFFFF); // 61% white border in light mode
  static Color get glassHighlight =>
      _dark ? const Color(0x24FFFFFF) : const Color(0xA6FFFFFF); // 65% white highlight
  static Color get glassLowlight =>
      _dark ? const Color(0x66000000) : const Color(0x0F000000);

  // Glow intensity levels - Premium shadow (no pure black)
  static Color get glowSubtle =>
      _dark ? const Color(0x0F0F172A) : const Color(0x0D3A86FF); // 5% blue glow in light mode
  static Color get glowMedium =>
      _dark ? const Color(0x140F172A) : const Color(0x143A86FF); // 8% blue glow in light mode
  static Color get glowStrong =>
      _dark ? const Color(0x1F0F172A) : const Color(0x1F3A86FF); // 12% blue glow in light mode

  // Primary accent glow - Theme aware
  static Color get accentGlow =>
      (_dark ? AxonBrandColors.neonCyan : AxonBrandColors.electricCyan).withOpacity(0.25);
  static Color get accentGlowStrong =>
      (_dark ? AxonBrandColors.neonCyan : AxonBrandColors.electricCyan).withOpacity(0.4);

  // Warning/Alert glow - Polished colors
  static Color get warningGlow =>
      (_dark ? AxonBrandColors.neonPurple : const Color(0xFFF59E0B)).withOpacity(0.25);

  // Success glow
  static Color get successGlow =>
      (_dark ? AxonBrandColors.neonGreen : const Color(0xFF10B981)).withOpacity(0.25);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPATIAL DESIGN SYSTEM - Glow BoxShadow Factory
// High blur + low opacity = floating 3D effect
// ═══════════════════════════════════════════════════════════════════════════

class SpatialGlow {
  // Subtle glow for cards (lowest elevation)
  static List<BoxShadow> subtle(Color color) {
    if (AxonThemeMode.isDark) {
      return [
        BoxShadow(
          color: color.withOpacity(0.2),
          blurRadius: 4,
          offset: const Offset(0, 1),
        ),
        BoxShadow(
          color: color.withOpacity(0.08),
          blurRadius: 16,
          offset: const Offset(0, 4),
        ),
      ];
    } else {
      return [
        BoxShadow(
          color: color.withOpacity(0.05),
          blurRadius: 16,
          offset: const Offset(0, 4),
        ),
      ];
    }
  }

  // Medium glow for active/focused cards
  static List<BoxShadow> medium(Color color) {
    if (AxonThemeMode.isDark) {
      return [
        BoxShadow(
          color: color.withOpacity(0.25),
          blurRadius: 8,
          offset: const Offset(0, 2),
        ),
        BoxShadow(
          color: color.withOpacity(0.12),
          blurRadius: 32,
          offset: const Offset(0, 8),
        ),
      ];
    } else {
      return [
        BoxShadow(
          color: color.withOpacity(0.08),
          blurRadius: 24,
          offset: const Offset(0, 8),
        ),
      ];
    }
  }

  // Strong glow for premium/featured cards (floating effect)
  static List<BoxShadow> strong(Color color) {
    if (AxonThemeMode.isDark) {
      return [
        BoxShadow(
          color: color.withOpacity(0.35),
          blurRadius: 12,
          offset: const Offset(0, 3),
        ),
        BoxShadow(
          color: color.withOpacity(0.18),
          blurRadius: 48,
          offset: const Offset(0, 12),
        ),
      ];
    } else {
      return [
        BoxShadow(
          color: color.withOpacity(0.12),
          blurRadius: 32,
          offset: const Offset(0, 12),
        ),
      ];
    }
  }

  // Extra strong for hero/premium elements
  static List<BoxShadow> hero(Color color) {
    if (AxonThemeMode.isDark) {
      return [
        BoxShadow(
          color: color.withOpacity(0.4),
          blurRadius: 16,
          offset: const Offset(0, 4),
        ),
        BoxShadow(
          color: color.withOpacity(0.25),
          blurRadius: 64,
          offset: const Offset(0, 16),
        ),
      ];
    } else {
      return [
        BoxShadow(
          color: color.withOpacity(0.16),
          blurRadius: 40,
          offset: const Offset(0, 16),
        ),
      ];
    }
  }

  // Glass morphism navbar shadow
  static List<BoxShadow> get glassDock => [
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.black.withOpacity(0.55)
              : const Color(0xFF3A86FF).withOpacity(0.08),
          blurRadius: 34,
          offset: const Offset(0, 18),
        ),
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.white.withOpacity(0.05)
              : Colors.white.withOpacity(0.65),
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
    final color = glowColor ?? AxonColors.accent;
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
            color: SpatialColors.glassWhite,
            borderRadius: BorderRadius.circular(borderRadius),
            border: Border.all(
              color: SpatialColors.glassBorder,
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
  static const oxfordBlue = Color(0xFFFFF0F5); // Pastel Lavender Pink base
  static const oxfordBlueDark = Color(0xFFEEF2FF); // Periwinkle base
  static const oxfordBlueLight = Color(0xFFE0F2FE); // Soft Ice Blue base
  static const electricCyan = AxonBrandColors.electricCyan;
  static const electricCyanGlow = Color(0x1F3A86FF);
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
  static const surfaceElevated = Color(0xFFFFFFFF); // Clean white cards
  static const surfaceHighlight = Color(0xFFF1F5F9); // Active states
  static const textPrimary = Color(0xFF0F172A); // Deep navy-black
  static const textSecondary = Color(0xFF475569); // Professional gray
  static const textTertiary = Color(0xFF94A3B8); // Hints, placeholders
  static const divider = Color(0x1A0F172A); // Ultra-thin border (10% navy)
}

// Premium Dark Theme - Onyx Profile
class _AxonDark {
  static const oxfordBlue = Color(0xFF020204); // Pure obsidian black
  static const oxfordBlueDark = Color(0xFF020204); // Background
  static const oxfordBlueLight = Color(0xFF0A0A10); // Elevated primary card layer
  static const electricCyan = AxonBrandColors.neonCyan;
  static const electricCyanGlow = Color(0x3300F0FF);
  static const electricCyanDim = Color(0xFF00B0CC);
  static const slateGray = Color(0xFF94A3B8); // Muted slate for body text
  static const slateGrayLight = Color(0xFF64748B); // Tertiary text
  static const poor = AxonBrandColors.neonPink; // Neon Pink for alert/error
  static const poorGlow = Color(0x20FF007F);
  static const belowAverage = Color(0xFFFFA500); // Vibrant orange
  static const average = Color(0xFFFFFF00); // Neon yellow
  static const good = AxonBrandColors.neonGreen; // Neon green
  static const excellent = AxonBrandColors.neonCyan;
  static const excellentGlow = Color(0x2000F0FF);
  static const surface = Color(0xFF020204); // Background
  static const surfaceElevated = Color(0xFF0A0A10); // Primary cards
  static const surfaceHighlight = Color(0xFF161622); // Modals, floating elements
  static const textPrimary = Color(0xFFFFFFFF); // Crisp readability
  static const textSecondary = Color(0xFF94A3B8); // Muted slate
  static const textTertiary = Color(0xFF64748B);
  static const divider = Color(0xFF1E1E2C); // Subtle divider
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
  static Color get accent => AxonAccentPalette.scheme.accent;
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
  static Color get accentGlow => accent.withOpacity(0.25);
  static Color get warningGlow => warning.withOpacity(0.25);
  static Color get errorGlow => error.withOpacity(0.25);

  static Color performanceColor(double score) {
    if (score < 0.3) return poor;
    if (score < 0.5) return belowAverage;
    if (score < 0.7) return average;
    if (score < 0.85) return good;
    return excellent;
  }

  static Color performanceGlow(double score) {
    return performanceColor(score).withOpacity(0.2);
  }

  static String performanceLabel(double score) {
    if (score < 0.3) return 'Critical';
    if (score < 0.5) return 'Sub-optimal';
    if (score < 0.7) return 'Stable';
    if (score < 0.9) return 'Optimal';
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
        primary: _AxonLight.electricCyan,
        onPrimary: Colors.white,
        secondary: _AxonLight.poor,
        surface: _AxonLight.surface,
        onSurface: _AxonLight.textPrimary,
        error: _AxonLight.poor,
        outline: _AxonLight.divider,
      ),
      textTheme: _buildTextTheme(
        primary: _AxonLight.textPrimary,
        secondary: _AxonLight.textSecondary,
        tertiary: _AxonLight.textTertiary,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: _AxonLight.oxfordBlueDark,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.inter(
          color: _AxonLight.textPrimary,
          fontSize: 20,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
        ),
        iconTheme: IconThemeData(color: _AxonLight.textPrimary),
      ),
      cardTheme: CardThemeData(
        color: _AxonLight.surfaceElevated,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: _AxonLight.divider, width: 0.5),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: const Color(0x1FEEF2FF), // Soft translucent periwinkle background
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: _AxonLight.divider),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: _AxonLight.divider),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: _AxonLight.electricCyan, width: 1.5),
        ),
        labelStyle: GoogleFonts.inter(color: _AxonLight.textSecondary),
        hintStyle: GoogleFonts.inter(color: _AxonLight.textTertiary),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _AxonLight.electricCyan,
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
          foregroundColor: _AxonLight.textPrimary,
          side: BorderSide(color: _AxonLight.divider, width: 1),
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
          backgroundColor: _AxonLight.electricCyan,
          foregroundColor: Colors.white,
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
          foregroundColor: _AxonLight.electricCyan,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
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
      dividerTheme: DividerThemeData(
        color: _AxonLight.divider,
        thickness: 1,
      ),
      canvasColor: _AxonLight.oxfordBlueDark,
      dialogTheme: DialogThemeData(
        backgroundColor: _AxonLight.surfaceElevated,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: _AxonLight.divider, width: 0.5),
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
        selectedItemColor: _AxonLight.electricCyan,
        unselectedItemColor: _AxonLight.textTertiary,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      sliderTheme: SliderThemeData(
        thumbColor: _AxonLight.electricCyan,
        activeTrackColor: _AxonLight.electricCyan,
        inactiveTrackColor: _AxonLight.divider,
        trackHeight: 4,
        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 12),
        overlayShape: RoundSliderOverlayShape(overlayRadius: 20),
        tickMarkShape: const RoundSliderTickMarkShape(tickMarkRadius: 2),
      ),
    );
  }

  static TextTheme _buildTextTheme({
    required Color primary,
    required Color secondary,
    required Color tertiary,
  }) {
    return TextTheme(
      displayLarge: GoogleFonts.inter(
        color: primary,
        fontSize: 32,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
        height: 1.1,
      ),
      displayMedium: GoogleFonts.inter(
        color: primary,
        fontSize: 28,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
        height: 1.15,
      ),
      displaySmall: GoogleFonts.inter(
        color: primary,
        fontSize: 24,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.2,
        height: 1.2,
      ),
      headlineLarge: GoogleFonts.inter(
        color: primary,
        fontSize: 32,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
        height: 1.25,
      ),
      headlineMedium: GoogleFonts.inter(
        color: primary,
        fontSize: 28,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
        height: 1.3,
      ),
      headlineSmall: GoogleFonts.inter(
        color: primary,
        fontSize: 24,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.2,
        height: 1.35,
      ),
      titleLarge: GoogleFonts.inter(
        color: primary,
        fontSize: 20,
        fontWeight: FontWeight.w700,
        letterSpacing: 0,
      ),
      titleMedium: GoogleFonts.inter(
        color: primary,
        fontSize: 16,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.1,
      ),
      titleSmall: GoogleFonts.inter(
        color: secondary,
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.1,
      ),
      bodyLarge: GoogleFonts.inter(
        color: primary,
        fontSize: 16,
        fontWeight: FontWeight.w400,
        height: 1.5,
        letterSpacing: 0,
      ),
      bodyMedium: GoogleFonts.inter(
        color: secondary,
        fontSize: 14,
        fontWeight: FontWeight.w400,
        height: 1.5,
        letterSpacing: 0.1,
      ),
      bodySmall: GoogleFonts.inter(
        color: tertiary,
        fontSize: 12,
        fontWeight: FontWeight.w400,
        height: 1.4,
      ),
      labelLarge: GoogleFonts.inter(
        color: primary,
        fontSize: 14,
        fontWeight: FontWeight.w700,
        letterSpacing: 0.2,
      ),
      labelMedium: GoogleFonts.inter(
        color: tertiary,
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.4,
      ),
      labelSmall: GoogleFonts.inter(
        color: tertiary,
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
      scaffoldBackgroundColor: _AxonDark.oxfordBlueDark,
      colorScheme: ColorScheme.dark(
        primary: _AxonDark.electricCyan,
        onPrimary: Colors.black,
        secondary: _AxonDark.poor,
        surface: _AxonDark.surface,
        onSurface: _AxonDark.textPrimary,
        error: _AxonDark.poor,
        outline: _AxonDark.divider,
      ),
      textTheme: _buildTextTheme(
        primary: _AxonDark.textPrimary,
        secondary: _AxonDark.textSecondary,
        tertiary: _AxonDark.textTertiary,
      ),
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
        color: _AxonDark.surface,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: _AxonDark.divider, width: 0.5),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: _AxonDark.surface,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: _AxonDark.divider),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: _AxonDark.divider),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: _AxonDark.electricCyan, width: 1.5),
        ),
        labelStyle: GoogleFonts.inter(color: _AxonDark.textSecondary),
        hintStyle: GoogleFonts.inter(color: _AxonDark.textTertiary),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _AxonDark.electricCyan,
          foregroundColor: Colors.black,
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
          foregroundColor: _AxonDark.textPrimary,
          side: BorderSide(color: _AxonDark.divider, width: 1.0),
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
          backgroundColor: _AxonDark.electricCyan,
          foregroundColor: Colors.black,
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
          foregroundColor: _AxonDark.electricCyan,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
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
      dividerTheme: DividerThemeData(
        color: _AxonDark.divider,
        thickness: 1,
      ),
      canvasColor: _AxonDark.oxfordBlueDark,
      dialogTheme: DialogThemeData(
        backgroundColor: _AxonDark.surfaceElevated,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: _AxonDark.divider, width: 0.5),
        ),
      ),
      bottomSheetTheme: const BottomSheetThemeData(
        backgroundColor: _AxonDark.surfaceElevated,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        ),
      ),
      bottomNavigationBarTheme: BottomNavigationBarThemeData(
        backgroundColor: _AxonDark.surface,
        selectedItemColor: _AxonDark.electricCyan,
        unselectedItemColor: _AxonDark.textTertiary,
        type: BottomNavigationBarType.fixed,
        elevation: 0,
      ),
      sliderTheme: SliderThemeData(
        thumbColor: _AxonDark.electricCyan,
        activeTrackColor: _AxonDark.electricCyan,
        inactiveTrackColor: _AxonDark.divider,
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
              ? Colors.black.withOpacity(0.3)
              : Colors.black.withOpacity(0.04),
          offset: const Offset(0, 2),
          blurRadius: 6,
          spreadRadius: -2,
        ),
      ];

  static List<BoxShadow> get subtleShadow => [
        BoxShadow(
          color: AxonThemeMode.isDark
              ? Colors.black.withOpacity(0.4)
              : Colors.black.withOpacity(0.03),
          offset: const Offset(0, 1),
          blurRadius: 4,
        ),
      ];

  static List<BoxShadow> glowShadow(Color color) => [
        BoxShadow(
          color: color.withOpacity(0.28),
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
                const Color(0xFF020204),
                const Color(0xFF020204),
                const Color(0xFF0A0A10),
              ]
            : [
                const Color(0xFFFFF0F5),
                const Color(0xFFEEF2FF),
                const Color(0xFFE0F2FE),
              ],
        stops: const [0.0, 0.5, 1.0],
      );

  // Subtle gradient from primary accent to darker version (adds "weight" to buttons)
  static LinearGradient get cyanGradient => LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: AxonThemeMode.isDark
            ? [
                AxonBrandColors.neonCyan,
                const Color(0xFF00B0CC),
              ]
            : [
                AxonBrandColors.electricCyan,
                const Color(0xFF2563EB),
              ],
      );

  static LinearGradient get accentGradient => LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: AxonThemeMode.isDark
            ? [
                AxonBrandColors.neonCyan,
                AxonBrandColors.neonPurple,
                AxonBrandColors.neonPink,
              ]
            : [
                AxonBrandColors.electricCyan,
                const Color(0xFF2563EB),
                AxonBrandColors.deepMidnight,
              ],
      );

  // Polished functional gradients
  static LinearGradient get successGradient => LinearGradient(
        colors: AxonThemeMode.isDark
            ? [AxonBrandColors.neonGreen, const Color(0xFF00CC66)]
            : [const Color(0xFF10B981), const Color(0xFF059669)],
      );

  static LinearGradient get errorGradient => LinearGradient(
        colors: AxonThemeMode.isDark
            ? [AxonBrandColors.neonPink, const Color(0xFFCC0066)]
            : [const Color(0xFFE11D48), const Color(0xFFBE123C)],
      );

  static LinearGradient get warningGradient => LinearGradient(
        colors: AxonThemeMode.isDark
            ? [const Color(0xFFFFA500), const Color(0xFFCC8400)]
            : [const Color(0xFFF59E0B), const Color(0xFFB45309)],
      );

  static LinearGradient get excellentGradient => LinearGradient(
        colors: AxonThemeMode.isDark
            ? [AxonBrandColors.neonCyan, const Color(0xFF00B0CC)]
            : [AxonBrandColors.electricCyan, const Color(0xFF2563EB)],
      );

  static LinearGradient get purpleGradient => LinearGradient(
        colors: AxonThemeMode.isDark
            ? [AxonBrandColors.neonCyan, AxonBrandColors.neonPurple]
            : [AxonBrandColors.electricCyan, AxonBrandColors.deepMidnight],
      );

  static LinearGradient get averageGradient => LinearGradient(
        colors: AxonThemeMode.isDark
            ? [const Color(0xFFFFFF00), const Color(0xFFD4D400)]
            : [const Color(0xFFFBBF24), const Color(0xFFF59E0B)],
      );

  static LinearGradient performanceGradient(double score) {
    if (score < 0.3) return errorGradient;
    if (score < 0.5) return warningGradient;
    if (score < 0.7) return averageGradient;
    if (score < 0.85) return successGradient;
    return excellentGradient;
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

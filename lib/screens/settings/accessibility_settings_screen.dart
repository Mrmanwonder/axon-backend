// lib/screens/settings/accessibility_settings_screen.dart
// ─────────────────────────────────────────────────────────────────
// Accessibility Settings Screen
// Reading density, motion, contrast, and preset management
// ─────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../theme/app_theme.dart';
import '../../services/personalization_service.dart';

import '../../utils/layout_utils.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/axon_widgets.dart';

class AccessibilitySettingsScreen extends ConsumerStatefulWidget {
  const AccessibilitySettingsScreen({super.key});

  @override
  ConsumerState<AccessibilitySettingsScreen> createState() =>
      _AccessibilitySettingsScreenState();
}

class _AccessibilitySettingsScreenState
    extends ConsumerState<AccessibilitySettingsScreen> {
  AccessibilityPreset _preset = const AccessibilityPreset();
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPreset();
  }

  Future<void> _loadPreset() async {
    final svc = PersonalizationService();
    final preset = await svc.getAccessibility();
    if (mounted) {
      setState(() {
        _preset = preset;
        _isLoading = false;
      });
    }
  }

  Future<void> _savePreset(AccessibilityPreset preset) async {
    final svc = PersonalizationService();
    await svc.setAccessibility(preset);
    setState(() => _preset = preset);
  }

  Widget _buildCommandHeader(String title, String subtitle) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title.toUpperCase(),
          style: GoogleFonts.googleSans(
            color: const Color(0xFF3A86FF),
            fontWeight: FontWeight.bold,
            letterSpacing: 3,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          subtitle,
          style: GoogleFonts.googleSans(
            color: isDark ? Colors.white : Colors.black,
            fontSize: 28,
            fontWeight: FontWeight.w900,
          ),
        ),
        const SizedBox(height: 24),
      ],
    );
  }

  Widget _buildFloatingPreviewCard() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textStyle = TextStyle(
      fontSize: 14 * _preset.density.fontSizeMultiplier,
      height: _preset.density.lineHeight,
      color: isDark ? Colors.white : Colors.black,
      fontWeight: _preset.boldText ? FontWeight.bold : FontWeight.normal,
    );

    return Container(
      padding: EdgeInsets.all(_preset.density.cardPadding),
      decoration: BoxDecoration(
        color: isDark
            ? Colors.white.withOpacity(0.05)
            : Colors.grey.shade100,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: const Color(0xFF3A86FF).withOpacity(0.3),
          width: _preset.contrast.borderWidth,
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF3A86FF).withOpacity(0.1),
            blurRadius: 20,
            offset: const Offset(0, 8),
          )
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: const Color(0xFF3A86FF),
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
              const SizedBox(width: 8),
              Text('LIVE PREVIEW',
                  style: GoogleFonts.googleSans(
                    color: const Color(0xFF3A86FF),
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1,
                  )),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            'Kinematics — Physics',
            style: textStyle.copyWith(
              fontWeight: FontWeight.w700,
              fontSize: 16 * _preset.density.fontSizeMultiplier,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Motion in one dimension. Understand velocity, acceleration, '
            'and the equations of motion that govern how objects move.',
            style: textStyle.copyWith(
              color: isDark
                  ? Colors.white.withOpacity(0.7)
                  : Colors.black.withOpacity(0.7),
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: const Color(0xFF3A86FF).withOpacity(0.15),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              'Chapter 1 of 10',
              style: TextStyle(
                color: const Color(0xFF3A86FF),
                fontSize: 12 * _preset.density.fontSizeMultiplier,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: -0.1);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final pageColor = isDark ? AxonColors.background : Colors.white;
    final appBarColor = isDark ? AxonColors.background : Colors.white;
    final primaryText = isDark ? Colors.white : AxonColors.textPrimary;
    final bottomClearance = bottomDockClearance(context);

    return Scaffold(
      backgroundColor: pageColor,
      appBar: AppBar(
        backgroundColor: appBarColor,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: primaryText),
          onPressed: () => popOrGo(context, '/settings'),
        ),
      ),
      body: _isLoading
          ? ListView(
              padding: EdgeInsets.fromLTRB(20, 20, 20, bottomClearance),
              children: const [
                AxonSkeleton(height: 22, width: 140),
                SizedBox(height: 8),
                AxonSkeleton(height: 40),
                SizedBox(height: 24),
                AxonSkeleton(height: 180, radius: 20),
                SizedBox(height: 24),
                AxonSkeleton(height: 18, width: 120),
                SizedBox(height: 12),
                AxonSkeleton(height: 100, radius: 16),
                SizedBox(height: 24),
                AxonSkeleton(height: 18, width: 100),
                SizedBox(height: 12),
                AxonSkeleton(height: 100, radius: 16),
              ],
            )
          : ListView(
              padding: EdgeInsets.fromLTRB(20, 20, 20, bottomClearance),
              children: [
                _buildCommandHeader(
                  'ACCESSIBILITY',
                  'Display Settings',
                ),
                ListenableBuilder(
                  listenable: greyscaleModeNotifierProvider,
                  builder: (context, _) {
                    if (greyscaleModeNotifierProvider.suggestedForUser) {
                      return _buildGreyscaleSuggestionBanner();
                    }
                    return const SizedBox.shrink();
                  },
                ),
                _buildFloatingPreviewCard(),
                const SizedBox(height: 32),
                _buildSectionTitle('Theme Mode'),
                _buildThemeModeSelector(),
                const SizedBox(height: 24),
                _buildSectionTitle('Reading Density'),
                _buildDensityTiles(),
                const SizedBox(height: 24),
                _buildSectionTitle('Motion'),
                _buildMotionSelector(),
                const SizedBox(height: 24),
                _buildSectionTitle('Contrast'),
                _buildContrastSelector(),
                const SizedBox(height: 24),
                _buildSectionTitle('Additional'),
                _buildToggles(),
                const SizedBox(height: 32),
              ],
            ),
    );
  }

  Widget _buildSectionTitle(String title) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        title.toUpperCase(),
        style: GoogleFonts.googleSans(
          color: isDark ? Colors.white.withOpacity(0.6) : Colors.black.withOpacity(0.6),
          fontSize: 11,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.5,
        ),
      ),
    );
  }

  Widget _buildGreyscaleSuggestionBanner() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? AxonColors.cardSurface : Colors.grey.shade100,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: isDark ? AxonColors.divider : Colors.black12),
      ),
      child: Row(
        children: [
          Icon(Icons.dark_mode_outlined, color: Colors.amber[400], size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Low light detected',
                  style: GoogleFonts.googleSans(
                    color: isDark ? Colors.white : Colors.black87,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Enable greyscale for easier reading',
                  style: GoogleFonts.googleSans(
                    color: isDark ? Colors.white.withOpacity(0.6) : Colors.black.withOpacity(0.6),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            icon: Icon(Icons.close, color: isDark ? Colors.white54 : Colors.black54, size: 20),
            onPressed: () {
              greyscaleModeNotifierProvider.dismissSuggestion();
            },
          ),
          const SizedBox(width: 8),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF3A86FF),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            ),
            onPressed: () {
              _savePreset(_preset.copyWith(greyscaleMode: true));
              greyscaleModeNotifierProvider.dismissSuggestion();
            },
            child: Text(
              'Enable',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 300.ms).slideY(begin: -0.1);
  }

  Widget _buildDensityTiles() {
    return Row(
      children: [
        Expanded(
          child: _buildDensityTile(
            'Compact',
            'More content',
            ReadingDensity.compact,
            0.85,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildDensityTile(
            'Standard',
            'Balanced',
            ReadingDensity.normal,
            1.0,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildDensityTile(
            'Relaxed',
            'More space',
            ReadingDensity.spacious,
            1.15,
          ),
        ),
      ],
    );
  }

  Widget _buildThemeModeSelector() {
    return ListenableBuilder(
      listenable: AxonThemeMode.notifier,
      builder: (context, _) {
        final currentMode = AxonThemeMode.mode;
        return Row(
          children: [
            Expanded(
              child: _buildThemeModeTile(
                'System',
                ThemeMode.system,
                currentMode == ThemeMode.system,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildThemeModeTile(
                'Light',
                ThemeMode.light,
                currentMode == ThemeMode.light,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildThemeModeTile(
                'Dark',
                ThemeMode.dark,
                currentMode == ThemeMode.dark,
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildThemeModeTile(
    String label,
    ThemeMode mode,
    bool isActive,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return GestureDetector(
      onTap: () {
        AxonThemeMode.notifier.value = mode;
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeInOut,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isActive
              ? const Color(0xFF3A86FF).withOpacity(0.12)
              : (isDark ? Colors.white.withOpacity(0.03) : Colors.black.withOpacity(0.03)),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isActive
                ? const Color(0xFF3A86FF)
                : (isDark ? Colors.white10 : Colors.black12),
            width: isActive ? 2 : 1,
          ),
          boxShadow: isActive
              ? [
                  BoxShadow(
                    color: const Color(0xFF3A86FF).withOpacity(0.15),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  )
                ]
              : [],
        ),
        child: Column(
          children: [
            Icon(
              mode == ThemeMode.system
                  ? Icons.brightness_auto
                  : mode == ThemeMode.light
                      ? Icons.light_mode
                      : Icons.dark_mode,
              color: isActive ? const Color(0xFF3A86FF) : (isDark ? Colors.white : Colors.black87),
              size: 20,
            ),
            const SizedBox(height: 8),
            Text(
              label.toUpperCase(),
              style: GoogleFonts.googleSans(
                color: isActive ? const Color(0xFF3A86FF) : (isDark ? Colors.white : Colors.black87),
                fontSize: 12,
                fontWeight: FontWeight.bold,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDensityTile(
    String label,
    String subtitle,
    ReadingDensity density,
    double scale,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final isActive = _preset.density == density;
    return GestureDetector(
      onTap: () => _savePreset(_preset.copyWith(density: density)),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeInOut,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isActive
              ? const Color(0xFF3A86FF).withOpacity(0.12)
              : (isDark ? Colors.white.withOpacity(0.03) : Colors.black.withOpacity(0.03)),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isActive
                ? const Color(0xFF3A86FF)
                : (isDark ? Colors.white10 : Colors.black12),
            width: isActive ? 2 : 1,
          ),
          boxShadow: isActive
              ? [
                  BoxShadow(
                    color: const Color(0xFF3A86FF).withOpacity(0.15),
                    blurRadius: 12,
                    offset: const Offset(0, 4),
                  )
                ]
              : [],
        ),
        child: Column(
          children: [
            Text(
              label.toUpperCase(),
              style: GoogleFonts.googleSans(
                color: isActive ? const Color(0xFF3A86FF) : (isDark ? Colors.white : Colors.black87),
                fontSize: 12,
                fontWeight: FontWeight.bold,
                letterSpacing: 1,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: GoogleFonts.googleSans(
                color: isActive ? const Color(0xFF3A86FF).withOpacity(0.7) : (isDark ? Colors.white60 : Colors.black54),
                fontSize: 10,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
              decoration: BoxDecoration(
                color: isActive
                    ? const Color(0xFF3A86FF)
                    : (isDark ? Colors.white.withOpacity(0.1) : Colors.black.withOpacity(0.05)),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                '${scale}x',
                style: GoogleFonts.googleSans(
                  color: isActive
                      ? Colors.white
                      : (isDark ? Colors.white60 : Colors.black54),
                  fontSize: 9,
                ),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(duration: 300.ms);
  }

  Widget _buildMotionSelector() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? Colors.white.withOpacity(0.03) : Colors.black.withOpacity(0.03),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark ? Colors.white10 : Colors.black12,
        ),
      ),
      child: Column(
        children: [
          _buildMotionRow(MotionPreference.full, 'Full Animations',
              'All transitions and effects'),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildMotionRow(
              MotionPreference.reduced, 'Reduced', 'Essential animations only'),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildMotionRow(
              MotionPreference.minimal, 'Minimal', 'Page transitions only'),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildMotionRow(MotionPreference.none, 'None', 'No animations'),
        ],
      ),
    );
  }

  Widget _buildMotionRow(MotionPreference motion, String label, String desc) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final isActive = _preset.motion == motion;
    return GestureDetector(
      onTap: () => _savePreset(_preset.copyWith(motion: motion)),
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: TextStyle(
                          color:
                              isActive ? const Color(0xFF3A86FF) : (isDark ? Colors.white : Colors.black87),
                          fontSize: 14,
                          fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(desc,
                      style: TextStyle(
                          color: isDark ? Colors.white60 : Colors.black54,
                          fontSize: 11)),
                ],
              ),
            ),
            if (isActive)
              Icon(Icons.check_circle,
                  color: const Color(0xFF3A86FF), size: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildContrastSelector() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? Colors.white.withOpacity(0.03) : Colors.black.withOpacity(0.03),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark ? Colors.white10 : Colors.black12,
        ),
      ),
      child: Column(
        children: [
          _buildContrastRow(
              ContrastMode.standard, 'Standard', 'Default contrast'),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildContrastRow(
              ContrastMode.high, 'High Contrast', 'Maximum text visibility'),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildContrastRow(
              ContrastMode.inverted, 'Inverted', 'Light text on dark always'),
        ],
      ),
    );
  }

  Widget _buildContrastRow(ContrastMode contrast, String label, String desc) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final isActive = _preset.contrast == contrast;
    return GestureDetector(
      onTap: () => _savePreset(_preset.copyWith(contrast: contrast)),
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(label,
                      style: TextStyle(
                          color:
                              isActive ? const Color(0xFF3A86FF) : (isDark ? Colors.white : Colors.black87),
                          fontSize: 14,
                          fontWeight: FontWeight.w600)),
                  const SizedBox(height: 2),
                  Text(desc,
                      style: TextStyle(
                          color: isDark ? Colors.white60 : Colors.black54,
                          fontSize: 11)),
                ],
              ),
            ),
            if (isActive)
              Icon(Icons.check_circle,
                  color: const Color(0xFF3A86FF), size: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildToggles() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? Colors.white.withOpacity(0.03) : Colors.black.withOpacity(0.03),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark ? Colors.white10 : Colors.black12,
        ),
      ),
      child: Column(
        children: [
          _buildToggleRow(
            'Reduce Transparency',
            'Disables blur and glassmorphism effects',
            _preset.reduceTransparency,
            (v) => _savePreset(_preset.copyWith(reduceTransparency: v)),
          ),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildToggleRow(
            'Bold Text',
            'Increases text weight for better visibility',
            _preset.boldText,
            (v) => _savePreset(_preset.copyWith(boldText: v)),
          ),
          Divider(color: isDark ? Colors.white12 : Colors.black12),
          _buildToggleRow(
            'Greyscale Mode',
            'Removes color for easier reading in dark',
            _preset.greyscaleMode,
            (v) => _savePreset(_preset.copyWith(greyscaleMode: v)),
          ),
        ],
      ),
    );
  }

  Widget _buildToggleRow(
    String label,
    String description,
    bool value,
    Function(bool) onChanged,
  ) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: TextStyle(
                        color: isDark ? Colors.white : Colors.black87,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 2),
                Text(description,
                    style: TextStyle(
                        color: isDark ? Colors.white60 : Colors.black54,
                        fontSize: 11)),
              ],
            ),
          ),
          Switch(
            value: value,
            onChanged: onChanged,
            activeThumbColor: const Color(0xFF3A86FF),
          ),
        ],
      ),
    );
  }
}

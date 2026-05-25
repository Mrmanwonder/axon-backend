// lib/screens/settings/coaching_persona_screen.dart
// ─────────────────────────────────────────────────────────────────
// Coaching Persona Selector
// Choose your coach's personality
// ─────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_animate/flutter_animate.dart';
import '../../theme/app_theme.dart';
import '../../services/haptics_service.dart';
import '../../widgets/common/axon_widgets.dart';
import '../../services/coaching_persona_service.dart';
import '../../services/personalization_service.dart';
import '../../utils/layout_utils.dart';
import '../../utils/nav_utils.dart';

class CoachingPersonaScreen extends ConsumerStatefulWidget {
  const CoachingPersonaScreen({super.key});

  @override
  ConsumerState<CoachingPersonaScreen> createState() =>
      _CoachingPersonaScreenState();
}

class _CoachingPersonaScreenState extends ConsumerState<CoachingPersonaScreen> {
  CoachingPersona _selected = CoachingPersona.mentor;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final svc = PersonalizationService();
    final state = await svc.load();
    if (mounted) {
      setState(() {
        _selected = state.coachingPersona;
        _isLoading = false;
      });
    }
  }

  Widget _buildCommandHeader(String title, String subtitle) {
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
            color: Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.w900,
          ),
        ),
        const SizedBox(height: 24),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final pageColor = isDark ? AxonColors.background : Colors.white;
    final primaryText = isDark ? Colors.white : AxonColors.textPrimary;
    final secondaryText = isDark
        ? AxonColors.textSecondary
        : Colors.black.withValues(alpha: 0.72);
    final bottomClearance = bottomDockClearance(context);

    return Scaffold(
      backgroundColor: pageColor,
      appBar: AppBar(
        backgroundColor: pageColor,
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
                AxonSkeleton(height: 140, radius: 24),
                SizedBox(height: 16),
                AxonSkeleton(height: 140, radius: 24),
                SizedBox(height: 16),
                AxonSkeleton(height: 140, radius: 24),
              ],
            )
          : ListView(
              padding: EdgeInsets.fromLTRB(20, 20, 20, bottomClearance),
              children: [
                _buildCommandHeader(
                  'COACHING STYLE',
                  'Character Profile',
                ),
                Text(
                  'How Axon communicates with you across reminders, reports, nudges, and study feedback.',
                  style: GoogleFonts.googleSans(
                    color: secondaryText,
                    fontSize: 14,
                    height: 1.5,
                  ),
                ),
                const SizedBox(height: 24),
                ...CoachingPersona.values.asMap().entries.map((e) =>
                    _buildPersonaCard(
                      e.value,
                      _selected == e.value,
                      isDark,
                    )
                        .animate()
                        .fadeIn(duration: 400.ms, delay: (e.key * 100).ms)
                        .slideX(begin: 0.1)),
                const SizedBox(height: 8),
              ],
            ),
    );
  }

  Widget _buildPersonaCard(
      CoachingPersona persona, bool isSelected, bool isDark) {
    return GestureDetector(
      onTap: () async {
        AxonHaptics.lightImpact();
        final svc = PersonalizationService();
        await svc.setCoachingPersona(persona);
        CoachingPersonaService().setActivePersona(persona);
        setState(() => _selected = persona);
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeInOut,
        padding: const EdgeInsets.all(20),
        margin: const EdgeInsets.only(bottom: 16),
        decoration: BoxDecoration(
          color: isSelected
              ? const Color(0xFF3A86FF).withValues(alpha: 0.12)
              : Colors.white.withValues(alpha: 0.03),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: isSelected
                ? const Color(0xFF3A86FF)
                : Colors.white.withValues(alpha: 0.1),
            width: isSelected ? 2 : 1,
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: const Color(0xFF3A86FF).withValues(alpha: 0.15),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  )
                ]
              : [],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: isSelected
                        ? const Color(0xFF3A86FF)
                        : Colors.white.withValues(alpha: 0.05),
                    shape: BoxShape.circle,
                  ),
                  child: Center(
                    child: Text(
                      persona.avatarEmoji,
                      style: GoogleFonts.googleSans(
                        color: isSelected
                            ? Colors.black
                            : Colors.white.withValues(alpha: 0.6),
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Text(
                    persona.displayName.toUpperCase(),
                    style: GoogleFonts.googleSans(
                      color: Colors.white,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.5,
                    ),
                  ),
                ),
                if (isSelected)
                  Icon(Icons.flash_on_rounded,
                      color: const Color(0xFF3A86FF), size: 18),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              persona.description,
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 14,
                height: 1.5,
              ),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                'VOICE: ${persona.voice.toUpperCase()}',
                style: GoogleFonts.googleSans(
                  color: const Color(0xFF3A86FF),
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

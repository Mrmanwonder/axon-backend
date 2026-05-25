// lib/screens/auth/motivation_selection_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:go_router/go_router.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../models/models.dart';
import '../../services/app_state.dart';
import '../../services/resource_crawler_service.dart';
import '../../services/google_drive_downloader.dart';
import '../../theme/app_theme.dart';
import '../../widgets/common/auth_frame.dart';
import '../../widgets/common/axon_widgets.dart';
import '../../widgets/common/gemma_download_progress.dart';

class MotivationSelectionScreen extends ConsumerStatefulWidget {
  final String board;
  final List<String> subjects;

  const MotivationSelectionScreen({
    super.key,
    required this.board,
    required this.subjects,
  });

  @override
  ConsumerState<MotivationSelectionScreen> createState() =>
      _MotivationSelectionScreenState();
}

class _MotivationSelectionScreenState
    extends ConsumerState<MotivationSelectionScreen> {
  MotivationStyle _motivationStyle = MotivationStyle.logicBased;
  double _targetHours = 4;
  bool _saving = false;
  String? _error;
  final _resourceService = ResourceCrawlerService();
  final _googleDriveDownloader = GoogleDriveDownloader.instance;
  bool _showGemmaDownload = false;
  bool _gemmaReady = false;

  @override
  void initState() {
    super.initState();
    final profile = ref.read(authStateProvider).user;
    if (profile != null) {
      _motivationStyle = profile.motivationStyle;
      _targetHours = _snapHalfHour(profile.targetStudyHours.clamp(1, 12));
    }
    _checkGemmaStatus();
  }

  Future<void> _checkGemmaStatus() async {
    final downloaded = await _googleDriveDownloader.isModelDownloaded();
    if (downloaded) {
      setState(() => _gemmaReady = true);
    }
  }

  double _snapHalfHour(double value) => (value * 2).round() / 2;

  String _formatHours(double hours) {
    if ((hours % 1).abs() < 0.001) {
      return '${hours.toInt()}h';
    }
    return '${hours.toStringAsFixed(1)}h';
  }

  Future<void> _save() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() => _error = 'Authentication expired. Sign in again.');
      return;
    }
    if (widget.subjects.isEmpty) {
      setState(() => _error = 'Select at least one subject.');
      return;
    }

    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      final profileService = ref.read(profileServiceProvider);
      final authNotifier = ref.read(authStateProvider.notifier);
      final savedProfile = await profileService
          .saveOnboarding(
            user: user,
            displayName:
                user.displayName ?? user.email?.split('@').first ?? 'Student',
            board: widget.board,
            subjects: widget.subjects,
            targetHours: _targetHours,
            motivationStyle: _motivationStyle,
          )
          .timeout(const Duration(seconds: 12));
      await authNotifier.applyOnboardingProfile(savedProfile);
      await authNotifier.updateMotivationStyle(_motivationStyle);
      ref.read(metricsProvider.notifier).updateTargetStudyHours(_targetHours);
      if (widget.subjects.isNotEmpty) {
        ref.read(metricsProvider.notifier).updateSubject(widget.subjects.first);
      }
      await _resourceService
          .scheduleInitialCrawl(
            uid: user.uid,
            board: widget.board,
            subjects: widget.subjects,
          )
          .timeout(const Duration(seconds: 8));

      if (!mounted) return;

      // Show Gemma download progress
      setState(() {
        _saving = false;
        _showGemmaDownload = true;
      });

      // Start Gemma model download in background
      _googleDriveDownloader.downloadGemmaModel().then((_) {
        if (mounted) {
          setState(() {});
        }
      }).catchError((e) {
        debugPrint('Gemma download failed: $e');
      });

      // Start past papers download in background
      _googleDriveDownloader
          .downloadPastPapersForSubjects(widget.subjects)
          .then((_) {
        debugPrint('Past papers download completed');
      }).catchError((e) {
        debugPrint('Past papers download failed: $e');
      });
    } catch (e) {
      setState(() {
        _saving = false;
        _error = 'Failed to save onboarding profile.';
      });
    }
  }

  void _navigateHome() {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Profile saved. Axon is preparing resources in the background.',
          style: GoogleFonts.googleSans(color: Colors.white),
        ),
        backgroundColor: AxonColors.accent,
      ),
    );
    context.go('/home');
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AxonThemeMode.isDark;
    if (_showGemmaDownload) {
      return Scaffold(
        backgroundColor:
            isDark ? const Color(0xFF000000) : const Color(0xFFFFFFFF),
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 40),
                Text(
                  'SETTING UP AXON',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontWeight: FontWeight.w700,
                    fontSize: 20,
                    letterSpacing: 2,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Downloading offline AI model for local inference.',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 32),
                GemmaDownloadProgressWidget(
                  downloader: _googleDriveDownloader,
                  onComplete: () {
                    if (!mounted) return;
                    setState(() => _gemmaReady = true);
                  },
                  margin: EdgeInsets.zero,
                ),
                const Spacer(),
                SizedBox(
                  width: double.infinity,
                  child: CyberButton(
                    label: _gemmaReady ? 'Go to Dashboard' : 'Skip & Continue',
                    isLoading: false,
                    onTap: _navigateHome,
                  ),
                ),
                const SizedBox(height: 16),
              ],
            ),
          ),
        ),
      );
    }

    return AuthFrame(
      eyebrow: 'STEP 3 OF 3',
      title: 'Tune your coaching.',
      subtitle:
          'Choose the tone and daily target that should drive Axon across the app.',
      onBack: () => context.go('/auth/subjects', extra: widget.board),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_error != null) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AxonColors.error.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
                border:
                    Border.all(color: AxonColors.error.withValues(alpha: 0.3)),
              ),
              child: Text(
                _error!,
                style: GoogleFonts.googleSans(
                    color: AxonColors.error, fontSize: 13),
              ),
            ),
            const SizedBox(height: 16),
          ],
          ...MotivationStyle.values.map((style) {
            final selected = style == _motivationStyle;
            return Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: GestureDetector(
                onTap: () => setState(() => _motivationStyle = style),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: selected
                        ? AxonColors.accent.withValues(alpha: 0.12)
                        : AxonColors.surface,
                    borderRadius: BorderRadius.circular(AxonRadius.lg),
                    border: Border.all(
                      color: selected ? AxonColors.accent : AxonColors.divider,
                      width: selected ? 1.5 : 1,
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 24,
                        height: 24,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color:
                              selected ? AxonColors.accent : Colors.transparent,
                          border: Border.all(
                            color: selected
                                ? AxonColors.accent
                                : AxonColors.textTertiary,
                            width: 2,
                          ),
                        ),
                        child: selected
                            ? const Icon(Icons.check,
                                size: 14, color: Colors.white)
                            : null,
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              style.displayName,
                              style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary,
                                fontWeight: FontWeight.w600,
                                fontSize: 16,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              style.description,
                              style: GoogleFonts.googleSans(
                                color: AxonColors.textTertiary,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          }),
          const SizedBox(height: 24),
          Text(
            'Daily study target',
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontWeight: FontWeight.w600,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: AxonSpatialSlider(
                  label: 'Target Study Hours',
                  value: ((_targetHours - 1) / 11).clamp(0.0, 1.0),
                  valueFormatter: (_) => _formatHours(_targetHours),
                  onChanged: (v) => setState(
                    () => _targetHours = _snapHalfHour(1 + (v * 11)),
                  ),
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(AxonRadius.md),
                ),
                child: Text(
                  _formatHours(_targetHours),
                  style: GoogleFonts.googleSans(
                    color: AxonColors.accent,
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 32),
          SizedBox(
            width: double.infinity,
            child: CyberButton(
              label: _saving ? 'Setting up...' : 'Start Learning',
              isLoading: _saving,
              onTap: _saving ? null : _save,
            ),
          ),
        ],
      ),
    );
  }
}

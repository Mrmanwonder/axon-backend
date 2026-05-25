import 'dart:async';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../services/study_lock_service.dart';
import '../../services/overlay_permission_service.dart';
import '../../theme/app_theme.dart';
import '../../services/haptics_service.dart';

final _studyLockService = StudyLockService.instance;

// Providers
final studyLockEnabledProvider = StateProvider<bool>((ref) => false);
final requiredMinutesProvider = StateProvider<int>((ref) => 60);
final blockedAppsProvider = StateProvider<List<String>>((ref) => []);
final todayProgressProvider = StateProvider<StudyLockProgress>((ref) => StudyLockProgress.empty());

class StudyLockProgress {
  final int loggedMinutes;
  final int requiredMinutes;
  final DateTime? sessionStart;
  final bool isActive;

  StudyLockProgress({
    required this.loggedMinutes,
    required this.requiredMinutes,
    this.sessionStart,
    required this.isActive,
  });

  factory StudyLockProgress.empty() => StudyLockProgress(
        loggedMinutes: 0,
        requiredMinutes: 60,
        sessionStart: null,
        isActive: false,
      );

  int get remainingMinutes => (requiredMinutes - loggedMinutes).clamp(0, requiredMinutes);
  double get progress => requiredMinutes > 0 ? (loggedMinutes / requiredMinutes).clamp(0.0, 1.0) : 0.0;
  bool get isComplete => loggedMinutes >= requiredMinutes;
}

class StudyLockSettingsScreen extends ConsumerStatefulWidget {
  const StudyLockSettingsScreen({super.key});

  @override
  ConsumerState<StudyLockSettingsScreen> createState() => _StudyLockSettingsScreenState();
}

class _StudyLockSettingsScreenState extends ConsumerState<StudyLockSettingsScreen> {
  bool _isLoading = true;
  Timer? _progressTimer;

  @override
  void initState() {
    super.initState();
    _loadState();
  }

  @override
  void dispose() {
    _progressTimer?.cancel();
    super.dispose();
  }

  Future<void> _loadState() async {
    final prefs = await SharedPreferences.getInstance();
    final enabled = prefs.getBool('study_lock_enabled') ?? false;
    final required = prefs.getInt('study_lock_required_minutes') ?? 60;
    final blockedList = prefs.getStringList('study_lock_blocked_apps') ?? [];
    final isActive = await _studyLockService.isActive();
    final logged = await _studyLockService.getLoggedMinutes();

    ref.read(studyLockEnabledProvider.notifier).state = enabled;
    ref.read(requiredMinutesProvider.notifier).state = required;
    ref.read(blockedAppsProvider.notifier).state = blockedList;
    ref.read(todayProgressProvider.notifier).state = StudyLockProgress(
      loggedMinutes: logged,
      requiredMinutes: required,
      isActive: isActive,
    );

    setState(() => _isLoading = false);

    if (isActive) {
      _startProgressTimer();
    }
  }

  void _startProgressTimer() {
    _progressTimer?.cancel();
    _progressTimer = Timer.periodic(const Duration(minutes: 1), (_) {
      _refreshProgress();
    });
  }

  Future<void> _refreshProgress() async {
    final logged = await _studyLockService.getLoggedMinutes();
    final required = ref.read(requiredMinutesProvider);
    final isActive = await _studyLockService.isActive();

    ref.read(todayProgressProvider.notifier).state = StudyLockProgress(
      loggedMinutes: logged,
      requiredMinutes: required,
      isActive: isActive,
    );

    if (!isActive && _progressTimer?.isActive == true) {
      _progressTimer?.cancel();
    }
  }

  Future<void> _saveEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('study_lock_enabled', value);
    ref.read(studyLockEnabledProvider.notifier).state = value;
    AxonHaptics.mediumImpact();
  }

  Future<void> _saveRequiredMinutes(int minutes) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('study_lock_required_minutes', minutes);
    ref.read(requiredMinutesProvider.notifier).state = minutes;
    AxonHaptics.selectionClick();
  }

  Future<void> _saveBlockedApps(List<String> apps) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList('study_lock_blocked_apps', apps);
    ref.read(blockedAppsProvider.notifier).state = apps;
    AxonHaptics.selectionClick();
  }

  Future<void> _startStudyLock() async {
    final apps = ref.read(blockedAppsProvider);
    final required = ref.read(requiredMinutesProvider);

    if (apps.isEmpty) {
      AlertService.show(
        context: context,
        title: 'Select Apps',
        message: 'Please select at least one app to block during your study session.',
        actionLabel: 'OK',
        onAction: () {},
      );
      return;
    }

    final hasOverlay = await overlayPermissionService.checkAndRequestOverlayPermission();
    final hasAccessibility = await overlayPermissionService.isAccessibilityServiceEnabled();

    if (!hasOverlay || !hasAccessibility) {
      if (mounted) {
        AlertService.show(
          context: context,
          title: 'Permissions Required',
          message: hasOverlay
              ? 'Enable the AXON accessibility service so Study Lock can detect blocked apps.'
              : 'Allow AXON to display over other apps so Study Lock can show the lock screen.',
          actionLabel: 'Open Settings',
          onAction: () {
            if (hasOverlay) {
              overlayPermissionService.openAccessibilitySettings();
            } else {
              overlayPermissionService.openOverlaySettings();
            }
          },
        );
      }
      return;
    }

    await _studyLockService.setDistractionApps(apps, required);
    await _saveEnabled(true);
    _startProgressTimer();

    ref.read(todayProgressProvider.notifier).state = StudyLockProgress(
      loggedMinutes: 0,
      requiredMinutes: required,
      isActive: true,
    );

    if (mounted) {
      AlertService.show(
        context: context,
        title: 'Study Lock Active',
        message: 'Complete $required minutes of focused study to unlock your blocked apps.',
        actionLabel: 'OK',
        onAction: () {},
      );
    }
  }

  Future<void> _stopStudyLock() async {
    await _studyLockService.deactivateStudyLock();
    _progressTimer?.cancel();
    await _saveEnabled(false);

    ref.read(todayProgressProvider.notifier).state = StudyLockProgress(
      loggedMinutes: 0,
      requiredMinutes: ref.read(requiredMinutesProvider),
      isActive: false,
    );

    if (mounted) {
      AlertService.show(
        context: context,
        title: 'Study Lock Stopped',
        message: 'Blocked apps are available again.',
        actionLabel: 'OK',
        onAction: () {},
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final progress = ref.watch(todayProgressProvider);
    final isEnabled = ref.watch(studyLockEnabledProvider);
    final requiredMinutes = ref.watch(requiredMinutesProvider);
    final blockedApps = ref.watch(blockedAppsProvider);

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Container(
        decoration: BoxDecoration(gradient: AxonGradients.backgroundGradient),
        child: SafeArea(
          child: _isLoading
              ? Center(child: CircularProgressIndicator(color: AxonColors.accent))
              : CustomScrollView(
                  slivers: [
                    SliverAppBar(
                      backgroundColor: Colors.transparent,
                      floating: true,
                      pinned: false,
                      leading: IconButton(
                        icon: Icon(Icons.arrow_back_rounded, color: AxonColors.textPrimary),
                        onPressed: () => Navigator.pop(context),
                      ),
                      title: Text(
                        'Study Lock',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textPrimary,
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      centerTitle: true,
                    ),
                    SliverPadding(
                      padding: const EdgeInsets.all(20),
                      sliver: SliverList(
                        delegate: SliverChildListDelegate([
                          // Current Progress Card
                          _ProgressCard(
                            progress: progress,
                            isEnabled: isEnabled,
                            onStartLock: _startStudyLock,
                            onStopLock: _stopStudyLock,
                          ).animate().fadeIn(duration: 400.ms),

                          const SizedBox(height: 16),

                          // Permission Status Banner
                          _PermissionBanner(),

                          const SizedBox(height: 24),

                          // Duration Section
                          _SectionHeader('Daily Study Target'),
                          const SizedBox(height: 12),
                          _DurationSelector(
                            selectedMinutes: requiredMinutes,
                            onChanged: (mins) {
                              _saveRequiredMinutes(mins);
                              ref.read(requiredMinutesProvider.notifier).state = mins;
                            },
                          ).animate().fadeIn(delay: 100.ms),

                          const SizedBox(height: 24),

                          // Blocked Apps Section
                          _SectionHeader('Blocked Apps'),
                          const SizedBox(height: 4),
                          Text(
                            'These apps will be locked until you complete your study target',
                            style: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary,
                              fontSize: 12,
                            ),
                          ),
                          const SizedBox(height: 16),
                          _BlockedAppsEditor(
                            selectedApps: blockedApps,
                            onAppsChanged: _saveBlockedApps,
                          ).animate().fadeIn(delay: 200.ms),

                          const SizedBox(height: 32),

                          // Presets
                          _SectionHeader('Quick Presets'),
                          const SizedBox(height: 12),
                          _PresetsRow(
                            onSelect: (minutes, apps) async {
                              await _saveRequiredMinutes(minutes);
                              await _saveBlockedApps(apps);
                            },
                          ).animate().fadeIn(delay: 300.ms),

                          const SizedBox(height: 40),

                          // Info Card
                          _InfoCard().animate().fadeIn(delay: 400.ms),

                          const SizedBox(height: 100),
                        ]),
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader(this.title);

  @override
  Widget build(BuildContext context) {
    return Text(
      title,
      style: GoogleFonts.googleSans(
        color: AxonColors.textPrimary,
        fontSize: 16,
        fontWeight: FontWeight.w600,
      ),
    );
  }
}

class _FocusGlassPanel extends StatelessWidget {
  final Widget child;
  final EdgeInsets padding;
  final Color? accent;

  const _FocusGlassPanel({
    required this.child,
    this.padding = const EdgeInsets.all(16),
    this.accent,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 26, sigmaY: 26),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: const Color(0xCC0B0D0E),
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: Colors.white.withValues(alpha: 0.16)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.35),
                blurRadius: 28,
                offset: const Offset(0, 18),
              ),
              if (accent != null)
                BoxShadow(
                  color: accent!.withValues(alpha: 0.16),
                  blurRadius: 32,
                  offset: const Offset(0, 12),
                ),
            ],
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white.withValues(alpha: 0.11),
                Colors.white.withValues(alpha: 0.035),
                Colors.black.withValues(alpha: 0.12),
              ],
            ),
          ),
          child: child,
        ),
      ),
    );
  }
}

class _ProgressCard extends StatelessWidget {
  final StudyLockProgress progress;
  final bool isEnabled;
  final VoidCallback onStartLock;
  final VoidCallback onStopLock;

  const _ProgressCard({
    required this.progress,
    required this.isEnabled,
    required this.onStartLock,
    required this.onStopLock,
  });

  @override
  Widget build(BuildContext context) {
    return _FocusGlassPanel(
      padding: const EdgeInsets.all(20),
      accent: isEnabled ? AxonColors.accent : null,
      child: Column(
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: isEnabled
                      ? AxonColors.accent.withValues(alpha: 0.15)
                      : AxonColors.surfaceElevated,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(
                  isEnabled ? Icons.lock_rounded : Icons.lock_open_rounded,
                  color: isEnabled ? AxonColors.accent : AxonColors.textTertiary,
                  size: 24,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      isEnabled
                          ? progress.isComplete
                              ? 'Target Reached!'
                              : 'Study Session Active'
                          : 'Study Lock Inactive',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      isEnabled
                          ? '${progress.remainingMinutes} min remaining'
                          : 'Set your daily target to get started',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              ),
              if (isEnabled)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.green.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    'ACTIVE',
                    style: GoogleFonts.googleSans(
                      color: Colors.green,
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
            ],
          ),
          if (isEnabled) ...[
            const SizedBox(height: 20),
            // Progress Bar
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: LinearProgressIndicator(
                value: progress.progress,
                minHeight: 10,
                backgroundColor: AxonColors.surfaceElevated,
                valueColor: AlwaysStoppedAnimation(
                  Color.lerp(AxonColors.accent, Colors.green, progress.progress) ?? AxonColors.accent,
                ),
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '${progress.loggedMinutes} min completed',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary,
                    fontSize: 12,
                  ),
                ),
                Text(
                  '${progress.requiredMinutes} min target',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            GestureDetector(
              onTap: onStopLock,
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: AxonColors.error.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AxonColors.error.withValues(alpha: 0.3)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.stop_circle_outlined, color: AxonColors.error, size: 18),
                    const SizedBox(width: 8),
                    Text(
                      'Stop Session',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.error,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ] else ...[
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: onStartLock,
                style: ElevatedButton.styleFrom(
                backgroundColor: AxonColors.accent,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                elevation: 0,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                child: Text(
                  'Start Study Lock',
                  style: GoogleFonts.googleSans(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _DurationSelector extends StatelessWidget {
  final int selectedMinutes;
  final ValueChanged<int> onChanged;

  const _DurationSelector({
    required this.selectedMinutes,
    required this.onChanged,
  });

  static const _options = [15, 30, 45, 60, 90, 120, 180];

  @override
  Widget build(BuildContext context) {
    return _FocusGlassPanel(
      padding: const EdgeInsets.all(16),
      accent: AxonColors.accent,
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Required Minutes',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$selectedMinutes min',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.accent,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: AxonColors.electricCyan,
              inactiveTrackColor: AxonColors.electricCyan.withValues(alpha: 0.3),
              thumbColor: AxonColors.electricCyan,
              overlayColor: AxonColors.electricCyan.withValues(alpha: 0.2),
              trackHeight: 6,
            ),
            child: Slider(
              value: selectedMinutes.toDouble(),
              min: 15,
              max: 180,
              divisions: 11,
              onChanged: (v) => onChanged(v.round()),
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _options.map((mins) {
              final isSelected = selectedMinutes == mins;
              return GestureDetector(
                onTap: () => onChanged(mins),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: isSelected
                        ? AxonColors.accent
                        : Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: isSelected
                          ? AxonColors.accent
                          : Colors.white.withValues(alpha: 0.12),
                    ),
                  ),
                  child: Text(
                    _formatDuration(mins),
                    style: GoogleFonts.googleSans(
                      color: isSelected ? Colors.white : AxonColors.textSecondary,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  String _formatDuration(int mins) {
    if (mins >= 60) {
      final hours = mins ~/ 60;
      final remaining = mins % 60;
      if (remaining == 0) return '${hours}h';
      return '${hours}h ${remaining}m';
    }
    return '${mins}m';
  }
}

class _BlockedAppsEditor extends StatefulWidget {
  final List<String> selectedApps;
  final ValueChanged<List<String>> onAppsChanged;

  const _BlockedAppsEditor({
    required this.selectedApps,
    required this.onAppsChanged,
  });

  @override
  State<_BlockedAppsEditor> createState() => _BlockedAppsEditorState();
}

class _BlockedAppsEditorState extends State<_BlockedAppsEditor> {
  late Set<String> _selected;

  @override
  void initState() {
    super.initState();
    _selected = Set<String>.from(widget.selectedApps);
  }

  void _toggle(String packageName) {
    setState(() {
      if (_selected.contains(packageName)) {
        _selected.remove(packageName);
      } else {
        _selected.add(packageName);
      }
    });
    widget.onAppsChanged(_selected.toList());
  }

  void _selectAll(List<AppBlockerService> apps) {
    setState(() {
      for (final app in apps) {
        _selected.add(app.packageName);
      }
    });
    widget.onAppsChanged(_selected.toList());
  }

  void _deselectAll(List<AppBlockerService> apps) {
    setState(() {
      for (final app in apps) {
        _selected.remove(app.packageName);
      }
    });
    widget.onAppsChanged(_selected.toList());
  }

  @override
  Widget build(BuildContext context) {
    final socialApps = CommonBlockedApps.socialMedia;
    final videoApps = CommonBlockedApps.video;

    return Column(
      children: [
        _AppCategoryRow(
          title: 'Social Media',
          apps: socialApps,
          selected: _selected,
          onToggle: _toggle,
          onSelectAll: () => _selectAll(socialApps),
          onDeselectAll: () => _deselectAll(socialApps),
        ),
        const SizedBox(height: 16),
        _AppCategoryRow(
          title: 'Video & Streaming',
          apps: videoApps,
          selected: _selected,
          onToggle: _toggle,
          onSelectAll: () => _selectAll(videoApps),
          onDeselectAll: () => _deselectAll(videoApps),
        ),
        const SizedBox(height: 16),
        // Custom app entry
        _CustomAppInput(
          onAdd: (packageName, appName) {
            setState(() {
              _selected.add(packageName);
            });
            widget.onAppsChanged(_selected.toList());
          },
        ),
      ],
    );
  }
}

class _AppCategoryRow extends StatelessWidget {
  final String title;
  final List<AppBlockerService> apps;
  final Set<String> selected;
  final ValueChanged<String> onToggle;
  final VoidCallback onSelectAll;
  final VoidCallback onDeselectAll;

  const _AppCategoryRow({
    required this.title,
    required this.apps,
    required this.selected,
    required this.onToggle,
    required this.onSelectAll,
    required this.onDeselectAll,
  });

  @override
  Widget build(BuildContext context) {
    final allSelected = apps.every((a) => selected.contains(a.packageName));
    final someSelected = apps.any((a) => selected.contains(a.packageName));

    return _FocusGlassPanel(
      padding: const EdgeInsets.all(16),
      accent: someSelected ? AxonColors.accent : null,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                title,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              GestureDetector(
                onTap: someSelected ? onDeselectAll : onSelectAll,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: AxonColors.accent.withValues(alpha: 0.18)),
                  ),
                  child: Text(
                    allSelected ? 'Deselect All' : 'Select All',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.accent,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: apps.map((app) {
              final isSelected = selected.contains(app.packageName);
              return FilterChip(
                label: Text(app.appName),
                selected: isSelected,
                onSelected: (_) => onToggle(app.packageName),
                selectedColor: AxonColors.accent.withValues(alpha: 0.22),
                backgroundColor: Colors.white.withValues(alpha: 0.04),
                checkmarkColor: AxonColors.accent,
                labelStyle: GoogleFonts.googleSans(
                  color: isSelected ? AxonColors.accent : AxonColors.textSecondary,
                  fontSize: 12,
                ),
                side: BorderSide(
                  color: isSelected
                      ? AxonColors.accent
                      : Colors.white.withValues(alpha: 0.14),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

class _CustomAppInput extends StatefulWidget {
  final void Function(String packageName, String appName) onAdd;

  const _CustomAppInput({required this.onAdd});

  @override
  State<_CustomAppInput> createState() => _CustomAppInputState();
}

class _CustomAppInputState extends State<_CustomAppInput> {
  final _packageController = TextEditingController();
  final _nameController = TextEditingController();

  @override
  void dispose() {
    _packageController.dispose();
    _nameController.dispose();
    super.dispose();
  }

  void _add() {
    final package = _packageController.text.trim();
    final name = _nameController.text.trim();
    if (package.isNotEmpty && name.isNotEmpty) {
      widget.onAdd(package, name);
      _packageController.clear();
      _nameController.clear();
    }
  }

  @override
  Widget build(BuildContext context) {
    return _FocusGlassPanel(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Custom App',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: AxonColors.warning.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'BETA',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.warning,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                flex: 2,
                child: TextField(
                  controller: _nameController,
                  decoration: InputDecoration(
                    hintText: 'App name',
                    hintStyle: GoogleFonts.googleSans(color: AxonColors.textTertiary, fontSize: 13),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
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
                      borderSide: BorderSide(color: AxonColors.accent),
                    ),
                  ),
                  style: GoogleFonts.googleSans(color: AxonColors.textPrimary, fontSize: 13),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                flex: 3,
                child: TextField(
                  controller: _packageController,
                  decoration: InputDecoration(
                    hintText: 'Package name (com.example)',
                    hintStyle: GoogleFonts.googleSans(color: AxonColors.textTertiary, fontSize: 13),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
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
                      borderSide: BorderSide(color: AxonColors.accent),
                    ),
                  ),
                  style: GoogleFonts.googleSans(color: AxonColors.textPrimary, fontSize: 13),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                onPressed: _add,
                icon: Icon(Icons.add_circle, color: AxonColors.accent),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _PresetsRow extends StatelessWidget {
  final Future<void> Function(int minutes, List<String> apps) onSelect;

  const _PresetsRow({required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: _PresetCard(
            title: 'Light',
            subtitle: '30 min, Social',
            icon: Icons.self_improvement,
            color: Colors.green,
            onTap: () async {
              await onSelect(30, ['com.instagram.android', 'com.twitter.android', 'com.snapchat.android']);
            },
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _PresetCard(
            title: 'Medium',
            subtitle: '60 min, Social + Video',
            icon: Icons.balance,
            color: Colors.orange,
            onTap: () async {
              final allApps = CommonBlockedApps.all.map((AppBlockerService a) => a.packageName).toList();
              await onSelect(60, allApps);
            },
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _PresetCard(
            title: 'Intense',
            subtitle: '120 min, All',
            icon: Icons.local_fire_department,
            color: Colors.red,
            onTap: () async {
              final allApps = CommonBlockedApps.all.map((AppBlockerService a) => a.packageName).toList();
              await onSelect(120, allApps);
            },
          ),
        ),
      ],
    );
  }
}

class _PresetCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const _PresetCard({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: _FocusGlassPanel(
        padding: const EdgeInsets.all(14),
        accent: color,
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.15),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: color, size: 20),
            ),
            const SizedBox(height: 8),
            Text(
              title,
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 10,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    ).animate().shimmer(
          delay: 800.ms,
          duration: 1200.ms,
          color: color.withValues(alpha: 0.1),
        );
  }
}

class _InfoCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.electricCyan.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.electricCyan.withValues(alpha: 0.15)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AxonColors.electricCyan.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(
              Icons.info_outline_rounded,
              color: AxonColors.electricCyan,
              size: 20,
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'How Study Lock Works',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.electricCyan,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Once activated, blocked apps will display a lock screen until you reach your study target. The lock timer runs in the background.',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 11,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PermissionBanner extends ConsumerStatefulWidget {
  @override
  ConsumerState<_PermissionBanner> createState() => _PermissionBannerState();
}

class _PermissionBannerState extends ConsumerState<_PermissionBanner> {
  bool _hasOverlay = false;
  bool _hasAccessibility = false;
  bool _isChecking = true;

  @override
  void initState() {
    super.initState();
    _checkPermissions();
  }

  Future<void> _checkPermissions() async {
    final overlay = await overlayPermissionService.hasOverlayPermission();
    final accessibility = await overlayPermissionService.isAccessibilityServiceEnabled();

    if (mounted) {
      setState(() {
        _hasOverlay = overlay;
        _hasAccessibility = accessibility;
        _isChecking = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isChecking) {
      return const SizedBox.shrink();
    }

    final allGranted = _hasOverlay && _hasAccessibility;
    if (allGranted) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.warning.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.warning.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: AxonColors.warning, size: 20),
              const SizedBox(width: 8),
              Text(
                'Permissions Required',
                style: GoogleFonts.googleSans(
                  color: AxonColors.warning,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (!_hasOverlay) ...[
            _PermissionRow(
              icon: Icons.layers_outlined,
              label: 'Display over apps',
              description: 'Required to show lock screen',
              isGranted: _hasOverlay,
              onGrant: () async {
                await overlayPermissionService.openOverlaySettings();
                await Future.delayed(const Duration(seconds: 2));
                _checkPermissions();
              },
            ),
            const SizedBox(height: 8),
          ],
          if (!_hasAccessibility) ...[
            _PermissionRow(
              icon: Icons.accessibility_new_outlined,
              label: 'Accessibility service',
              description: 'Required to detect blocked apps',
              isGranted: _hasAccessibility,
              onGrant: () async {
                await overlayPermissionService.openAccessibilitySettings();
                await Future.delayed(const Duration(seconds: 2));
                _checkPermissions();
              },
            ),
          ],
          if (_hasOverlay && _hasAccessibility) ...[
            Row(
              children: [
                Icon(Icons.check_circle, color: Colors.green, size: 16),
                const SizedBox(width: 8),
                Text(
                  'All permissions granted',
                  style: GoogleFonts.googleSans(
                    color: Colors.green,
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _PermissionRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String description;
  final bool isGranted;
  final VoidCallback onGrant;

  const _PermissionRow({
    required this.icon,
    required this.label,
    required this.description,
    required this.isGranted,
    required this.onGrant,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, color: AxonColors.textSecondary, size: 18),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
              ),
              Text(
                description,
                style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 11,
                ),
              ),
            ],
          ),
        ),
        if (!isGranted)
          TextButton(
            onPressed: onGrant,
            style: TextButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              minimumSize: Size.zero,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
            child: Text(
              'Enable',
              style: GoogleFonts.googleSans(
                color: AxonColors.accent,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          )
        else
          Icon(Icons.check, color: Colors.green, size: 18),
      ],
    );
  }
}

// Import alert service
class AlertService {
  static void show({
    required BuildContext context,
    required String title,
    required String message,
    required String actionLabel,
    required VoidCallback onAction,
  }) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AxonColors.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text(title, style: GoogleFonts.googleSans(color: AxonColors.textPrimary, fontWeight: FontWeight.w700)),
        content: Text(message, style: GoogleFonts.googleSans(color: AxonColors.textSecondary, fontSize: 13)),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(ctx);
              onAction();
            },
            child: Text(actionLabel, style: GoogleFonts.googleSans(color: AxonColors.accent)),
          ),
        ],
      ),
    );
  }
}

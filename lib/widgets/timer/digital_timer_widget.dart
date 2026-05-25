// lib/widgets/timer/digital_timer_widget.dart
// High-Performance Digital Timer Widget
// Features: Progress Ring (CustomPainter), WakeLock, Preset Engine, Haptics

import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import '../../models/session_config.dart';
import '../../services/haptics_service.dart';
import '../../theme/app_theme.dart';
import '../common/rose_loader.dart';

class DigitalTimerController extends ChangeNotifier {
  Duration _duration = const Duration(minutes: 25);
  Duration _elapsed = Duration.zero;
  bool _isRunning = false;
  bool _isStopwatch = false;
  bool _isPomodoro = false;
  bool _isStrictExamMode = false;
  Timer? _ticker;

  TimerPreset? _selectedPreset;
  List<TimerPreset> _userFavorites = [];
  List<TimerPreset> _boardPresets = [];
  bool _presetsLoading = false;

  Duration get duration => _duration;
  Duration get elapsed => _elapsed;
  Duration get remaining => _duration - _elapsed;
  bool get isRunning => _isRunning;
  bool get isStopwatch => _isStopwatch;
  bool get isPomodoro => _isPomodoro;
  bool get isStrictExamMode => _isStrictExamMode;
  TimerPreset? get selectedPreset => _selectedPreset;
  List<TimerPreset> get userFavorites => _userFavorites;
  List<TimerPreset> get boardPresets => _boardPresets;
  bool get presetsLoading => _presetsLoading;

  double get progress {
    if (_isStopwatch) return 0.0;
    if (_duration.inSeconds == 0) return 0.0;
    return (_elapsed.inSeconds / _duration.inSeconds).clamp(0.0, 1.0);
  }

  String get displayTime {
    final d = _isStopwatch ? _elapsed : remaining;
    final h = d.inHours;
    final m = d.inMinutes.remainder(60);
    final s = d.inSeconds.remainder(60);

    if (h > 0) {
      return '${h.toString().padLeft(2, '0')}:${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
    }
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  Future<void> loadPresets({String? subjectId}) async {
    _presetsLoading = true;
    notifyListeners();

    await _loadBoardPresets(subjectId);
    await _loadUserFavorites();

    _presetsLoading = false;
    notifyListeners();
  }

  Future<void> _loadBoardPresets(String? subjectId) async {
    final defaults = [
      TimerPreset(
        id: 'quick_review',
        name: 'Quick Review',
        duration: const Duration(minutes: 15),
        isBoardPreset: true,
      ),
      TimerPreset(
        id: 'deep_work',
        name: 'Deep Work',
        duration: const Duration(minutes: 90),
        isBoardPreset: true,
      ),
      TimerPreset(
        id: 'past_paper_p2',
        name: 'Past Paper P2',
        duration: const Duration(minutes: 75),
        isBoardPreset: true,
      ),
      TimerPreset(
        id: 'p1_mock',
        name: 'P1 Mock',
        duration: const Duration(hours: 1, minutes: 15),
        isBoardPreset: true,
      ),
      TimerPreset(
        id: 'p4_deep_dive',
        name: 'P4 Deep Dive',
        duration: const Duration(hours: 2),
        isBoardPreset: true,
      ),
    ];

    if (subjectId != null) {
      final subjectPresets = [
        TimerPreset(
          id: '${subjectId}_p2',
          name: 'Paper 2',
          subjectId: subjectId,
          duration: const Duration(hours: 1, minutes: 15),
          isBoardPreset: true,
        ),
        TimerPreset(
          id: '${subjectId}_p4',
          name: 'Paper 4',
          subjectId: subjectId,
          duration: const Duration(hours: 2),
          isBoardPreset: true,
        ),
      ];
      _boardPresets = [...defaults, ...subjectPresets];
    } else {
      _boardPresets = defaults;
    }
  }

  Future<void> _loadUserFavorites() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString('timer_favorites');
      if (raw != null && raw.isNotEmpty) {
        final List<dynamic> list = List<dynamic>.from(
          (await jsonDecode(raw) as List).map((e) => TimerPreset.fromJson(e)),
        );
        _userFavorites = list.cast<TimerPreset>();
      }
    } catch (_) {
      _userFavorites = [];
    }
  }

  Future<void> addFavorite(TimerPreset preset) async {
    final newPreset = TimerPreset(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: preset.name,
      subjectId: preset.subjectId,
      duration: preset.duration,
      isUserFavorite: true,
      createdAt: DateTime.now(),
    );
    _userFavorites = [..._userFavorites, newPreset];
    await _saveFavorites();
    notifyListeners();
  }

  Future<void> removeFavorite(String id) async {
    _userFavorites = _userFavorites.where((p) => p.id != id).toList();
    await _saveFavorites();
    notifyListeners();
  }

  Future<void> _saveFavorites() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = jsonEncode(_userFavorites.map((p) => p.toJson()).toList());
    await prefs.setString('timer_favorites', raw);
  }

  void selectPreset(TimerPreset preset) {
    AxonHaptics.heavyImpact();
    _selectedPreset = preset;
    _duration = preset.duration;
    if (_isPomodoro) {
      _duration = _calculatePomodoroDuration(preset.duration);
    }
    notifyListeners();
  }

  void setDuration(Duration d) {
    AxonHaptics.heavyImpact();
    _duration = d;
    _selectedPreset = null;
    notifyListeners();
  }

  void setStopwatchMode(bool enabled) {
    AxonHaptics.heavyImpact();
    _isStopwatch = enabled;
    if (enabled) {
      _isPomodoro = false;
      _isStrictExamMode = false;
    }
    notifyListeners();
  }

  void setPomodoroMode(bool enabled) {
    AxonHaptics.heavyImpact();
    _isPomodoro = enabled;
    if (enabled) {
      _isStopwatch = false;
      _isStrictExamMode = false;
      _duration = _calculatePomodoroDuration(_duration);
    }
    notifyListeners();
  }

  void setStrictExamMode(bool enabled) {
    AxonHaptics.heavyImpact();
    _isStrictExamMode = enabled;
    if (enabled) {
      _isStopwatch = false;
      _isPomodoro = false;
    }
    notifyListeners();
  }

  Duration _calculatePomodoroDuration(Duration total) {
    final config = PomodoroConfig.autoCalculate(total);
    return config.focusDuration;
  }

  Future<void> start() async {
    if (_isStrictExamMode) {
      await WakelockPlus.enable();
    }
    AxonHaptics.heavyImpact();
    _isRunning = true;
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
      _elapsed = _elapsed + const Duration(seconds: 1);

      if (!_isStopwatch && _elapsed >= _duration) {
        stop();
        AxonHaptics.rewardHarmonic();
      }
      notifyListeners();
    });
    notifyListeners();
  }

  Future<void> pause() async {
    if (_isStrictExamMode) return;
    AxonHaptics.lightImpact();
    _ticker?.cancel();
    _isRunning = false;
    notifyListeners();
  }

  Future<void> stop() async {
    await WakelockPlus.disable();
    AxonHaptics.mediumImpact();
    _ticker?.cancel();
    _isRunning = false;
    notifyListeners();
  }

  void reset() {
    AxonHaptics.mediumImpact();
    _elapsed = Duration.zero;
    _isRunning = false;
    notifyListeners();
  }

  @override
  void dispose() {
    _ticker?.cancel();
    WakelockPlus.disable();
    super.dispose();
  }

  Map<String, dynamic> toJson() => {
        'duration': _duration.inSeconds,
        'isStopwatch': _isStopwatch,
        'isPomodoro': _isPomodoro,
        'isStrictExamMode': _isStrictExamMode,
      };
}

class DigitalTimerView extends ConsumerStatefulWidget {
  final String? subjectName;
  final String? chapterName;
  final DigitalTimerController? controller;
  final VoidCallback? onStart;
  final VoidCallback? onPause;
  final VoidCallback? onStop;

  const DigitalTimerView({
    super.key,
    this.subjectName,
    this.chapterName,
    this.controller,
    this.onStart,
    this.onPause,
    this.onStop,
  });

  @override
  ConsumerState<DigitalTimerView> createState() => _DigitalTimerViewState();
}

class _DigitalTimerViewState extends ConsumerState<DigitalTimerView>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  DigitalTimerController? _controller;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (widget.controller != null) {
      _controller = widget.controller;
    } else {
      _controller ??= DigitalTimerController();
      _controller!.loadPresets();
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _controller!,
      builder: (context, _) {
        final isDark = AxonThemeMode.isDark;
        final bgColor =
            isDark ? const Color(0xFF090A0B) : const Color(0xFFF5F5F5);

        return Scaffold(
          backgroundColor: bgColor,
          body: SafeArea(
            child: Column(
              children: [
                _buildHeader(),
                const Spacer(),
                _buildBigClock(),
                const Spacer(),
                _buildModeBar(),
                const SizedBox(height: 24),
                _buildPresetDrawer(),
                const SizedBox(height: 24),
                _buildActionButton(),
                const SizedBox(height: 40),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildHeader() {
    if (widget.subjectName == null) return const SizedBox.shrink();
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.school_outlined,
              color: AxonColors.textSecondary, size: 16),
          const SizedBox(width: 8),
          Text(
            '${widget.subjectName} > ${widget.chapterName ?? ""}',
            style: GoogleFonts.googleSans(
              color: AxonColors.textSecondary,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBigClock() {
    return GestureDetector(
      onTap: _showDurationPicker,
      child: Stack(
        alignment: Alignment.center,
        children: [
          SizedBox(
            width: 300,
            height: 300,
            child: CustomPaint(
              painter: _ProgressRingPainter(
                progress: _controller!.progress,
                isRunning: _controller!.isRunning,
                isStopwatch: _controller!.isStopwatch,
              ),
            ),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              AnimatedBuilder(
                animation: _pulseController,
                builder: (context, child) {
                  return Opacity(
                    opacity: _controller!.isRunning
                        ? 0.5 + (_pulseController.value * 0.5)
                        : 1.0,
                    child: child,
                  );
                },
                child: Text(
                  _controller!.displayTime,
                  style: GoogleFonts.jetBrainsMono(
                    fontSize: 80,
                    fontWeight: FontWeight.w500,
                    color: AxonColors.textPrimary,
                    letterSpacing: 4,
                  ),
                ),
              ),
              if (_controller!.isPomodoro)
                Text(
                  'Pomodoro',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.electricCyan,
                    fontSize: 12,
                  ),
                ),
              if (_controller!.isStrictExamMode)
                Container(
                  margin: const EdgeInsets.only(top: 8),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                  decoration: BoxDecoration(
                    color: AxonColors.warning.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(24),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.lock_outline,
                          color: AxonColors.warning, size: 12),
                      const SizedBox(width: 4),
                      Text(
                        'Strict Mode',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.warning,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildModeBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _ModeButton(
            label: 'Pomodoro',
            isActive: _controller!.isPomodoro,
            onTap: () => _controller!.setPomodoroMode(!_controller!.isPomodoro),
          ),
          const SizedBox(width: 12),
          _ModeButton(
            label: 'Stopwatch',
            icon: Icons.timer_outlined,
            isActive: _controller!.isStopwatch,
            onTap: () =>
                _controller!.setStopwatchMode(!_controller!.isStopwatch),
          ),
          const SizedBox(width: 12),
          _ModeButton(
            label: 'Exam',
            icon: Icons.lock_outline,
            isActive: _controller!.isStrictExamMode,
            onTap: () =>
                _controller!.setStrictExamMode(!_controller!.isStrictExamMode),
            isWarning: true,
          ),
        ],
      ),
    );
  }

  Widget _buildPresetDrawer() {
    return Container(
      height: 80,
      margin: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'Presets',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              if (_controller!.selectedPreset != null)
                GestureDetector(
                  onTap: () {
                    _controller!.addFavorite(_controller!.selectedPreset!);
                  },
                  child: Row(
                    children: [
                      Icon(Icons.favorite_border,
                          size: 14, color: AxonColors.electricCyan),
                      const SizedBox(width: 4),
                      Text(
                        'Save',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.electricCyan,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          Expanded(
            child: _controller!.presetsLoading
                ? Center(child: RoseLoader(size: 24))
                : ListView(
                    scrollDirection: Axis.horizontal,
                    children: [
                      ..._controller!.boardPresets.map((p) => _PresetChip(
                            preset: p,
                            isSelected: _controller!.selectedPreset?.id == p.id,
                            onTap: () => _controller!.selectPreset(p),
                          )),
                      if (_controller!.userFavorites.isNotEmpty) ...[
                        Container(
                          width: 1,
                          margin: const EdgeInsets.symmetric(horizontal: 8),
                          color: AxonColors.divider,
                        ),
                        ..._controller!.userFavorites.map((p) => _PresetChip(
                              preset: p,
                              isSelected:
                                  _controller!.selectedPreset?.id == p.id,
                              onTap: () => _controller!.selectPreset(p),
                              isFavorite: true,
                            )),
                      ],
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton() {
    final isRunning = _controller!.isRunning;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 40),
      child: SizedBox(
        width: double.infinity,
        height: 48,
        child: ElevatedButton(
          onPressed: isRunning
              ? () {
                  _controller!.stop();
                  widget.onStop?.call();
                }
              : () {
                  _controller!.start();
                  widget.onStart?.call();
                },
          style: ElevatedButton.styleFrom(
            backgroundColor:
                isRunning ? AxonColors.warning : AxonColors.electricCyan,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(24),
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                isRunning ? Icons.stop_rounded : Icons.play_arrow_rounded,
                size: 24,
              ),
              const SizedBox(width: 8),
              Text(
                isRunning ? 'STOP' : 'IGNITE',
                style: GoogleFonts.googleSans(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 1,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showDurationPicker() {
    AxonHaptics.heavyImpact();
    showModalBottomSheet(
      context: context,
      backgroundColor: AxonColors.surface,
      builder: (ctx) => _DurationPickerSheet(
        initialDuration: _controller!.duration,
        onDurationSelected: (d) => _controller!.setDuration(d),
      ),
    );
  }
}

class _ProgressRingPainter extends CustomPainter {
  final double progress;
  final bool isRunning;
  final bool isStopwatch;

  _ProgressRingPainter({
    required this.progress,
    required this.isRunning,
    required this.isStopwatch,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 8;
    final strokeWidth = 6.0;

    final bgPaint = Paint()
      ..color = AxonColors.divider.withValues(alpha: 0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawCircle(center, radius, bgPaint);

    if (!isStopwatch && progress > 0) {
      final progressPaint = Paint()
        ..color = isRunning ? AxonColors.electricCyan : AxonColors.accent
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      final sweepAngle = 2 * math.pi * progress;
      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        -math.pi / 2,
        sweepAngle,
        false,
        progressPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _ProgressRingPainter old) {
    return old.progress != progress ||
        old.isRunning != isRunning ||
        old.isStopwatch != isStopwatch;
  }
}

class _ModeButton extends StatelessWidget {
  final String label;
  final IconData? icon;
  final bool isActive;
  final VoidCallback onTap;
  final bool isWarning;

  const _ModeButton({
    required this.label,
    this.icon,
    required this.isActive,
    required this.onTap,
    this.isWarning = false,
  });

  @override
  Widget build(BuildContext context) {
    final activeColor =
        isWarning ? AxonColors.warning : AxonColors.electricCyan;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        decoration: BoxDecoration(
          color: isActive
              ? activeColor.withValues(alpha: 0.15)
              : AxonColors.surface,
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: isActive ? activeColor : AxonColors.divider,
            width: isActive ? 2 : 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null)
              Padding(
                padding: const EdgeInsets.only(right: 6),
                child: Icon(
                  icon,
                  size: 16,
                  color: isActive ? activeColor : AxonColors.textSecondary,
                ),
              ),
            Text(
              label,
              style: GoogleFonts.googleSans(
                color: isActive ? activeColor : AxonColors.textSecondary,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PresetChip extends StatelessWidget {
  final TimerPreset preset;
  final bool isSelected;
  final VoidCallback onTap;
  final bool isFavorite;

  const _PresetChip({
    required this.preset,
    required this.isSelected,
    required this.onTap,
    this.isFavorite = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.only(right: 8),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: isSelected
              ? AxonColors.electricCyan.withValues(alpha: 0.15)
              : AxonColors.surfaceElevated,
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: isSelected ? AxonColors.electricCyan : AxonColors.divider,
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (isFavorite)
              Padding(
                padding: const EdgeInsets.only(right: 6),
                child: Icon(
                  Icons.favorite,
                  size: 12,
                  color: AxonColors.warning,
                ),
              ),
            Text(
              preset.name,
              style: GoogleFonts.googleSans(
                color: isSelected
                    ? AxonColors.electricCyan
                    : AxonColors.textPrimary,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(width: 6),
            Text(
              preset.formattedDuration,
              style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary,
                fontSize: 11,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DurationPickerSheet extends StatefulWidget {
  final Duration initialDuration;
  final ValueChanged<Duration> onDurationSelected;

  const _DurationPickerSheet({
    required this.initialDuration,
    required this.onDurationSelected,
  });

  @override
  State<_DurationPickerSheet> createState() => _DurationPickerSheetState();
}

class _DurationPickerSheetState extends State<_DurationPickerSheet> {
  late int _hours;
  late int _minutes;

  @override
  void initState() {
    super.initState();
    _hours = widget.initialDuration.inHours;
    _minutes = widget.initialDuration.inMinutes % 60;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 320,
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          Container(
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: AxonColors.divider,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            'Set Duration',
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 24),
          Expanded(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _NumberColumn(
                  label: 'Hours',
                  value: _hours,
                  maxValue: 3,
                  onChanged: (v) {
                    AxonHaptics.selectionClick();
                    setState(() => _hours = v);
                  },
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    ':',
                    style: GoogleFonts.jetBrainsMono(
                      fontSize: 32,
                      color: AxonColors.textPrimary,
                    ),
                  ),
                ),
                _NumberColumn(
                  label: 'Minutes',
                  value: _minutes,
                  maxValue: 59,
                  onChanged: (v) {
                    AxonHaptics.selectionClick();
                    setState(() => _minutes = v);
                  },
                ),
              ],
            ),
          ),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () {
                AxonHaptics.heavyImpact();
                final duration = Duration(hours: _hours, minutes: _minutes);
                widget.onDurationSelected(duration);
                Navigator.pop(context);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: AxonColors.electricCyan,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(24),
                ),
              ),
              child: Text('Set Duration'),
            ),
          ),
        ],
      ),
    );
  }
}

class _NumberColumn extends StatelessWidget {
  final String label;
  final int value;
  final int maxValue;
  final ValueChanged<int> onChanged;

  const _NumberColumn({
    required this.label,
    required this.value,
    required this.maxValue,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: GoogleFonts.googleSans(
            color: AxonColors.textTertiary,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 8),
        Container(
          height: 120,
          width: 60,
          decoration: BoxDecoration(
            color: AxonColors.surfaceElevated,
            borderRadius: BorderRadius.circular(24),
          ),
          child: ListWheelScrollView.useDelegate(
            itemExtent: 40,
            perspective: 0.005,
            diameterRatio: 1.5,
            controller: FixedExtentScrollController(initialItem: value),
            onSelectedItemChanged: onChanged,
            childDelegate: ListWheelChildBuilderDelegate(
              childCount: maxValue + 1,
              builder: (context, index) {
                final isSelected = index == value;
                return Center(
                  child: Text(
                    index.toString().padLeft(2, '0'),
                    style: GoogleFonts.jetBrainsMono(
                      fontSize: isSelected ? 28 : 20,
                      color: isSelected
                          ? AxonColors.textPrimary
                          : AxonColors.textTertiary,
                      fontWeight:
                          isSelected ? FontWeight.w600 : FontWeight.w400,
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ],
    );
  }
}

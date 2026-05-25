import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import '../../services/exam_data_service.dart';
import 'rose_loader.dart';

class FlipClockEvolutionWidget extends StatefulWidget {
  final bool isCompact;
  final Function(String)? onSubjectTap;
  final Function(ExamEvent)? onExamTap;

  const FlipClockEvolutionWidget({
    super.key,
    this.isCompact = false,
    this.onSubjectTap,
    this.onExamTap,
  });

  @override
  State<FlipClockEvolutionWidget> createState() =>
      _FlipClockEvolutionWidgetState();
}

class _FlipClockEvolutionWidgetState extends State<FlipClockEvolutionWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _flipController;
  ExamEvent _targetExam = ExamEvent(
    board: 'Cambridge A-Level',
    subject: 'Loading...',
    component: '---',
    date: DateTime.now().add(const Duration(days: 365)),
    startTime: '09:00',
    endTime: '12:00',
  );
  Duration _timeRemaining = Duration.zero;
  Timer? _timer;
  WidgetColorMode _colorMode = WidgetColorMode.focusBlue;
  String? _manualSubject;
  bool _isDataLoaded = false;

  @override
  void initState() {
    super.initState();
    _flipController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _timer = Timer.periodic(
      const Duration(seconds: 1),
      (_) => _updateTimeRemaining(),
    );
    _loadExamData();
  }

  Future<void> _loadExamData() async {
    final service = ExamDataService();
    await service.initialize();

    final manualSubject = await service.getManualSubject();
    final colorMode = await service.getColorMode();
    final targetExam = service.getTargetExam(manualSubject);

    if (!mounted) return;

    setState(() {
      _manualSubject = manualSubject;
      _colorMode = colorMode;
      _targetExam = targetExam;
      _isDataLoaded = true;
      _updateTimeRemaining();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _flipController.dispose();
    super.dispose();
  }

  void _updateTimeRemaining() {
    if (!mounted) return;
    final service = ExamDataService();
    setState(() {
      _timeRemaining = service.getTimeRemaining(_targetExam);
    });
  }

  void _handleTap() {
    HapticFeedback.mediumImpact();
    _flipController.forward(from: 0);
    widget.onExamTap?.call(_targetExam);
  }

  void _handleLongPress() async {
    HapticFeedback.heavyImpact();

    final service = ExamDataService();
    final upcoming = service.getUpcomingExams(limit: 5);

    if (upcoming.isEmpty) return;

    final selected = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: const Color(0xFF121212),
      builder: (ctx) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Padding(
            padding: EdgeInsets.all(16),
            child: Text(
              'SELECT SUBJECT',
              style: TextStyle(
                color: Color(0xFF8B949E),
                fontSize: 12,
                letterSpacing: 2,
              ),
            ),
          ),
          ListTile(
            title: const Text(
              'Auto-Nearest',
              style: TextStyle(color: Colors.white),
            ),
            leading: const Icon(Icons.auto_awesome, color: Color(0xFF3A86FF)),
            onTap: () {
              service.setManualSubject(null);
              Navigator.pop(ctx, null);
            },
          ),
          ...upcoming.map(
            (exam) => ListTile(
              title: Text(
                exam.subject,
                style: const TextStyle(color: Colors.white),
              ),
              subtitle: Text(
                exam.component,
                style: const TextStyle(color: Color(0xFF666666)),
              ),
              onTap: () {
                service.setManualSubject(exam.component);
                Navigator.pop(ctx, exam.component);
              },
            ),
          ),
          const SizedBox(height: 16),
        ],
      ),
    );

    if (selected != null || _manualSubject != null) {
      _loadExamData();
      widget.onSubjectTap?.call(selected ?? '');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_isDataLoaded) {
      return _buildLoadingWidget();
    }

    if (_targetExam.subject == 'No Exams Found') {
      return _buildNoExamsWidget();
    }

    final isUrgent = _timeRemaining.inHours < 24;
    final accentColor = _colorMode == WidgetColorMode.neonRed && isUrgent
        ? const Color(0xFFFF4444)
        : const Color(0xFF3A86FF);

    final days = _timeRemaining.inDays;
    final hours = _timeRemaining.inHours.remainder(24);
    final minutes = _timeRemaining.inMinutes.remainder(60);
    final seconds = _timeRemaining.inSeconds.remainder(60);

    if (widget.isCompact) {
      return _buildCompactLayout(days, hours, minutes, seconds, accentColor);
    }

    return GestureDetector(
      onTap: _handleTap,
      onLongPress: _handleLongPress,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFF0A0A0A),
          borderRadius: BorderRadius.circular(24),
          border:
              Border.all(color: accentColor.withValues(alpha: 0.3), width: 1),
          boxShadow: [
            BoxShadow(
              color: accentColor.withValues(alpha: 0.1),
              blurRadius: 20,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(accentColor),
            const SizedBox(height: 12),
            _buildFlipDisplay(days, hours, minutes, seconds, accentColor),
            const SizedBox(height: 12),
            _buildFooter(accentColor),
          ],
        ),
      ),
    );
  }

  Widget _buildLoadingWidget() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0A0A0A),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2A2A2A)),
      ),
      child: const Center(
        child: RoseLoader(size: 24, color: Color(0xFF3A86FF)),
      ),
    );
  }

  Widget _buildNoExamsWidget() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF0A0A0A),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2A2A2A)),
      ),
      child: const Column(
        children: [
          Icon(Icons.event_busy, color: Color(0xFF666666), size: 48),
          SizedBox(height: 8),
          Text(
            'NO EXAMS SCHEDULED',
            style: TextStyle(
              color: Color(0xFF666666),
              fontSize: 12,
              letterSpacing: 2,
            ),
          ),
          SizedBox(height: 4),
          Text(
            'Check datesheet cache',
            style: TextStyle(color: Color(0xFF444444), fontSize: 10),
          ),
        ],
      ),
    );
  }

  Widget _buildHeader(Color accentColor) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
          decoration: BoxDecoration(
            color: accentColor.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            _targetExam.component,
            style: TextStyle(
              color: accentColor,
              fontSize: 10,
              fontWeight: FontWeight.w600,
              letterSpacing: 1,
            ),
          ),
        ),
        const Spacer(),
        Container(
          width: 6,
          height: 6,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: accentColor,
            boxShadow: [
              BoxShadow(
                color: accentColor.withValues(alpha: 0.5),
                blurRadius: 6,
                spreadRadius: 1,
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildFlipDisplay(
    int days,
    int hours,
    int minutes,
    int seconds,
    Color accentColor,
  ) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _FlipUnit(
          value: _twoDigits(days),
          label: 'DAYS',
          accentColor: accentColor,
        ),
        const SizedBox(width: 8),
        _ColonSeparator(),
        const SizedBox(width: 8),
        _FlipUnit(
          value: _twoDigits(hours),
          label: 'HRS',
          accentColor: accentColor,
        ),
        const SizedBox(width: 8),
        _ColonSeparator(),
        const SizedBox(width: 8),
        _FlipUnit(
          value: _twoDigits(minutes),
          label: 'MIN',
          accentColor: accentColor,
        ),
        const SizedBox(width: 8),
        _ColonSeparator(),
        const SizedBox(width: 8),
        _FlipUnit(
          value: _twoDigits(seconds),
          label: 'SEC',
          accentColor: accentColor,
        ),
      ],
    );
  }

  Widget _buildFooter(Color accentColor) {
    final modeText = _manualSubject != null ? 'MANUAL' : 'AUTO-MODE';
    return Row(
      children: [
        Icon(Icons.schedule,
            size: 10, color: accentColor.withValues(alpha: 0.6)),
        const SizedBox(width: 4),
        Expanded(
          child: Text(
            _targetExam.subject,
            style: TextStyle(
              color: accentColor.withValues(alpha: 0.8),
              fontSize: 9,
              fontWeight: FontWeight.w500,
              letterSpacing: 1,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ),
        Text(
          modeText,
          style: const TextStyle(
            color: Color(0xFF444444),
            fontSize: 7,
            letterSpacing: 1,
          ),
        ),
      ],
    );
  }

  Widget _buildCompactLayout(
    int days,
    int hours,
    int minutes,
    int seconds,
    Color accentColor,
  ) {
    return GestureDetector(
      onTap: _handleTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xFF0A0A0A),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: accentColor.withValues(alpha: 0.3)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              _targetExam.component,
              style: TextStyle(
                color: accentColor,
                fontSize: 10,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${_twoDigits(days)}:${_twoDigits(hours)}:${_twoDigits(minutes)}',
              style: const TextStyle(
                fontFamily: 'monospace',
                color: Color(0xFFFFFFFF),
                fontSize: 16,
                fontWeight: FontWeight.w700,
                letterSpacing: 2,
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _twoDigits(int n) => n.toString().padLeft(2, '0');
}

class _FlipUnit extends StatelessWidget {
  final String value;
  final String label;
  final Color accentColor;

  const _FlipUnit({
    required this.value,
    required this.label,
    required this.accentColor,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          width: 44,
          height: 52,
          child: Stack(
            children: [
              Positioned(
                top: 0,
                left: 0,
                right: 0,
                height: 24,
                child: _FlapTopHalf(accentColor: accentColor, value: value),
              ),
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                height: 24,
                child: _FlapBottomHalf(accentColor: accentColor, value: value),
              ),
              _FoldLine(),
            ],
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: const TextStyle(
            color: Color(0xFF666666),
            fontSize: 7,
            letterSpacing: 1,
          ),
        ),
      ],
    );
  }
}

class _FlapTopHalf extends StatelessWidget {
  final Color accentColor;
  final String value;

  const _FlapTopHalf({required this.accentColor, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A1A),
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(6),
          topRight: Radius.circular(6),
        ),
        border: Border.all(color: const Color(0xFF2A2A2A)),
      ),
      child: ClipRRect(
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(5),
          topRight: Radius.circular(5),
        ),
        child: CustomPaint(
          painter: _DotMatrixPainter(),
          child: Center(
            child: _StencilDigit(
              digit: value.isNotEmpty ? value[0] : '0',
              fontSize: 28,
              color: accentColor,
            ),
          ),
        ),
      ),
    );
  }
}

class _FlapBottomHalf extends StatelessWidget {
  final Color accentColor;
  final String value;

  const _FlapBottomHalf({required this.accentColor, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF0A0A0A),
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(6),
          bottomRight: Radius.circular(6),
        ),
        border: Border.all(color: const Color(0xFF2A2A2A)),
      ),
      child: Center(
        child: _StencilDigit(
          digit: value.length > 1 ? value[1] : '0',
          fontSize: 28,
          color: accentColor,
        ),
      ),
    );
  }
}

class _FoldLine extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: 0,
      right: 0,
      top: 24,
      child: Container(
        height: 4,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF0A0A0A), Color(0xFF333333), Color(0xFF0A0A0A)],
            stops: [0.0, 0.5, 1.0],
          ),
        ),
      ),
    );
  }
}

class _ColonSeparator extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 4,
          height: 4,
          decoration: const BoxDecoration(
            color: Color(0xFF444444),
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(height: 24),
        Container(
          width: 4,
          height: 4,
          decoration: const BoxDecoration(
            color: Color(0xFF444444),
            shape: BoxShape.circle,
          ),
        ),
      ],
    );
  }
}

class _DotMatrixPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF2A2A2A)
      ..style = PaintingStyle.fill;

    const dotSize = 1.5;
    final spacingX = size.width / 8;
    final spacingY = size.height / 6;

    for (var row = 0; row < 4; row++) {
      for (var col = 0; col < 6; col++) {
        final x = col * spacingX + spacingX / 2;
        final y = row * spacingY + spacingY / 2;

        final distanceFromFold = (y / size.height).clamp(0.0, 1.0);
        final opacity = (1 - distanceFromFold) * 0.5;

        paint.color = const Color(
          0xFF2A2A2A,
        ).withValues(alpha: opacity.clamp(0.0, 0.5));
        canvas.drawCircle(Offset(x, y), dotSize / 2, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _StencilDigit extends StatelessWidget {
  final String digit;
  final double fontSize;
  final Color color;

  const _StencilDigit({
    required this.digit,
    required this.fontSize,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: fontSize * 0.75,
      child: Stack(
        children: [
          Text(
            digit,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: fontSize,
              fontWeight: FontWeight.w700,
              color: const Color(0xFFFFFFFF),
              height: 1,
            ),
          ),
          if (digit == '0')
            Positioned.fill(
              child: Center(
                child: Transform.rotate(
                  angle: 0.785398,
                  child: Container(
                    width: fontSize * 0.6,
                    height: fontSize * 0.06,
                    color: const Color(0xFF0A0A0A),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class FocusBlueFlipClock extends StatelessWidget {
  final String subjectCode;
  final Duration timeRemaining;
  final VoidCallback? onTap;

  const FocusBlueFlipClock({
    super.key,
    required this.subjectCode,
    required this.timeRemaining,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final days = timeRemaining.inDays;
    final hours = timeRemaining.inHours.remainder(24);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF0A0A0A),
          borderRadius: BorderRadius.circular(16),
          border:
              Border.all(color: const Color(0xFF3A86FF).withValues(alpha: 0.3)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              decoration: BoxDecoration(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(3),
              ),
              child: Text(
                subjectCode,
                style: const TextStyle(
                  color: Color(0xFF3A86FF),
                  fontSize: 8,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              '${days}d ${hours}h',
              style: const TextStyle(
                fontFamily: 'monospace',
                fontSize: 14,
                fontWeight: FontWeight.w700,
                color: Color(0xFFFFFFFF),
                letterSpacing: 1,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

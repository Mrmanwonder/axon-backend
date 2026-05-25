import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import '../../services/sync_service.dart';

class IndustrialTimerWidget extends StatefulWidget {
  final bool showControls;
  final VoidCallback? onPlay;
  final VoidCallback? onPause;
  final double? width;
  final double? height;

  const IndustrialTimerWidget({
    super.key,
    this.showControls = true,
    this.onPlay,
    this.onPause,
    this.width,
    this.height,
  });

  @override
  State<IndustrialTimerWidget> createState() => _IndustrialTimerWidgetState();
}

class _IndustrialTimerWidgetState extends State<IndustrialTimerWidget> {
  int _durationSeconds = 0;
  bool _isRunning = false;
  Timer? _timer;
  String _currentSubject = '';
  String _currentChapter = '';
  final _syncService = SyncService();

  @override
  void initState() {
    super.initState();
    _loadActiveSession();
  }

  Future<void> _loadActiveSession() async {
    try {
      final session = await _syncService.getActiveSession();
      if (session != null) {
        final elapsed = session.elapsed is int ? session.elapsed as int : 0;
        setState(() {
          _durationSeconds = elapsed;
          _currentSubject = session.subject;
          _currentChapter = session.chapter;
          _isRunning = true;
        });
        _startTimer();
      }
    } catch (e) {
      debugPrint('Error loading active session: $e');
    }
  }

  void _startTimer() {
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          _durationSeconds++;
        });
      }
    });
  }

  void _stopTimer() {
    _timer?.cancel();
    setState(() {
      _isRunning = false;
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final containerWidth =
        widget.width ?? (screenWidth - 32).clamp(280.0, 400.0);
    final containerHeight = widget.height ?? 80.0;

    final hours = _durationSeconds ~/ 3600;
    final minutes = (_durationSeconds % 3600) ~/ 60;
    final seconds = _durationSeconds % 60;

    final timeString = hours > 0
        ? '${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}'
        : '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';

    final leftSectionWidth =
        widget.showControls ? containerWidth * 0.75 : containerWidth;
    final rightSectionWidth = widget.showControls ? containerWidth * 0.25 : 0.0;

    return LayoutBuilder(
      builder: (context, constraints) {
        final fontSize = (containerHeight * 0.45).clamp(24.0, 40.0);
        final iconSize = (containerHeight * 0.25).clamp(16.0, 24.0);
        final buttonSize = (containerHeight * 0.45).clamp(36.0, 48.0);

        return GestureDetector(
          onTap: () => HapticFeedback.selectionClick(),
          child: Container(
            width: containerWidth,
            height: containerHeight,
            decoration: BoxDecoration(
              color: const Color(0xFF121212),
              borderRadius: BorderRadius.circular(containerHeight * 0.3),
              border: Border.all(color: const Color(0xFF2A2A2A), width: 1),
            ),
            child: Row(
              children: [
                SizedBox(
                  width: leftSectionWidth,
                  height: containerHeight,
                  child: _DotMatrixBackground(
                    child: Stack(
                      children: [
                        Center(
                          child: _StencilTimerText(
                              timeString: timeString, fontSize: fontSize),
                        ),
                        if (_currentSubject.isNotEmpty)
                          Positioned(
                            bottom: 4,
                            left: 0,
                            right: 0,
                            child: Text(
                              '$_currentSubject • $_currentChapter',
                              style: const TextStyle(
                                  color: Color(0xFF666666), fontSize: 8),
                              textAlign: TextAlign.center,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
                if (widget.showControls)
                  Container(
                    width: rightSectionWidth,
                    height: containerHeight,
                    decoration: BoxDecoration(
                      color: const Color(0xFF0A0A0A),
                      borderRadius: BorderRadius.only(
                        topRight: Radius.circular(containerHeight * 0.3),
                        bottomRight: Radius.circular(containerHeight * 0.3),
                      ),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _IndustrialButton(
                          icon: Icons.pause_rounded,
                          isActive: _isRunning,
                          size: buttonSize,
                          iconSize: iconSize,
                          onTap: () {
                            HapticFeedback.lightImpact();
                            _stopTimer();
                            widget.onPause?.call();
                          },
                        ),
                        SizedBox(height: containerHeight * 0.15),
                        _IndustrialButton(
                          icon: Icons.play_arrow_rounded,
                          isActive: !_isRunning,
                          size: buttonSize,
                          iconSize: iconSize,
                          onTap: () {
                            HapticFeedback.lightImpact();
                            _startTimer();
                            widget.onPlay?.call();
                          },
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _DotMatrixBackground extends StatelessWidget {
  final Widget child;
  const _DotMatrixBackground({required this.child});

  @override
  Widget build(BuildContext context) {
    return ShaderMask(
      shaderCallback: (rect) => const LinearGradient(
        begin: Alignment.centerLeft,
        end: Alignment.center,
        colors: [Color(0xFF3A3A3A), Color(0xFF3A3A3A), Colors.transparent],
        stops: [0.0, 0.4, 1.0],
      ).createShader(rect),
      blendMode: BlendMode.srcIn,
      child: CustomPaint(painter: _DotMatrixPainter(), child: child),
    );
  }
}

class _DotMatrixPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF2A2A2A)
      ..style = PaintingStyle.fill;
    const dotSize = 2.5;
    final spacing = size.width / 15;
    final cols = (size.width / spacing).ceil();
    final rows = (size.height / (size.height / 8)).ceil().clamp(4, 8);
    for (var row = 0; row < rows; row++) {
      for (var col = 0; col < cols; col++) {
        final x = col * spacing + spacing / 2;
        final y = row * (size.height / rows) + (size.height / rows) / 2;
        canvas.drawCircle(Offset(x, y), dotSize / 2, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _StencilTimerText extends StatelessWidget {
  final String timeString;
  final double fontSize;
  const _StencilTimerText({required this.timeString, this.fontSize = 32});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: timeString.split('').map((char) {
        if (char == ':') {
          return Padding(
            padding: EdgeInsets.symmetric(horizontal: fontSize * 0.08),
            child: Text(':',
                style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: fontSize,
                    fontWeight: FontWeight.w700,
                    color: const Color(0xFF666666))),
          );
        }
        return _StencilDigit(digit: char, fontSize: fontSize);
      }).toList(),
    );
  }
}

class _StencilDigit extends StatelessWidget {
  final String digit;
  final double fontSize;
  const _StencilDigit({required this.digit, this.fontSize = 32});

  @override
  Widget build(BuildContext context) {
    final digitWidth = fontSize * 0.75;
    return SizedBox(
      width: digitWidth,
      child: Stack(
        children: [
          Text(digit,
              style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: fontSize,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFFFFFFFF))),
          Positioned.fill(
            child: Center(
              child: Container(
                  width: digitWidth * 0.9,
                  height: fontSize * 0.06,
                  color: const Color(0xFF121212),
                  transform: Matrix4.rotationZ(0.785398)),
            ),
          ),
        ],
      ),
    );
  }
}

class _IndustrialButton extends StatelessWidget {
  final IconData icon;
  final bool isActive;
  final double size;
  final double iconSize;
  final VoidCallback? onTap;
  const _IndustrialButton(
      {required this.icon,
      this.isActive = false,
      this.size = 40,
      this.iconSize = 18,
      this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: isActive ? const Color(0xFF1A1A1A) : const Color(0xFF0F0F0F),
          shape: BoxShape.circle,
          border: Border.all(color: const Color(0xFF2A2A2A), width: 1),
          boxShadow: [
            BoxShadow(
                color: Colors.black.withValues(alpha: 0.3),
                blurRadius: 4,
                offset: const Offset(0, 2))
          ],
        ),
        child: Icon(icon,
            color: isActive ? const Color(0xFFFFFFFF) : const Color(0xFF666666),
            size: iconSize),
      ),
    );
  }
}

class CompactIndustrialTimer extends StatelessWidget {
  final int durationSeconds;
  final bool isRunning;
  const CompactIndustrialTimer(
      {super.key, required this.durationSeconds, this.isRunning = false});

  @override
  Widget build(BuildContext context) {
    final minutes = (durationSeconds ~/ 60).remainder(60);
    final seconds = durationSeconds % 60;
    final timeString =
        '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';

    return LayoutBuilder(
      builder: (context, constraints) {
        final containerWidth = constraints.maxWidth.clamp(100.0, 200.0);
        final fontSize = (containerWidth * 0.12).clamp(14.0, 22.0);
        return Container(
          padding: EdgeInsets.symmetric(
              horizontal: containerWidth * 0.12,
              vertical: containerWidth * 0.08),
          decoration: BoxDecoration(
            color: const Color(0xFF121212),
            borderRadius: BorderRadius.circular(containerWidth * 0.15),
            border: Border.all(color: const Color(0xFF2A2A2A)),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: containerWidth * 0.08,
                height: containerWidth * 0.08,
                decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: isRunning
                        ? const Color(0xFF39D353)
                        : const Color(0xFF666666)),
              ),
              SizedBox(width: containerWidth * 0.1),
              Text(timeString,
                  style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: fontSize,
                      fontWeight: FontWeight.w600,
                      color: const Color(0xFFFFFFFF),
                      letterSpacing: 2)),
            ],
          ),
        );
      },
    );
  }
}

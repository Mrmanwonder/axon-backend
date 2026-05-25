import 'dart:math';
import 'package:flutter/material.dart';
import '../../theme/app_theme.dart';

class SiriWaveform extends StatefulWidget {
  final bool isListening;

  const SiriWaveform({
    super.key,
    required this.isListening,
  });

  @override
  State<SiriWaveform> createState() => _SiriWaveformState();
}

class _SiriWaveformState extends State<SiriWaveform> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );
    if (widget.isListening) {
      _controller.repeat();
    }
  }

  @override
  void didUpdateWidget(SiriWaveform oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isListening && !oldWidget.isListening) {
      _controller.repeat();
    } else if (!widget.isListening && oldWidget.isListening) {
      _controller.stop();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
      height: widget.isListening ? 40 : 0,
      width: double.infinity,
      child: widget.isListening
          ? AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                return CustomPaint(
                  painter: _SiriWavePainter(
                    time: _controller.value * 2 * pi,
                  ),
                );
              },
            )
          : const SizedBox(),
    );
  }
}

class _SiriWavePainter extends CustomPainter {
  final double time;

  _SiriWavePainter({required this.time});

  @override
  void paint(Canvas canvas, Size size) {
    // We will draw 4 overlapping sine waves with different parameters
    final colors = [
      AxonColors.electricCyan.withValues(alpha: 0.6),
      AxonColors.accentPurple.withValues(alpha: 0.7),
      AxonColors.vibrantBlue.withValues(alpha: 0.8),
      Colors.white.withValues(alpha: 0.5),
    ];

    final baseAmplitudes = [0.4, 0.6, 0.8, 1.0];
    final speeds = [1.0, 1.5, 0.8, 1.2];
    final phases = [0.0, 1.0, 2.0, 3.0];
    
    // Smooth random amplitude multiplier to simulate speaking organically
    final randomPulse = 0.6 + 0.4 * sin(time * 3) * cos(time * 2);

    final midY = size.height / 2;

    for (int i = 0; i < 4; i++) {
      final path = Path();
      final paint = Paint()
        ..color = colors[i]
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0 + (i * 0.5)
        ..strokeCap = StrokeCap.round;
      
      if (i == 2) {
        // give one of them a glow
        paint.maskFilter = const MaskFilter.blur(BlurStyle.normal, 4);
      }

      bool first = true;
      for (double x = 0; x <= size.width; x += 3) {
        // normalize x to -1..1
        final nx = (x / size.width) * 2 - 1;
        
        // Attenuate at the edges so it fades out to 0 at the ends
        final attenuation = pow(1.0 - pow(nx, 2), 2).toDouble();

        // compute the wave
        final phaseOffset = time * speeds[i] + phases[i];
        final frequency = 3.0 + i; 
        
        final yOffset = sin(nx * frequency * pi + phaseOffset) * 
                        (size.height / 2) * 
                        baseAmplitudes[i] * 
                        randomPulse * 
                        attenuation;

        if (first) {
          path.moveTo(x, midY + yOffset);
          first = false;
        } else {
          path.lineTo(x, midY + yOffset);
        }
      }
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _SiriWavePainter oldDelegate) {
    return oldDelegate.time != time;
  }
}

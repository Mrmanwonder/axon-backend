import 'package:flutter/material.dart';
import 'package:flutter_math_fork/flutter_math.dart';

class MathExpression extends StatelessWidget {
  final String formulaTex;
  final Color tintColor;
  final double fontSize;

  const MathExpression({
    super.key,
    required this.formulaTex,
    this.tintColor = Colors.white,
    this.fontSize = 18,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.02),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Center(
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Math.tex(
            formulaTex,
            textStyle: TextStyle(
              color: tintColor,
              fontSize: fontSize,
            ),
            onErrorFallback: (err) => Text(
              formulaTex,
              style: TextStyle(
                color: tintColor.withValues(alpha: 0.5),
                fontSize: fontSize,
                fontFamily: 'monospace',
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class InlineMathExpression extends StatelessWidget {
  final String formulaTex;
  final Color tintColor;
  final double fontSize;

  const InlineMathExpression({
    super.key,
    required this.formulaTex,
    this.tintColor = Colors.white,
    this.fontSize = 16,
  });

  @override
  Widget build(BuildContext context) {
    return Math.tex(
      formulaTex,
      textStyle: TextStyle(
        color: tintColor,
        fontSize: fontSize,
      ),
      onErrorFallback: (err) => Text(
        formulaTex,
        style: TextStyle(
          color: tintColor.withValues(alpha: 0.5),
          fontSize: fontSize,
          fontFamily: 'monospace',
        ),
      ),
    );
  }
}
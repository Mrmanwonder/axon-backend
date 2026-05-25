import 'package:flutter/material.dart';
import 'package:flutter_math_fork/flutter_math.dart';

class MathMessage extends StatelessWidget {
  final String text;
  final double fontSize;

  const MathMessage({
    super.key,
    required this.text,
    this.fontSize = 15,
  });

  @override
  Widget build(BuildContext context) {
    final spans = _parse(text);
    return RichText(
      text: TextSpan(children: spans),
    );
  }

  List<InlineSpan> _parse(String src) {
    final spans = <InlineSpan>[];
    int i = 0;

    while (i < src.length) {
      // Display math $$...$$
      if (i + 1 < src.length && src[i] == r'$' && src[i + 1] == r'$') {
        final close = src.indexOf(r'$$', i + 2);
        if (close != -1) {
          final formula = src.substring(i + 2, close).trim();
          spans.add(WidgetSpan(
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.symmetric(vertical: 8),
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.03),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
              ),
              child: Center(
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Math.tex(
                    formula,
                    textStyle: TextStyle(color: Colors.white, fontSize: fontSize + 2),
                    onErrorFallback: (_) => Text(formula,
                      style: const TextStyle(color: Colors.white54, fontFamily: 'monospace', fontSize: 14)),
                  ),
                ),
              ),
            ),
          ));
          i = close + 2;
          continue;
        }
      }

      // Inline math $...$
      if (src[i] == r'$' && i + 1 < src.length && src[i + 1] != r'$') {
        final close = src.indexOf(r'$', i + 1);
        if (close != -1) {
          final formula = src.substring(i + 1, close).trim();
          spans.add(WidgetSpan(
            alignment: PlaceholderAlignment.middle,
            child: Math.tex(
              formula,
              textStyle: const TextStyle(color: Color(0xFF3A86FF), fontSize: 15),
              onErrorFallback: (_) => Text(formula,
                style: const TextStyle(color: Color(0xFF3A86FF), fontFamily: 'monospace', fontSize: 14)),
            ),
          ));
          i = close + 1;
          continue;
        }
      }

      // Collect plain text up to next $ or \n
      final next = src.indexOf(r'$', i);
      final chunk = next == -1 ? src.substring(i) : src.substring(i, next);
      if (chunk.isNotEmpty) {
        spans.add(TextSpan(
          text: chunk,
          style: TextStyle(color: Colors.white.withValues(alpha: 0.88), fontSize: fontSize, height: 1.55),
        ));
      }
      i = next == -1 ? src.length : next;
    }

    return spans;
  }
}

import 'package:flutter/material.dart';
import 'package:flutter_math_fork/flutter_math.dart';

class MathMessage extends StatefulWidget {
  final String text;
  final double fontSize;

  const MathMessage({
    super.key,
    required this.text,
    this.fontSize = 15,
  });

  @override
  State<MathMessage> createState() => _MathMessageState();
}

class _MathMessageState extends State<MathMessage> {
  late String _cachedText;
  late List<InlineSpan> _cachedSpans;

  @override
  void initState() {
    super.initState();
    _cachedText = widget.text;
    _cachedSpans = _parse(widget.text);
  }

  @override
  void didUpdateWidget(covariant MathMessage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.text != _cachedText) {
      _cachedText = widget.text;
      _cachedSpans = _parse(widget.text);
    }
  }

  @override
  Widget build(BuildContext context) {
    return RichText(
      text: TextSpan(children: _cachedSpans),
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
                    textStyle: TextStyle(color: Colors.white, fontSize: widget.fontSize + 2),
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
          style: TextStyle(color: Colors.white.withValues(alpha: 0.88), fontSize: widget.fontSize, height: 1.55),
        ));
      }
      i = next == -1 ? src.length : next;
    }

    return spans;
  }
}

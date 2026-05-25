import '../models/models.dart';

class MarkSchemeSegmenter {
  const MarkSchemeSegmenter();

  String segment({
    required String markSchemeText,
    required PdfQuestion question,
  }) {
    final normalized = _normalize(markSchemeText);
    if (normalized.isEmpty) {
      return '';
    }

    final block = _extractQuestionBlock(
      normalized: normalized,
      questionNumber: question.questionNumber,
    );
    if (block.isEmpty) {
      return '';
    }

    if (question.parts.isEmpty) {
      return block;
    }

    final partAware = _extractRelevantParts(
      block: block,
      partLabels: question.parts.map((part) => part.label).toList(),
    );
    return partAware.isEmpty ? block : partAware;
  }

  String _normalize(String input) {
    return input
        .replaceAll('\r\n', '\n')
        .replaceAll('\r', '\n')
        .replaceAll(RegExp(r'[ \t]+'), ' ')
        .replaceAll(RegExp(r'\n{3,}'), '\n\n')
        .trim();
  }

  String _extractQuestionBlock({
    required String normalized,
    required int questionNumber,
  }) {
    final lines = normalized.split('\n');
    final anchors = <_QuestionAnchor>[];

    for (var i = 0; i < lines.length; i++) {
      final number = _matchQuestionHeader(lines[i]);
      if (number != null) {
        anchors.add(_QuestionAnchor(lineIndex: i, questionNumber: number));
      }
    }

    if (anchors.isNotEmpty) {
      for (var i = 0; i < anchors.length; i++) {
        final anchor = anchors[i];
        if (anchor.questionNumber != questionNumber) {
          continue;
        }
        final nextLine = i + 1 < anchors.length ? anchors[i + 1].lineIndex : lines.length;
        return lines.sublist(anchor.lineIndex, nextLine).join('\n').trim();
      }
    }

    final inlineMatch = RegExp(
      '(?:^|\\n)\\s*(?:Q(?:uestion)?\\s*)?$questionNumber(?:[\\.\\):\\-]|\\s)',
      caseSensitive: false,
    ).firstMatch(normalized);
    if (inlineMatch == null) {
      return '';
    }

    final start = inlineMatch.start;
    final remainder = normalized.substring(start);
    final nextMatch = RegExp(
      '\\n\\s*(?:Q(?:uestion)?\\s*)?${questionNumber + 1}(?:[\\.\\):\\-]|\\s)',
      caseSensitive: false,
    ).firstMatch(remainder);
    final end = nextMatch == null ? remainder.length : nextMatch.start;
    return remainder.substring(0, end).trim();
  }

  int? _matchQuestionHeader(String line) {
    final match = RegExp(
      r'^\s*(?:question\s*)?(\d{1,2})(?:[\.\):\-]\s*|\s+\[\d+|\s{2,}|\s+[A-Z])',
      caseSensitive: false,
    ).firstMatch(line);
    if (match == null) {
      return null;
    }
    return int.tryParse(match.group(1) ?? '');
  }

  String _extractRelevantParts({
    required String block,
    required List<String> partLabels,
  }) {
    final normalizedLabels = partLabels
        .map((label) => label.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), ''))
        .where((label) => label.isNotEmpty)
        .toList();
    if (normalizedLabels.isEmpty) {
      return block;
    }

    final lines = block.split('\n');
    final partStarts = <_PartAnchor>[];
    for (var i = 0; i < lines.length; i++) {
      final lineLabel = _matchPartHeader(lines[i]);
      if (lineLabel != null) {
        partStarts.add(_PartAnchor(lineIndex: i, label: lineLabel));
      }
    }
    if (partStarts.isEmpty) {
      return block;
    }

    final collected = <String>[];
    final headerEnd = partStarts.first.lineIndex.clamp(0, lines.length);
    if (headerEnd > 0) {
      collected.add(lines.sublist(0, headerEnd).join('\n').trim());
    }

    for (final requested in normalizedLabels) {
      for (var i = 0; i < partStarts.length; i++) {
        final anchor = partStarts[i];
        if (anchor.label != requested) {
          continue;
        }
        final nextLine = i + 1 < partStarts.length ? partStarts[i + 1].lineIndex : lines.length;
        collected.add(lines.sublist(anchor.lineIndex, nextLine).join('\n').trim());
        break;
      }
    }

    return collected.where((segment) => segment.trim().isNotEmpty).join('\n').trim();
  }

  String? _matchPartHeader(String line) {
    final match = RegExp(
      r'^\s*\(?([a-z])\)?(?:[\.\):\-]\s*|\s{2,}|\s+)',
      caseSensitive: false,
    ).firstMatch(line);
    if (match == null) {
      return null;
    }
    return (match.group(1) ?? '').toLowerCase();
  }
}

class _QuestionAnchor {
  final int lineIndex;
  final int questionNumber;

  const _QuestionAnchor({
    required this.lineIndex,
    required this.questionNumber,
  });
}

class _PartAnchor {
  final int lineIndex;
  final String label;

  const _PartAnchor({
    required this.lineIndex,
    required this.label,
  });
}

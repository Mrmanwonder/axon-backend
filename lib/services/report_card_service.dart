import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'app_state.dart';

class ReportCardService {

  Future<void> importReportCardAndUpdateScore(BuildContext context, WidgetRef ref) async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf', 'jpg', 'jpeg', 'png'],
      );

      if (result == null || result.files.single.path == null) return;

      final path = result.files.single.path!;
      final isPdf = path.toLowerCase().endsWith('.pdf');

      String text = '';
      if (isPdf) {
        // Would use native PDF extraction here
        text = '';
      } else {
        if (context.mounted) {
          showDialog(
            context: context,
            builder: (ctx) => AlertDialog(
              title: const Text('Scanning Not Available'),
              content: const Text('Report card scanning is not yet available. Please enter your scores manually.'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(),
                  child: const Text('OK'),
                ),
              ],
            ),
          );
        }
        return;
      }

      final score = _parseScoreFromText(text);
      if (score != null) {
        ref.read(metricsProvider.notifier).updateMockScore(score);
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Report card imported. Mock score updated to ${score.toStringAsFixed(1)}%')),
          );
        }
      } else {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not find a valid score in the document.')),
          );
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error importing report card: $e')),
        );
      }
    }
  }

  double? _parseScoreFromText(String text) {
    // 1. Look for "Total Average: XX%" or "Average: XX%"
    final avgRegex = RegExp(r'(?:average|total|aggregate)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*%', caseSensitive: false);
    final avgMatch = avgRegex.firstMatch(text);
    if (avgMatch != null) {
      return double.tryParse(avgMatch.group(1)!);
    }

    // 2. Look for percentages and average them
    final pctRegex = RegExp(r'(\d+(?:\.\d+)?)\s*%');
    final pctMatches = pctRegex.allMatches(text);
    if (pctMatches.isNotEmpty) {
      double sum = 0;
      for (final m in pctMatches) {
        sum += double.parse(m.group(1)!);
      }
      return sum / pctMatches.length;
    }

    // 3. Look for X/Y scores
    final fractionRegex = RegExp(r'(\d+)\s*/\s*(\d+)');
    final fractionMatches = fractionRegex.allMatches(text);
    if (fractionMatches.isNotEmpty) {
      double sumPct = 0;
      for (final m in fractionMatches) {
        final obtained = double.parse(m.group(1)!);
        final total = double.parse(m.group(2)!);
        if (total > 0) {
          sumPct += (obtained / total) * 100;
        }
      }
      return sumPct / fractionMatches.length;
    }

    return null;
  }
}

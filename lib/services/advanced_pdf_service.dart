import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:path_provider/path_provider.dart';

class RenderedPdfPage {
  final int pageNumber;
  final String filePath;
  final Uint8List bytes;

  const RenderedPdfPage({
    required this.pageNumber,
    required this.filePath,
    required this.bytes,
  });
}

class AdvancedPdfService {
  static final AdvancedPdfService _instance = AdvancedPdfService._internal();
  factory AdvancedPdfService() => _instance;
  AdvancedPdfService._internal();

  Future<Map<int, List<String>>> detectDiagrams(
    String filePath, {
    Function(double progress)? onProgress,
  }) async {
    final result = <int, List<String>>{};

    if (kIsWeb || Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
      debugPrint(
          'AdvancedPdfService.detectDiagrams is unavailable on desktop/web.');
      return result;
    }

    try {
      final imageLabeler = ImageLabeler(
        options: ImageLabelerOptions(confidenceThreshold: 0.5),
      );
      imageLabeler.close();
    } catch (e) {
      debugPrint('Diagram detection failed: $e');
    }

    return result;
  }

  Future<List<RenderedPdfPage>> renderPagesForAnalysis(
    String filePath, {
    int maxPages = 12,
    int width = 1600,
    int height = 2200,
  }) async {
    final renderedPages = <RenderedPdfPage>[];

    if (kIsWeb || Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
      debugPrint(
          'AdvancedPdfService.renderPagesForAnalysis is unavailable on desktop/web.');
      return renderedPages;
    }

    final tempDir = await getTemporaryDirectory();
    final renderDir = Directory(
      '${tempDir.path}/axon_pdf_render/${filePath.hashCode}',
    );
    await renderDir.create(recursive: true);

    return renderedPages;
  }

  Future<List<RenderedPdfPage>> renderPagesAtDpi(
    String filePath, {
    int maxPages = 12,
    int dpi = 300,
  }) {
    const widthInches = 8.27;
    const heightInches = 11.69;
    return renderPagesForAnalysis(
      filePath,
      maxPages: maxPages,
      width: (widthInches * dpi).round(),
      height: (heightInches * dpi).round(),
    );
  }

  Future<Map<int, String>> extractTextWithMlKit(
    String filePath, {
    int maxPages = 20,
    Function(double progress)? onProgress,
  }) async {
    final pages = await renderPagesForAnalysis(filePath, maxPages: maxPages);
    final recognizer = TextRecognizer(script: TextRecognitionScript.latin);
    final results = <int, String>{};

    try {
      for (int i = 0; i < pages.length; i++) {
        final page = pages[i];
        try {
          final inputImage = InputImage.fromFilePath(page.filePath);
          final recognized = await recognizer.processImage(inputImage);
          final text = recognized.text.trim();
          if (text.isNotEmpty) {
            results[page.pageNumber] = text;
          }
        } catch (e) {
          debugPrint('ML Kit OCR failed on page ${page.pageNumber}: $e');
        }
        if (onProgress != null) {
          onProgress((i + 1) / pages.length);
        }
      }
    } finally {
      await recognizer.close();
    }

    return results;
  }
}

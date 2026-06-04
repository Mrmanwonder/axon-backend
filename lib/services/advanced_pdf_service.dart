import 'dart:typed_data';

class RenderedPdfPage {
  final int pageNumber;
  final double width;
  final double height;
  final Uint8List bytes;

  RenderedPdfPage({
    required this.pageNumber,
    required this.width,
    required this.height,
    Uint8List? bytes,
  }) : bytes = bytes ?? Uint8List(0);
}

class AdvancedPdfService {
  AdvancedPdfService();

  Future<Map<int, String>> extractTextWithMlKit(
    String filePath, {
    int maxPages = 12,
  }) async =>
      {};

  Future<List<RenderedPdfPage>> renderPagesAtDpi(
    String filePath, {
    int maxPages = 12,
    int dpi = 300,
  }) async =>
      [];
}

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:syncfusion_flutter_pdfviewer/pdfviewer.dart';
import '../../theme/app_theme.dart';

class PdfMockScreen extends ConsumerStatefulWidget {
  final String filePath;
  final String title;

  const PdfMockScreen({
    super.key,
    required this.filePath,
    required this.title,
  });

  @override
  ConsumerState<PdfMockScreen> createState() => _PdfMockScreenState();
}

class _PdfMockScreenState extends ConsumerState<PdfMockScreen> {
  late PdfViewerController _pdfController;

  @override
  void initState() {
    super.initState();
    _pdfController = PdfViewerController();
  }

  @override
  void dispose() {
    _pdfController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: AxonColors.background,
        foregroundColor: AxonColors.textPrimary,
        title: Text(
          widget.title,
          style: GoogleFonts.inter(fontWeight: FontWeight.w600),
        ),
      ),
      body: widget.filePath.isEmpty
          ? Center(
              child: Text(
                'No PDF file selected',
                style: TextStyle(color: AxonColors.textPrimary),
              ),
            )
          : SfPdfViewer.file(
              File(widget.filePath),
              controller: _pdfController,
            ),
    );
  }
}
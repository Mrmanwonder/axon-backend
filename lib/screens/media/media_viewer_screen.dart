// lib/screens/media/media_viewer_screen.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_pdfviewer/pdfviewer.dart';
import 'package:url_launcher/url_launcher.dart';
import '../../services/media_viewer_service.dart';
import '../../utils/nav_utils.dart';

class MediaViewerScreen extends ConsumerStatefulWidget {
  final String? url;
  final String? title;
  final MediaType? mediaType;
  final String? resumeKey;

  const MediaViewerScreen({
    super.key,
    this.url,
    this.title,
    this.mediaType,
    this.resumeKey,
  });

  @override
  ConsumerState<MediaViewerScreen> createState() => _MediaViewerScreenState();
}

class _MediaViewerScreenState extends ConsumerState<MediaViewerScreen> {
  final PdfViewerController _pdfController = PdfViewerController();

  bool _isLoading = true;
  String? _error;
  int _targetPage = 0;

  // Annotation State
  PdfAnnotationMode _annotationMode = PdfAnnotationMode.none;

  @override
  void initState() {
    super.initState();
    _initializeViewer();
  }

  Future<void> _initializeViewer() async {
    if (widget.url == null || widget.url!.isEmpty) {
      if (mounted) setState(() { _error = 'INVALID_SOURCE_URL'; _isLoading = false; });
      return;
    }

    final detectedType = widget.mediaType ?? MediaViewerService.detectType(widget.url!);

    // Pre-fetch the saved position, but wait to apply it until document loads
    if (detectedType == MediaType.pdf && widget.resumeKey != null) {
      final position = await PdfResumeService.getPosition(widget.resumeKey!);
      if (position != null) {
        _targetPage = position.page;
      }
    }

    if (mounted) setState(() => _isLoading = false);
  }

  Future<void> _savePosition() async {
    if (widget.resumeKey != null && _pdfController.pageNumber > 0) {
      await PdfResumeService.savePosition(
        widget.resumeKey!, 
        _pdfController.pageNumber, 
        _pdfController.scrollOffset.dy,
      );
    }
  }

  @override
  void dispose() {
    _savePosition();
    _pdfController.dispose();
    super.dispose();
  }

  void _onDocumentLoaded(PdfDocumentLoadedDetails details) {
    if (_targetPage > 1) {
      // Jump to the saved page only after the PDF engine has rendered the file
      _pdfController.jumpToPage(_targetPage);
    }
  }

  void _toggleAnnotation(PdfAnnotationMode mode) {
    HapticFeedback.selectionClick();
    setState(() {
      _annotationMode = _annotationMode == mode ? PdfAnnotationMode.none : mode;
    });
  }

  @override
  Widget build(BuildContext context) {
    final type = widget.mediaType ?? MediaViewerService.detectType(widget.url ?? '');
    final title = widget.title ?? 'DOCUMENT_VIEWER';

    return Scaffold(
      backgroundColor: const Color(0xFF000000), // Pure Matte Black
      body: Stack(
        children: [
          // 1. The Core Viewer Layer
          Positioned.fill(
            child: _isLoading 
                ? const Center(child: CircularProgressIndicator(color: Colors.white24))
                : _error != null 
                    ? _buildErrorState() 
                    : _buildViewer(type),
          ),

          // 2. Floating Structural Glass Header
          Positioned(
            top: 0, left: 0, right: 0,
            child: _buildGlassHeader(title),
          ),

          // 3. Floating Annotation Dock (Only for PDFs)
          if (type == MediaType.pdf && !_isLoading && _error == null)
            Positioned(
              bottom: 32, left: 0, right: 0,
              child: _buildAnnotationDock(),
            ),
        ],
      ),
    );
  }

  Widget _buildViewer(MediaType type) {
    if (type == MediaType.pdf) {
      return SfPdfViewer.network(
        widget.url!,
        controller: _pdfController,
        canShowScrollHead: false, // Cleaner UI
        pageSpacing: 8,
        onDocumentLoaded: _onDocumentLoaded,
        onPageChanged: (details) => _savePosition(), // Auto-save on page turn
      );
    } else if (type == MediaType.youtube) {
      return Center(
        child: FilledButton.tonalIcon(
          onPressed: _openExternalMedia,
          icon: const Icon(Icons.open_in_new_rounded),
          label: const Text('Open Video'),
        ),
      );
    }
    return _buildErrorState("UNSUPPORTED_MEDIA_TYPE");
  }

  Future<void> _openExternalMedia() async {
    final rawUrl = widget.url;
    if (rawUrl == null || rawUrl.isEmpty) return;
    final uri = Uri.tryParse(rawUrl);
    if (uri == null) return;
    await launchUrl(uri, mode: LaunchMode.externalApplication);
  }

  // --- UI Components: The "Matte-Utilitarian" Aesthetic ---

  Widget _buildGlassHeader(String title) {
    return RepaintBoundary(
      child: ClipRRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
          child: Container(
            padding: EdgeInsets.only(
              top: MediaQuery.paddingOf(context).top + 12, 
              bottom: 16, left: 20, right: 20
            ),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.03),
              border: Border(
                bottom: BorderSide(color: Colors.white.withValues(alpha: 0.08), width: 0.5),
              ),
            ),
            child: Row(
              children: [
                GestureDetector(
                  onTap: () {
                    HapticFeedback.lightImpact();
                    _savePosition();
                    popOrGo(context, '/home');
                  },
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: Colors.white.withValues(alpha: 0.05),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Icon(Icons.arrow_back_ios_new, color: Colors.white, size: 16),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text("SOURCE_MATERIAL", style: TextStyle(color: Colors.white24, fontSize: 9, letterSpacing: 1.5, fontFamily: 'Monospace')),
                      const SizedBox(height: 2),
                      Text(
                        title.toUpperCase(),
                        style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.5),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAnnotationDock() {
    return Center(
      child: RepaintBoundary(
        child: ClipRRect(
          borderRadius: BorderRadius.circular(24),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
            child: Container(
              height: 56,
              padding: const EdgeInsets.symmetric(horizontal: 8),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white.withValues(alpha: 0.1), width: 0.5),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _ToolButton(
                    icon: Icons.pan_tool_outlined,
                    isActive: _annotationMode == PdfAnnotationMode.none,
                    onTap: () => _toggleAnnotation(PdfAnnotationMode.none),
                  ),
                  Container(width: 1, height: 24, color: Colors.white.withValues(alpha: 0.1), margin: const EdgeInsets.symmetric(horizontal: 4)),
                  _ToolButton(
                    icon: Icons.format_color_text,
                    isActive: _annotationMode == PdfAnnotationMode.highlight,
                    onTap: () => _toggleAnnotation(PdfAnnotationMode.highlight),
                  ),
                  _ToolButton(
                    icon: Icons.format_underlined,
                    isActive: _annotationMode == PdfAnnotationMode.underline,
                    onTap: () => _toggleAnnotation(PdfAnnotationMode.underline),
                  ),
                  _ToolButton(
                    icon: Icons.format_strikethrough,
                    isActive: _annotationMode == PdfAnnotationMode.strikethrough,
                    onTap: () => _toggleAnnotation(PdfAnnotationMode.strikethrough),
                  ),
                  Container(width: 1, height: 24, color: Colors.white.withValues(alpha: 0.1), margin: const EdgeInsets.symmetric(horizontal: 4)),
                  _ToolButton(
                    icon: Icons.undo,
                    isActive: false,
                    onTap: () {
                      HapticFeedback.lightImpact();
                      // Syncfusion doesn't expose a direct 'undo' without managing the annotation collection, 
                      // but we can clear selections or rely on the user tapping an annotation to delete it.
                    },
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildErrorState([String? explicitError]) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.warning_amber_rounded, size: 32, color: Colors.white24),
          const SizedBox(height: 16),
          Text(
            explicitError ?? _error ?? "SYSTEM_ERROR", 
            style: const TextStyle(color: Colors.white54, fontSize: 12, fontFamily: 'Monospace')
          ),
        ],
      ),
    );
  }
}

class _ToolButton extends StatelessWidget {
  final IconData icon;
  final bool isActive;
  final VoidCallback onTap;

  const _ToolButton({required this.icon, required this.isActive, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        padding: const EdgeInsets.all(10),
        margin: const EdgeInsets.symmetric(horizontal: 4),
        decoration: BoxDecoration(
          color: isActive ? Colors.white.withValues(alpha: 0.1) : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(
          icon,
          size: 20,
          color: isActive ? Colors.white : Colors.white38,
        ),
      ),
    );
  }
}

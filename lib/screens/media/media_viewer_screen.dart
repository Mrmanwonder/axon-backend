// lib/screens/media/media_viewer_screen.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

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


  bool _isLoading = true;
  String? _error;

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

    if (mounted) setState(() => _isLoading = false);
  }

  @override
  void dispose() {
    super.dispose();
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
        ],
      ),
    );
  }

  Widget _buildViewer(MediaType type) {
    if (type == MediaType.youtube) {
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



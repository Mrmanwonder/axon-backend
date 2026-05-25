// lib/screens/video/video_player_screen.dart
// Video Player Screen - displays video content in-app
//
// Usage: Navigate to /video?url=<encoded_url>&title=<encoded_title>

import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:url_launcher/url_launcher.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/rose_loader.dart';

class VideoPlayerScreen extends StatefulWidget {
  final String? url;
  final String? title;

  const VideoPlayerScreen({
    super.key,
    this.url,
    this.title,
  });

  @override
  State<VideoPlayerScreen> createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen> {
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadVideo();
  }

  void _loadVideo() {
    if (widget.url == null || widget.url!.isEmpty) {
      setState(() {
        _error = 'No video URL provided';
        _isLoading = false;
      });
      return;
    }
    setState(() {
      _isLoading = false;
    });
  }

  Future<void> _openInExternalPlayer() async {
    final uri = Uri.parse(widget.url!);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = AxonThemeMode.isDark;
    final bgColor = isDark ? const Color(0xFF090A0B) : const Color(0xFFF5F5F5);

    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        backgroundColor: AxonColors.surface,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: AxonColors.textPrimary),
          onPressed: () => popOrGo(context, '/home'),
        ),
        title: Text(
          widget.title ?? 'Video Player',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.open_in_new, color: AxonColors.textSecondary),
            onPressed: _openInExternalPlayer,
            tooltip: 'Open in external player',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: RoseLoader(size: 24))
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.error_outline,
                          size: 48, color: AxonColors.textTertiary),
                      const SizedBox(height: 16),
                      Text(
                        _error!,
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textSecondary,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                )
              : Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.play_circle_outline,
                          size: 64, color: AxonColors.electricCyan),
                      const SizedBox(height: 16),
                      Text(
                        'Video: ${widget.url}',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textSecondary,
                          fontSize: 14,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 24),
                      ElevatedButton.icon(
                        onPressed: _openInExternalPlayer,
                        icon: const Icon(Icons.open_in_new),
                        label: const Text('Open in External Player'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AxonColors.electricCyan,
                          foregroundColor: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
    );
  }
}

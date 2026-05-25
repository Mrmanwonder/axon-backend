import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../services/google_drive_downloader.dart';

class GemmaDownloadProgressWidget extends StatefulWidget {
  final GoogleDriveDownloader downloader;
  final VoidCallback? onComplete;
  final EdgeInsets? margin;

  const GemmaDownloadProgressWidget({
    super.key,
    required this.downloader,
    this.onComplete,
    this.margin,
  });

  @override
  State<GemmaDownloadProgressWidget> createState() =>
      _GemmaDownloadProgressWidgetState();
}

class _GemmaDownloadProgressWidgetState
    extends State<GemmaDownloadProgressWidget> {
  late GemmaDownloadState _state;

  @override
  void initState() {
    super.initState();
    _state = widget.downloader.state.value;
    widget.downloader.state.addListener(_onStateChanged);
  }

  @override
  void dispose() {
    widget.downloader.state.removeListener(_onStateChanged);
    super.dispose();
  }

  void _onStateChanged() {
    if (!mounted) return;
    setState(() {
      _state = widget.downloader.state.value;
    });

    if (_state.isComplete && widget.onComplete != null) {
      widget.onComplete!();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: widget.margin ??
          const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF121212),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: _state.isComplete
              ? const Color(0xFF22C55E).withValues(alpha: 0.3)
              : _state.error != null
                  ? const Color(0xFFEF4444).withValues(alpha: 0.3)
                  : const Color(0xFF3A86FF).withValues(alpha: 0.2),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildHeader(),
            const SizedBox(height: 12),
            _buildProgressBar(),
            const SizedBox(height: 8),
            _buildStatusText(),
            if (_state.isDownloading ||
                _state.isComplete ||
                _state.error != null)
              const SizedBox(height: 12),
            if (_state.isDownloading ||
                _state.error != null ||
                _state.status.contains('Paused'))
              _buildActionButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: _state.isComplete
                ? const Color(0xFF22C55E).withValues(alpha: 0.15)
                : const Color(0xFF3A86FF).withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Icon(
            _state.isComplete
                ? Icons.check_circle_rounded
                : _state.error != null
                    ? Icons.error_outline_rounded
                    : Icons.download_rounded,
            color: _state.isComplete
                ? const Color(0xFF22C55E)
                : _state.error != null
                    ? const Color(0xFFEF4444)
                    : const Color(0xFF3A86FF),
            size: 20,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            _state.isComplete ? 'MODEL READY' : 'OFFLINE AI MODEL',
            style: GoogleFonts.googleSans(
              color: const Color(0xFF8B949E),
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.5,
            ),
          ),
        ),
        if (_state.isComplete)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0xFF22C55E).withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              'ACTIVE',
              style: GoogleFonts.googleSans(
                color: const Color(0xFF22C55E),
                fontSize: 9,
                fontWeight: FontWeight.w700,
                letterSpacing: 1,
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildProgressBar() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(6),
      child: LinearProgressIndicator(
        value: _state.progress,
        minHeight: 8,
        backgroundColor: const Color(0xFF1A1A1A),
        valueColor: AlwaysStoppedAnimation(
          _state.isComplete
              ? const Color(0xFF22C55E)
              : _state.error != null
                  ? const Color(0xFFEF4444)
                  : const Color(0xFF3A86FF),
        ),
      ),
    );
  }

  Widget _buildStatusText() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Expanded(
          child: Text(
            _state.status,
            style: GoogleFonts.googleSans(
              color: _state.error != null
                  ? const Color(0xFFEF4444)
                  : const Color(0xFF8B949E),
              fontSize: 12,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ),
        Text(
          '${(_state.progress * 100).toInt()}%',
          style: GoogleFonts.googleSans(
            color: _state.isComplete
                ? const Color(0xFF22C55E)
                : const Color(0xFF3A86FF),
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  Widget _buildActionButtons() {
    return Row(
      children: [
        if (_state.isDownloading)
          Expanded(
            child: _ActionButton(
              icon: Icons.pause_rounded,
              label: 'Pause',
              color: const Color(0xFFF59E0B),
              onTap: () {
                HapticFeedback.lightImpact();
                widget.downloader.pauseDownload();
              },
            ),
          ),
        if (_state.status.contains('Paused'))
          Expanded(
            child: _ActionButton(
              icon: Icons.play_arrow_rounded,
              label: 'Resume',
              color: const Color(0xFF22C55E),
              onTap: () {
                HapticFeedback.lightImpact();
                widget.downloader.resumeDownload();
              },
            ),
          ),
        if (_state.error != null || _state.status.contains('failed'))
          Expanded(
            child: _ActionButton(
              icon: Icons.refresh_rounded,
              label: 'Retry',
              color: const Color(0xFF3A86FF),
              onTap: () {
                HapticFeedback.lightImpact();
                widget.downloader.triggerDownload();
              },
            ),
          ),
        if (_state.isDownloading || _state.status.contains('Paused'))
          Expanded(
            child: _ActionButton(
              icon: Icons.close_rounded,
              label: 'Cancel',
              color: const Color(0xFFEF4444),
              onTap: () {
                HapticFeedback.lightImpact();
                widget.downloader.cancelDownload();
              },
            ),
          ),
      ],
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 4),
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: color.withValues(alpha: 0.25)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: color, size: 16),
            const SizedBox(width: 6),
            Text(
              label,
              style: GoogleFonts.googleSans(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

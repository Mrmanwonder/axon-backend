import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../services/alert_service.dart';
import '../../services/hybrid_speech_service.dart';
import '../../theme/app_theme.dart';

// ═══════════════════════════════════════════════════════════════════════════
// Data
// ═══════════════════════════════════════════════════════════════════════════

class AskAxonAttachment {
  final String name;
  final IconData icon;
  final VoidCallback? onRemove;
  const AskAxonAttachment({
    required this.name,
    this.icon = Icons.insert_drive_file_outlined,
    this.onRemove,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Widget
// ═══════════════════════════════════════════════════════════════════════════

class AskAxonWidget extends ConsumerStatefulWidget {
  final ValueChanged<String>? onSubmit;
  final String? hintText;
  final String? initialText;
  final List<AskAxonAttachment> attachments;
  final VoidCallback? onAttach;
  final VoidCallback? onPlan;
  final VoidCallback? onDoubt;
  final bool isPlanActive;
  final bool isDoubtActive;
  final bool isSending;
  final bool autofocus;

  const AskAxonWidget({
    super.key,
    this.onSubmit,
    this.hintText = 'Ask anything',
    this.initialText,
    this.attachments = const [],
    this.onAttach,
    this.onPlan,
    this.onDoubt,
    this.isPlanActive = false,
    this.isDoubtActive = false,
    this.isSending = false,
    this.autofocus = false,
  });

  @override
  ConsumerState<AskAxonWidget> createState() => _AskAxonWidgetState();
}

class _AskAxonWidgetState extends ConsumerState<AskAxonWidget> {
  static const Color _accent = Color(0xFF3A86FF);
  static const Duration _debounceDelay = Duration(milliseconds: 400);

  final TextEditingController _textController = TextEditingController();
  final FocusNode _focusNode = FocusNode();
  DateTime _lastSubmit = DateTime(2000);

  bool _isListening = false;
  bool _speechAvailable = false;

  // Audio level smoothing
  double _smoothedLevel = 0.0;

  // Subscriptions
  StreamSubscription<HybridTranscript>? _transcriptSub;
  StreamSubscription<double>? _levelSub;

  @override
  void initState() {
    super.initState();
    final init = widget.initialText?.trim();
    if (init != null && init.isNotEmpty) {
      _textController.text = init;
      _textController.selection = TextSelection.collapsed(offset: init.length);
    }
    unawaited(_checkAvailability());
  }

  Future<void> _checkAvailability() async {
    try {
      final ok = await HybridSpeechService.instance.isAvailable();
      if (mounted) setState(() => _speechAvailable = ok);
    } catch (_) {
      if (mounted) setState(() => _speechAvailable = false);
    }
  }

  // ── Voice toggle ────────────────────────────────────────────────────────

  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    try {
      HapticFeedback.lightImpact();

      // Subscribe to transcripts
      _transcriptSub?.cancel();
      _transcriptSub = HybridSpeechService.instance.transcriptStream
          .listen(_onTranscript, onError: _onTranscriptError);

      // Subscribe to audio levels
      _levelSub?.cancel();
      _levelSub =
          HybridSpeechService.instance.audioLevelStream.listen(_onAudioLevel);

      await HybridSpeechService.instance.startListening();

      if (!mounted) return;
      setState(() {
        _isListening = true;
        _speechAvailable = true;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _isListening = false;
        _speechAvailable = false;
      });
      AlertService.showError(context, 'Voice unavailable', error.toString());
    }
  }

  Future<void> _stopListening({bool isDisposing = false}) async {
    if (!mounted) return;
    if (!isDisposing) {
      setState(() => _isListening = false);
    }
    _smoothedLevel = 0.0;

    try {
      _transcriptSub?.cancel();
      _transcriptSub = null;
      _levelSub?.cancel();
      _levelSub = null;

      await HybridSpeechService.instance.stopListening();
    } catch (error) {
      if (!mounted || isDisposing) return;
      AlertService.showError(context, 'Voice input failed', error.toString());
    }
  }

  void _onTranscript(HybridTranscript transcript) {
    if (!mounted) return;
    final text = transcript.text.trim();
    if (text.isEmpty) return;
    _setText(text);
  }

  void _onTranscriptError(Object error) {
    if (!mounted) return;
    setState(() => _isListening = false);
    AlertService.showError(context, 'Voice input failed', error.toString());
  }

  void _onAudioLevel(double level) {
    if (!mounted) return;
    _smoothedLevel = _smoothedLevel * 0.6 + level * 0.4;
  }

  void _setText(String value) {
    setState(() {
      _textController.text = value;
      _textController.selection = TextSelection.collapsed(offset: value.length);
    });
  }

  void _submitText() {
    if (widget.isSending) return;
    final now = DateTime.now();
    if (now.difference(_lastSubmit) < _debounceDelay) return;

    final text = _textController.text.trim();
    if (text.isEmpty) return;

    _lastSubmit = now;
    widget.onSubmit?.call(text);
    _textController.clear();
    _focusNode.unfocus();
  }

  // ── Build ───────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final hasText = _textController.text.trim().isNotEmpty;
    final isDark = AxonThemeMode.isDark;
    final panelColor = isDark ? AxonColors.surfaceElevated : Colors.white;
    final borderColor = isDark
        ? Colors.white.withValues(alpha: 0.09)
        : Colors.black.withValues(alpha: 0.07);
    final shadowColor = isDark
        ? Colors.black.withValues(alpha: 0.34)
        : Colors.black.withValues(alpha: 0.16);

    return Material(
      color: Colors.transparent,
      child: Container(
        width: double.infinity,
        constraints: const BoxConstraints(maxWidth: 430),
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
        decoration: BoxDecoration(
          color: panelColor,
          borderRadius: BorderRadius.circular(28),
          border: Border.all(color: borderColor),
          boxShadow: [
            BoxShadow(
              color: shadowColor,
              blurRadius: 18,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Attachment strip
            if (widget.attachments.isNotEmpty) ...[
              _AttachmentStrip(attachments: widget.attachments),
              const SizedBox(height: 10),
            ],

            // Text field
            TextField(
              controller: _textController,
              focusNode: _focusNode,
              autofocus: widget.autofocus,
              minLines: 1,
              maxLines: 4,
              textInputAction: TextInputAction.send,
              style: TextStyle(
                color: AxonColors.textPrimary,
                fontSize: 15,
                height: 1.3,
              ),
              decoration: InputDecoration(
                hintText: _isListening ? 'Listening...' : widget.hintText,
                hintStyle: TextStyle(
                  color: _isListening
                      ? _accent.withValues(alpha: 0.7)
                      : AxonColors.textTertiary,
                  fontSize: 15,
                ),
                border: InputBorder.none,
                isDense: true,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
              ),
              onChanged: (_) => setState(() {}),
              onSubmitted: (_) {
                if (!widget.isSending) _submitText();
              },
            ),

            // ── Action row ──
            Row(
              children: [
                _RoundToolButton(
                  icon: Icons.attach_file_rounded,
                  onTap: widget.onAttach,
                  active: widget.attachments.isNotEmpty,
                ),
                const SizedBox(width: 8),
                _PillToolButton(
                  icon: Icons.event_note_rounded,
                  label: 'Plan',
                  onTap: widget.onPlan,
                  active: widget.isPlanActive,
                ),
                const SizedBox(width: 8),
                _PillToolButton(
                  icon: Icons.menu_book_outlined,
                  label: 'Doubt',
                  onTap: widget.onDoubt,
                  active: widget.isDoubtActive,
                ),
                const Spacer(),
                _VoiceButton(
                  enabled: _speechAvailable || !_isListening,
                  isListening: _isListening,
                  audioLevel: _smoothedLevel,
                  onTap: _toggleListening,
                ),
                if (hasText && !widget.isSending) ...[
                  const SizedBox(width: 8),
                  _RoundToolButton(
                    icon: Icons.arrow_upward_rounded,
                    foreground: Colors.white,
                    background: _accent,
                    onTap: _submitText,
                  ),
                ],
                if (widget.isSending) ...[
                  const SizedBox(width: 8),
                  const SizedBox(
                    width: 36, height: 36,
                    child: Center(
                      child: SizedBox(
                        width: 16, height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Color(0xFF3A86FF),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _transcriptSub?.cancel();
    _levelSub?.cancel();
    _stopListening(isDisposing: true);
    _textController.dispose();
    _focusNode.dispose();
    super.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-widgets (unchanged API, updated mic button visuals)
// ═══════════════════════════════════════════════════════════════════════════

class _AttachmentStrip extends StatelessWidget {
  final List<AskAxonAttachment> attachments;
  const _AttachmentStrip({required this.attachments});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        children: [
          for (var i = 0; i < attachments.length; i++) ...[
            _AttachmentChip(attachment: attachments[i]),
            if (i != attachments.length - 1) const SizedBox(width: 8),
          ],
        ],
      ),
    );
  }
}

class _AttachmentChip extends StatelessWidget {
  final AskAxonAttachment attachment;
  const _AttachmentChip({required this.attachment});

  @override
  Widget build(BuildContext context) {
    final chipColor = AxonThemeMode.isDark
        ? AxonColors.surfaceHighlight
        : const Color(0xFFF4F4F4);
    return Container(
      height: 42,
      constraints: const BoxConstraints(maxWidth: 184),
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: chipColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AxonColors.divider),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(attachment.icon, color: AxonColors.textPrimary, size: 22),
          const SizedBox(width: 10),
          Flexible(
            child: Text(
              attachment.name,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(color: AxonColors.textPrimary, fontSize: 14),
            ),
          ),
          const SizedBox(width: 10),
          GestureDetector(
            onTap: attachment.onRemove,
            behavior: HitTestBehavior.opaque,
            child: Icon(Icons.close_rounded,
                color: AxonColors.textSecondary, size: 18),
          ),
        ],
      ),
    );
  }
}

class _RoundToolButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onTap;
  final Color foreground;
  final Color background;
  final bool active;

  const _RoundToolButton({
    required this.icon,
    this.onTap,
    this.foreground = Colors.black,
    this.background = Colors.white,
    this.active = false,
  });

  @override
  Widget build(BuildContext context) {
    final effectiveForeground = active
        ? Colors.white
        : background == Colors.white
            ? AxonColors.textPrimary
            : foreground;
    final effectiveBackground = active
        ? const Color(0xFF3A86FF)
        : background == Colors.white
            ? (AxonThemeMode.isDark
                ? AxonColors.surfaceHighlight
                : Colors.white)
            : background;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Ink(
        width: 36,
        height: 26,
        decoration: BoxDecoration(
          color: effectiveBackground,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: active
                ? const Color(0xFF3A86FF).withValues(alpha: 0.5)
                : AxonColors.divider,
          ),
        ),
        child: Icon(icon, color: effectiveForeground, size: 17),
      ),
    );
  }
}

class _PillToolButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  final bool active;

  const _PillToolButton({
    required this.icon,
    required this.label,
    this.onTap,
    this.active = false,
  });

  @override
  Widget build(BuildContext context) {
    final foreground = active ? Colors.white : AxonColors.textPrimary;
    final background = active
        ? const Color(0xFF3A86FF)
        : AxonThemeMode.isDark
            ? AxonColors.surfaceHighlight
            : Colors.white;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Ink(
        height: 26,
        padding: const EdgeInsets.symmetric(horizontal: 10),
        decoration: BoxDecoration(
          color: background,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: active
                ? const Color(0xFF3A86FF).withValues(alpha: 0.5)
                : AxonColors.divider,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: foreground, size: 15),
            const SizedBox(width: 4),
            Text(label,
                style: TextStyle(
                  color: foreground,
                  fontSize: 12,
                  fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                )),
          ],
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Voice Button — audio-reactive mic indicator
// ═══════════════════════════════════════════════════════════════════════════

class _VoiceButton extends StatelessWidget {
  final bool enabled;
  final bool isListening;
  final double audioLevel;
  final Future<void> Function() onTap;

  const _VoiceButton({
    required this.enabled,
    required this.isListening,
    required this.audioLevel,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final idleBackground = AxonThemeMode.isDark
        ? AxonColors.surfaceHighlight
        : const Color(0xFFF5F5F5);
    final idleForeground = AxonColors.textPrimary;
    return InkWell(
      onTap: enabled ? () => unawaited(onTap()) : null,
      borderRadius: BorderRadius.circular(16),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        width: 34,
        height: 28,
        decoration: BoxDecoration(
          color: isListening
              ? const Color(0xFF3A86FF).withValues(alpha: 0.15)
              : idleBackground,
          borderRadius: BorderRadius.circular(16),
          border: isListening
              ? Border.all(
                  color: const Color(0xFF3A86FF).withValues(alpha: 0.3),
                  width: 1.5)
              : null,
        ),
        child: Center(
          child: isListening
              ? CustomPaint(
                  size: const Size(22, 18),
                  painter: _LiveMicBarsPainter(audioLevel: audioLevel),
                )
              : Icon(Icons.mic_none_rounded, color: idleForeground, size: 17),
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Mic button's mini bars — audio-level reactive only
// ═══════════════════════════════════════════════════════════════════════════

class _LiveMicBarsPainter extends CustomPainter {
  final double audioLevel;

  _LiveMicBarsPainter({
    required this.audioLevel,
  });

  @override
  void paint(Canvas canvas, Size size) {
    const bars = 5;
    final gap = size.width / (bars - 1);
    final centerY = size.height / 2;
    // Base heights for each bar (EQ-style: lower edges, higher middle)
    const barBaseHeights = [0.3, 0.65, 1.0, 0.65, 0.3];
    // Clamp level so bars never fully disappear
    final level = (audioLevel * 0.85).clamp(0.05, 1.0);

    for (int i = 0; i < bars; i++) {
      final h = size.height * barBaseHeights[i] * level;

      final paint = Paint()
        ..strokeCap = StrokeCap.round
        ..strokeWidth = 2.5
        ..color = const Color(0xFF3A86FF).withValues(alpha: 0.5 + level * 0.5);

      canvas.drawLine(
        Offset(i * gap, centerY - h / 2),
        Offset(i * gap, centerY + h / 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _LiveMicBarsPainter old) {
    return (old.audioLevel - audioLevel).abs() > 0.01;
  }
}

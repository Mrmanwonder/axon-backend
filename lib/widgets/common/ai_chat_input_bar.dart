// lib/widgets/ai_chat_input_bar.dart
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

class AiChatInputBar extends StatefulWidget {
  final TextEditingController? controller;
  final Function(String)? onSend;
  final Function()? onMicTap;
  final Function()? onAttachTap;
  final bool isWorking;
  final String hintText;

  const AiChatInputBar({
    super.key,
    this.controller,
    this.onSend,
    this.onMicTap,
    this.onAttachTap,
    this.isWorking = false,
    this.hintText = 'ASK AXON...',
  });

  @override
  State<AiChatInputBar> createState() => _AiChatInputBarState();
}

class _AiChatInputBarState extends State<AiChatInputBar>
    with SingleTickerProviderStateMixin {
  late final TextEditingController _controller;
  final FocusNode _focusNode = FocusNode();
  bool _hasText = false;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? TextEditingController();
    _controller.addListener(() {
      final hasText = _controller.text.isNotEmpty;
      if (hasText != _hasText) setState(() => _hasText = hasText);
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeInOutCubic,
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      // Morphing height based on text presence
      height: _hasText ? 140 : 56,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(_hasText ? 24 : 28),
        boxShadow: [
          BoxShadow(
            color: _hasText
                ? Colors.blue.withValues(alpha: 0.08)
                : Colors.transparent,
            blurRadius: 20,
            spreadRadius: 2,
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(_hasText ? 24 : 28),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
          child: Container(
            padding: const EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(_hasText ? 24 : 28),
              border: Border.all(
                color: Colors.white.withValues(alpha: 0.12),
                width: 0.8,
              ),
            ),
            child: Column(
              children: [
                // 1. INPUT AREA (Always visible, shifts to top when _hasText)
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Center(
                      child: TextField(
                        controller: _controller,
                        focusNode: _focusNode,
                        maxLines: _hasText ? 3 : 1,
                        style: GoogleFonts.robotoMono(
                          color: Colors.white,
                          fontSize: 14,
                        ),
                        cursorColor: Colors.blue,
                        decoration: InputDecoration(
                          hintText: widget.hintText,
                          hintStyle: GoogleFonts.robotoMono(
                            color: Colors.white24,
                            fontSize: 10,
                            letterSpacing: 1.5,
                          ),
                          border: InputBorder.none,
                          contentPadding:
                              EdgeInsets.symmetric(vertical: _hasText ? 12 : 0),
                        ),
                      ),
                    ),
                  ),
                ),

                // 2. ACTION ROW (Visible as trailing icons in pill, or bottom row in box)
                if (!_hasText) ...[
                  _buildPillActions()
                ] else ...[
                  _buildExpandedActions()
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  // Layout for the "Pill" state (Idle)
  Widget _buildPillActions() {
    return Positioned.fill(
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          _buildIconButton(
              Icons.add_rounded, widget.onAttachTap, Colors.white38),
          const Spacer(),
          _buildIconButton(
              Icons.mic_none_rounded, widget.onMicTap, Colors.white54),
        ],
      ),
    );
  }

  // Layout for the "Expanded" state (Typing)
  Widget _buildExpandedActions() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Row(
        children: [
          _buildIconButton(
              Icons.add_rounded, widget.onAttachTap, Colors.white38),
          const Spacer(),
          // Send Button
          GestureDetector(
            onTap: () {
              if (_controller.text.isNotEmpty) {
                widget.onSend?.call(_controller.text);
                _controller.clear();
                HapticFeedback.mediumImpact();
              }
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.blue.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.blue.withValues(alpha: 0.3)),
              ),
              child: Row(
                children: [
                  Text("SEND",
                      style: GoogleFonts.robotoMono(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.bold)),
                  const SizedBox(width: 8),
                  const Icon(Icons.arrow_upward_rounded,
                      color: Colors.white, size: 14),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton(IconData icon, VoidCallback? onTap, Color color) {
    return IconButton(
      onPressed: () {
        HapticFeedback.lightImpact();
        onTap?.call();
      },
      icon: Icon(icon, color: color, size: 20),
    );
  }
}

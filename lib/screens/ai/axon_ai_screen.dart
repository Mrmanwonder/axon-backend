import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../models/models.dart';
import '../../theme/app_theme.dart';
import '../../services/ask_axon_context_service.dart';
import '../../services/chat_history_service.dart';
import '../../services/grok_service.dart';
import '../../widgets/common/ask_axon_widget.dart';
import '../../widgets/math/math_markdown_message.dart';
import '../shell_screen.dart';

/// Riverpod provider for AI chat state.
final aiChatProvider = StateNotifierProvider<_AiChatNotifier, _AiChatState>(
  (ref) => _AiChatNotifier(),
);

class AxonAiScreen extends ConsumerStatefulWidget {
  final String title;
  final String? initialQuestion;
  final String? contextNote;
  final Future<String>? contextFuture;
  final MotivationStyle motivationStyle;

  const AxonAiScreen({
    super.key,
    this.title = 'Ask Axon',
    this.initialQuestion,
    this.contextNote,
    this.contextFuture,
    this.motivationStyle = MotivationStyle.logicBased,
  });

  @override
  ConsumerState<AxonAiScreen> createState() => _AxonAiScreenState();
}

class _AxonAiScreenState extends ConsumerState<AxonAiScreen> {
  final ScrollController _scrollController = ScrollController();
  String _screenContext = '';

  @override
  void initState() {
    super.initState();
    Future.microtask(() => _initializeChat());
  }

  Future<void> _initializeChat() async {
    await ChatHistoryService.instance.initialize();
    await _restoreHistory();
    await _loadContext();

    final initial = widget.initialQuestion?.trim();
    if (initial != null && initial.isNotEmpty && mounted) {
      ref.read(aiChatProvider.notifier).submitPrompt(initial, widget.motivationStyle, _screenContext);
    }
  }

  Future<void> _restoreHistory() async {
    final restored = <_AiMessage>[];
    for (final item in ChatHistoryService.instance.history) {
      for (final entry in item.entries) {
        final role = entry.key == 'assistant' ? _AiRole.assistant : _AiRole.user;
        final text = entry.value.trim();
        if (text.isNotEmpty) restored.add(_AiMessage(role: role, text: text));
      }
    }
    if (!mounted || restored.isEmpty) return;
    ref.read(aiChatProvider.notifier).restoreHistory(restored);
  }

  Future<void> _loadContext() async {
    final parts = <String>[];
    if ((widget.contextNote ?? '').trim().isNotEmpty) {
      parts.add(widget.contextNote!.trim());
    }

    if (widget.contextFuture != null) {
      try {
        final value = await widget.contextFuture!.timeout(
          const Duration(seconds: 8),
          onTimeout: () => '',
        );
        if (value.trim().isNotEmpty) parts.add(value.trim());
      } catch (_) {}
    }

    try {
      final appContext = await AskAxonContextService.instance
          .buildContext(extraContext: parts.join('\n\n'))
          .timeout(const Duration(seconds: 8), onTimeout: () => '');
      if (appContext.trim().isNotEmpty) {
        parts..clear()..add(appContext.trim());
      }
    } catch (_) {}

    if (!mounted) return;
    _screenContext = parts.join('\n\n').trim();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 260),
        curve: Curves.easeOutCubic,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(aiChatProvider);

    ref.listen(aiChatProvider, (_, __) {
      if (_scrollController.hasClients &&
          _scrollController.position.pixels >=
              _scrollController.position.maxScrollExtent - 120) {
        _scrollToBottom();
      }
    });

    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: false,
        title: Text(
          widget.title.toUpperCase(),
          style: GoogleFonts.robotoMono(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            letterSpacing: 1.2,
            color: AxonColors.textPrimary,
          ),
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: state.messages.isEmpty
                ? _buildEmptyState()
                : _buildMessageList(state),
          ),
          _buildInputSection(state),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(28),
        child: Text(
          'Ready for session input.\nSelect a mode to begin.',
          textAlign: TextAlign.center,
          style: GoogleFonts.googleSans(color: Colors.white38, fontSize: 14),
        ),
      ),
    );
  }

  Widget _buildMessageList(_AiChatState state) {
    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      itemCount: state.messages.length + (state.isSending ? 1 : 0),
      itemBuilder: (context, i) {
        if (state.isSending && i == state.messages.length) {
          return const _ThinkingIndicator(key: ValueKey('thinking'));
        }
        return RepaintBoundary(
          child: _AiMessageBubble(
            key: ValueKey('msg-$i'),
            message: state.messages[i],
          ),
        );
      },
    );
  }

  Widget _buildInputSection(_AiChatState state) {
    final notifier = ref.read(aiChatProvider.notifier);
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
      child: AskAxonWidget(
        onSubmit: (text) => notifier.submitPrompt(text, widget.motivationStyle, _screenContext),
        onAttach: notifier.pickFiles,
        onPlan: notifier.togglePlanMode,
        onDoubt: notifier.toggleDoubtMode,
        isSending: state.isSending,
        isPlanActive: state.mode == _AssistMode.plan,
        isDoubtActive: state.mode == _AssistMode.doubt,
        hintText: switch (state.mode) {
          _AssistMode.plan => 'Ask Axon to optimize your study block...',
          _AssistMode.doubt => 'Paste your math/physics doubt here...',
          _AssistMode.general => 'Ask anything...',
        },
        attachments: state.attachments.map((file) => AskAxonAttachment(
          name: file.name,
          icon: _attachmentIcon(file.extension),
          onRemove: () => notifier.removeAttachment(file),
        )).toList(),
      ),
    );
  }

  IconData _attachmentIcon(String extension) {
    switch (extension) {
      case 'pdf': return Icons.picture_as_pdf_outlined;
      case 'png': case 'jpg': case 'jpeg': case 'webp': return Icons.image_outlined;
      case 'txt': case 'md': case 'csv': case 'json': return Icons.description_outlined;
      default: return Icons.insert_drive_file_outlined;
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }
}

// ─────────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────

class _AiMessageBubble extends StatelessWidget {
  final _AiMessage message;
  const _AiMessageBubble({required this.message, super.key});

  @override
  Widget build(BuildContext context) {
    final isUser = message.role == _AiRole.user;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        child: GlassCard(
          padding: const EdgeInsets.all(14),
          child: MathMessage(text: message.text),
        ),
      ),
    );
  }
}

class _ThinkingIndicator extends StatelessWidget {
  const _ThinkingIndicator({super.key});
  @override
  Widget build(BuildContext context) => Align(
    alignment: Alignment.centerLeft,
    child: Container(
      margin: const EdgeInsets.only(bottom: 12),
      child: const GlassCard(
        padding: EdgeInsets.all(12),
        child: SizedBox(
          width: 18, height: 18,
          child: CircularProgressIndicator(
            strokeWidth: 2, color: Color(0xFF3A86FF),
          ),
        ),
      ),
    ),
  );
}

// ─────────────────────────────────────────────────────────────────
// STATE MANAGEMENT
// ─────────────────────────────────────────────────────────────────

class _AiChatState {
  final List<_AiMessage> messages;
  final List<_AttachedFile> attachments;
  final bool isSending;
  final _AssistMode mode;

  _AiChatState({
    required this.messages,
    this.attachments = const [],
    this.isSending = false,
    this.mode = _AssistMode.general,
  });

  _AiChatState copyWith({
    List<_AiMessage>? messages,
    List<_AttachedFile>? attachments,
    bool? isSending,
    _AssistMode? mode,
  }) =>
      _AiChatState(
        messages: messages ?? this.messages,
        attachments: attachments ?? this.attachments,
        isSending: isSending ?? this.isSending,
        mode: mode ?? this.mode,
      );
}

class _AiChatNotifier extends StateNotifier<_AiChatState> {
  _AiChatNotifier() : super(_AiChatState(messages: []));

  void restoreHistory(List<_AiMessage> history) {
    state = state.copyWith(messages: history);
  }

  void togglePlanMode() => state = state.copyWith(
    mode: state.mode == _AssistMode.plan ? _AssistMode.general : _AssistMode.plan,
  );

  void toggleDoubtMode() => state = state.copyWith(
    mode: state.mode == _AssistMode.doubt ? _AssistMode.general : _AssistMode.doubt,
  );

  void removeAttachment(_AttachedFile file) {
    state = state.copyWith(
      attachments: state.attachments.where((a) => a != file).toList(),
    );
  }

  Future<void> pickFiles() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        allowMultiple: true,
        withData: false,
        type: FileType.custom,
        allowedExtensions: const [
          'pdf', 'png', 'jpg', 'jpeg', 'webp',
          'txt', 'md', 'csv', 'json', 'doc', 'docx',
        ],
      );
      if (result == null || result.files.isEmpty) return;

      final added = <_AttachedFile>[];
      for (final file in result.files) {
        final preview = await _previewFile(file);
        added.add(_AttachedFile(
          name: file.name,
          path: file.path,
          size: file.size,
          extension: (file.extension ?? '').toLowerCase(),
          previewText: preview,
        ));
      }
      state = state.copyWith(
        attachments: [...state.attachments, ...added],
      );
    } catch (_) {}
  }

  Future<String> _previewFile(PlatformFile file) async {
    final path = file.path;
    if (path == null || path.isEmpty) return '';
    final ext = (file.extension ?? '').toLowerCase();
    const textExtensions = {'txt', 'md', 'csv', 'json'};
    if (!textExtensions.contains(ext) || file.size > 220 * 1024) return '';
    try {
      final text = await File(path).readAsString();
      return text.length <= 2500 ? text : '${text.substring(0, 2500)}\n...';
    } catch (_) {
      return '';
    }
  }

  Future<void> submitPrompt(String text, MotivationStyle style, String screenContext) async {
    if (text.trim().isEmpty || state.isSending) return;

    final userMsg = _AiMessage(role: _AiRole.user, text: text.trim());
    state = state.copyWith(
      messages: [...state.messages, userMsg],
      isSending: true,
    );

    await Future.delayed(Duration.zero);

    final buffer = StringBuffer();
    var assistantIndex = -1;

    try {
      final effectivePrompt = [
        _modeInstructionForUserPrompt(state.mode),
        text.trim(),
        _attachmentContext(),
      ].where((p) => p.trim().isNotEmpty).join('\n\n');

      DateTime lastUpdate = DateTime.now();

      await for (final chunk in GrokService().chatStream(
        effectivePrompt,
        systemPrompt: _buildSystemPrompt(style, state.mode, screenContext),
        context: _recentChatContext(),
        maxTokens: 512,
        temperature: 0.45,
      )) {
        buffer.write(chunk);
        if (assistantIndex < 0) {
          assistantIndex = state.messages.length;
          state = state.copyWith(
            messages: [...state.messages, const _AiMessage(role: _AiRole.assistant, text: '')],
          );
        }
        final now = DateTime.now();
        if (now.difference(lastUpdate).inMilliseconds <= 33) continue;
        lastUpdate = now;
        final updatedMessages = List<_AiMessage>.from(state.messages);
        if (assistantIndex < updatedMessages.length) {
          updatedMessages[assistantIndex] = _AiMessage(
            role: _AiRole.assistant,
            text: buffer.toString(),
          );
          state = state.copyWith(messages: updatedMessages);
        }
      }

      if (assistantIndex >= 0) {
        final fullText = buffer.toString().trim();
        final finalMessages = List<_AiMessage>.from(state.messages);
        if (assistantIndex < finalMessages.length) {
          finalMessages[assistantIndex] = _AiMessage(
            role: _AiRole.assistant,
            text: fullText.isEmpty ? 'I could not reach Axon AI.' : fullText,
          );
          state = state.copyWith(messages: finalMessages);
        }
      }
    } catch (error) {
      state = state.copyWith(
        messages: [
          ...state.messages,
          _AiMessage(
            role: _AiRole.assistant,
            text: 'I could not reach Axon AI: $error',
          ),
        ],
      );
    } finally {
      state = state.copyWith(isSending: false);
    }
  }

  String _attachmentContext() {
    if (state.attachments.isEmpty) return '';
    final buffer = StringBuffer('Attached files:');
    for (final file in state.attachments) {
      buffer.writeln(
        '- ${file.name} (${file.extension.isEmpty ? 'file' : file.extension}, ${_formatBytes(file.size)}${file.path == null ? '' : ', path: ${file.path}'})',
      );
      if (file.previewText.trim().isNotEmpty) {
        buffer.writeln('  Preview:\n${file.previewText.trim()}');
      }
    }
    return buffer.toString().trim();
  }

  List<Map<String, String>> _recentChatContext() {
    final result = <Map<String, String>>[];
    for (var i = state.messages.length - 1; i >= 0 && result.length < 8; i--) {
      final m = state.messages[i];
      final text = m.text.trim();
      if (text.isEmpty) continue;
      result.add({
        'role': m.role == _AiRole.user ? 'user' : 'assistant',
        'content': text,
      });
    }
    return result.reversed.toList();
  }

  String _formatBytes(int bytes) {
    if (bytes < 1024) return '$bytes B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
}

/// Standardized prompt builder — keeps UI code clean.
String _buildSystemPrompt(MotivationStyle style, _AssistMode mode, String screenContext) {
  final modeInstruction = switch (mode) {
    _AssistMode.plan =>
      'Current mode: Daily Plan. Turn the user data into concrete study blocks, priorities, recovery guidance, and next actions. If the user asks a broad question, answer it through the lens of today and upcoming plan execution.',
    _AssistMode.doubt =>
      'Current mode: Doubt Solver. Solve the academic doubt step by step. Ask for missing question text, subject, or mark scheme only when required. Use attached file context when available.',
    _AssistMode.general =>
      'Current mode: General. Answer directly and use the best available app context.',
  };

  return '''
You are Axon, a precise academic study assistant.

Tone & Style:
- Be direct: Start with the answer or a high-level summary. Never repeat the user's query back.
- Never use conversational filler like "It looks like you're asking about..." or "To help you effectively..."
- Prioritize clarity and conciseness — if a concept can be explained in two sentences instead of five, do it.
- Use \$\$...\$\$ for display LaTeX and \$...\$ for inline. All math must use LaTeX.
- Keep line breaks generous for readability.

Answer the student's message directly and use the available context when it is relevant.
Do not invent exam dates, syllabus facts, marks, or user data. If context is missing, say what you need.
Keep answers concise by default, but give step-by-step working for academic doubts.
Motivation style: ${style.name}.
$modeInstruction

Context:
${screenContext.isEmpty ? 'No extra context supplied.' : screenContext}
''';
}

String _modeInstructionForUserPrompt(_AssistMode mode) {
  return switch (mode) {
    _AssistMode.plan =>
      'Use Daily Plan mode. Help me plan or adjust my study day using the app context.',
    _AssistMode.doubt =>
      'Use Doubt Solver mode. Solve this doubt carefully and show the working.',
    _AssistMode.general => '',
  };
}

// ─────────────────────────────────────────────────────────────────
// DATA CLASSES
// ─────────────────────────────────────────────────────────────────

enum _AiRole { assistant, user }
enum _AssistMode { general, plan, doubt }

class _AiMessage {
  final _AiRole role;
  final String text;
  const _AiMessage({required this.role, required this.text});

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is _AiMessage && role == other.role && text == other.text;

  @override
  int get hashCode => Object.hash(role, text);
}

class _AttachedFile {
  final String name;
  final String? path;
  final int size;
  final String extension;
  final String previewText;

  const _AttachedFile({
    required this.name,
    this.path,
    required this.size,
    required this.extension,
    this.previewText = '',
  });
}

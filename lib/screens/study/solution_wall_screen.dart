import 'dart:io';

import 'package:cached_network_image/cached_network_image.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../../models/models.dart';
import '../../services/cloudinary_service.dart';
import '../../services/peer_solution_service.dart';
import '../../theme/app_theme.dart';

class SolutionWallScreen extends StatefulWidget {
  const SolutionWallScreen({
    super.key,
    required this.questionId,
    required this.question,
    required this.sourceTitle,
  });

  final String questionId;
  final PdfQuestion question;
  final String sourceTitle;

  @override
  State<SolutionWallScreen> createState() => _SolutionWallScreenState();
}

class _SolutionWallScreenState extends State<SolutionWallScreen> {
  final PeerSolutionService _service = PeerSolutionService.instance;
  final ImagePicker _picker = ImagePicker();
  final CloudinaryService _cloudinary = CloudinaryService();
  final Map<String, int> _voteState = {};
  bool _submitting = false;

  @override
  Widget build(BuildContext context) {
    final currentUid = FirebaseAuth.instance.currentUser?.uid ?? '';
    return Scaffold(
      backgroundColor: const Color(0xFF050505),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('Solution Wall'),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _submitting ? null : _openComposer,
        icon: const Icon(Icons.draw_rounded),
        label: const Text('Add Solution'),
      ),
      body: StreamBuilder<List<PeerSolutionEntry>>(
        stream: _service.watchSolutions(widget.questionId),
        builder: (context, snapshot) {
          final entries = snapshot.data ?? const <PeerSolutionEntry>[];
          return ListView(
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 96),
            children: [
              _QuestionHeader(question: widget.question, sourceTitle: widget.sourceTitle),
              const SizedBox(height: 16),
              if (entries.isEmpty)
                _EmptyWall(onCompose: _openComposer)
              else
                for (final entry in entries) ...[
                  _SolutionCard(
                    entry: entry,
                    canDelete: entry.authorUid == currentUid,
                    currentVote: _voteState[entry.id] ?? 0,
                    onVote: (value) async {
                      await _service.vote(entry.id, value);
                      if (!mounted) return;
                      setState(() {
                        _voteState[entry.id] =
                            (_voteState[entry.id] == value) ? 0 : value;
                      });
                    },
                    onFlag: () => _service.flagSolution(entry.id),
                    onDelete: () => _service.deleteSolution(entry.id),
                  ),
                  const SizedBox(height: 12),
                ],
            ],
          );
        },
      ),
    );
  }

  Future<void> _openComposer() async {
    final summaryController = TextEditingController();
    final explanationController = TextEditingController();
    File? selectedImage;

    final shouldSubmit = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setModalState) {
            Future<void> pickImage(ImageSource source) async {
              final file = await _picker.pickImage(
                source: source,
                imageQuality: 90,
              );
              if (file == null) return;
              setModalState(() => selectedImage = File(file.path));
            }

            return Padding(
              padding: EdgeInsets.fromLTRB(
                20,
                20,
                20,
                MediaQuery.of(context).viewInsets.bottom + 20,
              ),
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Upload a clean logic solution',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: summaryController,
                      decoration: const InputDecoration(
                        labelText: 'Logic summary',
                        hintText: 'State the key idea in one line',
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: explanationController,
                      minLines: 5,
                      maxLines: 9,
                      decoration: const InputDecoration(
                        labelText: 'Elegant explanation',
                        hintText: 'Explain why this method works better than a raw mark scheme line list.',
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        OutlinedButton.icon(
                          onPressed: () => pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt_outlined),
                          label: const Text('Camera'),
                        ),
                        const SizedBox(width: 12),
                        OutlinedButton.icon(
                          onPressed: () => pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_outlined),
                          label: const Text('Gallery'),
                        ),
                      ],
                    ),
                    if (selectedImage != null) ...[
                      const SizedBox(height: 12),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.file(selectedImage!, height: 180, fit: BoxFit.cover),
                      ),
                    ],
                    const SizedBox(height: 20),
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                        onPressed: () {
                          if (summaryController.text.trim().isEmpty ||
                              explanationController.text.trim().isEmpty) {
                            return;
                          }
                          Navigator.of(context).pop(true);
                        },
                        child: const Text('Publish'),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );

    if (shouldSubmit != true) return;
    setState(() => _submitting = true);
    try {
      var imageUrl = '';
      var imagePublicId = '';
      if (selectedImage != null) {
        final upload = await _cloudinary.uploadStudyArtifact(selectedImage!);
        if (upload != null) {
          imageUrl = (upload['secure_url'] ?? '').toString();
          imagePublicId = (upload['public_id'] ?? '').toString();
        }
      }
      await _service.submitSolution(
        questionId: widget.questionId,
        questionText: widget.question.fullText,
        subject: widget.question.subjectTag,
        chapter: widget.question.chapterTag,
        topic: widget.question.topicTag,
        sourceTitle: widget.sourceTitle,
        logicSummary: summaryController.text,
        explanation: explanationController.text,
        imageUrl: imageUrl,
        imagePublicId: imagePublicId,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Solution published to the wall.')),
      );
    } finally {
      if (mounted) {
        setState(() => _submitting = false);
      }
      summaryController.dispose();
      explanationController.dispose();
    }
  }
}

class _QuestionHeader extends StatelessWidget {
  const _QuestionHeader({
    required this.question,
    required this.sourceTitle,
  });

  final PdfQuestion question;
  final String sourceTitle;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.24),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            sourceTitle,
            style: const TextStyle(color: Colors.white54, fontSize: 12),
          ),
          const SizedBox(height: 6),
          Text(
            'Q${question.questionNumber}',
            style: const TextStyle(
              color: Color(0xFF3A86FF),
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            question.fullText,
            style: const TextStyle(color: Colors.white, fontSize: 15),
          ),
        ],
      ),
    );
  }
}

class _EmptyWall extends StatelessWidget {
  const _EmptyWall({required this.onCompose});

  final VoidCallback onCompose;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        children: [
          const Icon(Icons.lightbulb_outline_rounded, color: Colors.white38, size: 36),
          const SizedBox(height: 12),
          const Text(
            'No elegant solutions uploaded yet.',
            style: TextStyle(color: Colors.white, fontSize: 16),
          ),
          const SizedBox(height: 8),
          const Text(
            'Add a clean logic walkthrough so the next student gets the why, not just the marking points.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white54),
          ),
          const SizedBox(height: 16),
          FilledButton(
            onPressed: onCompose,
            child: const Text('Upload First Solution'),
          ),
        ],
      ),
    );
  }
}

class _SolutionCard extends StatelessWidget {
  const _SolutionCard({
    required this.entry,
    required this.canDelete,
    required this.currentVote,
    required this.onVote,
    required this.onFlag,
    required this.onDelete,
  });

  final PeerSolutionEntry entry;
  final bool canDelete;
  final int currentVote;
  final ValueChanged<int> onVote;
  final VoidCallback onFlag;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      entry.logicSummary,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${entry.authorName} • ${entry.voteScore} votes',
                      style: const TextStyle(color: Colors.white54, fontSize: 12),
                    ),
                  ],
                ),
              ),
              IconButton(
                onPressed: onFlag,
                icon: const Icon(Icons.flag_outlined, color: Colors.white38),
              ),
              if (canDelete)
                IconButton(
                  onPressed: onDelete,
                  icon: const Icon(Icons.delete_outline_rounded, color: Colors.white38),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            entry.explanation,
            style: const TextStyle(color: Colors.white70, height: 1.45),
          ),
          if (entry.imageUrl.isNotEmpty) ...[
            const SizedBox(height: 12),
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: CachedNetworkImage(
                imageUrl: entry.imageUrl,
                height: 220,
                width: double.infinity,
                fit: BoxFit.cover,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              IconButton(
                onPressed: () => onVote(1),
                icon: Icon(
                  currentVote == 1
                      ? Icons.thumb_up_alt_rounded
                      : Icons.thumb_up_alt_outlined,
                  color: currentVote == 1 ? const Color(0xFF3A86FF) : Colors.white54,
                ),
              ),
              IconButton(
                onPressed: () => onVote(-1),
                icon: Icon(
                  currentVote == -1
                      ? Icons.thumb_down_alt_rounded
                      : Icons.thumb_down_alt_outlined,
                  color: currentVote == -1 ? Colors.orange : Colors.white54,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

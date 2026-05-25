// lib/screens/university/achievement_upload_screen.dart

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../../models/university.dart';
import '../../services/university_service.dart';
import '../../theme/app_theme.dart';

class AchievementUploadScreen extends StatefulWidget {
  const AchievementUploadScreen({super.key});

  @override
  State<AchievementUploadScreen> createState() => _AchievementUploadScreenState();
}

class _AchievementUploadScreenState extends State<AchievementUploadScreen> {
  final _titleController = TextEditingController();
  final _descController = TextEditingController();
  final _uniService = UniversityService();
  
  bool _isAnalyzing = false;
  UserAchievement? _result;

  Future<void> _analyze() async {
    final title = _titleController.text.trim();
    final desc = _descController.text.trim();
    if (title.isEmpty || desc.isEmpty) return;

    setState(() {
      _isAnalyzing = true;
      _result = null;
    });

    final result = await _uniService.rateAchievement(title, desc);

    setState(() {
      _result = result;
      _isAnalyzing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        title: const Text('Add Achievement'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('What did you achieve?', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('Describe your award, project, or role in detail.', style: TextStyle(color: AxonColors.textSecondary, fontSize: 14)),
            const SizedBox(height: 24),
            _buildTextField('Title', 'e.g., International Math Olympiad Silver', _titleController),
            const SizedBox(height: 20),
            _buildTextField('Description', 'Provide context, scope, and your specific contribution...', _descController, maxLines: 5),
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              height: 56,
              child: ElevatedButton(
                onPressed: _isAnalyzing ? null : _analyze,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF3A86FF),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  disabledBackgroundColor: Colors.white10,
                ),
                child: _isAnalyzing
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                    : const Text('Analyze with AI', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
              ),
            ),
            if (_result != null) _buildResultCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildTextField(String label, String hint, TextEditingController controller, {int maxLines = 1}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 14, fontWeight: FontWeight.w500)),
        const SizedBox(height: 8),
        TextField(
          controller: controller,
          maxLines: maxLines,
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            hintText: hint,
            hintStyle: TextStyle(color: AxonColors.textTertiary, fontSize: 14),
            filled: true,
            fillColor: AxonColors.surface,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(16), borderSide: BorderSide.none),
            contentPadding: const EdgeInsets.all(16),
          ),
        ),
      ],
    );
  }

  Widget _buildResultCard() {
    final rating = (_result!.importanceRating * 10).toStringAsFixed(1);
    final color = _result!.importanceRating > 0.7 ? Colors.greenAccent : _result!.importanceRating > 0.4 ? Colors.orangeAccent : Colors.blueAccent;

    return Container(
      margin: const EdgeInsets.only(top: 32),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(color: color.withValues(alpha: 0.1), borderRadius: BorderRadius.circular(8)),
                child: Text('Rating: $rating/10', style: TextStyle(color: color, fontWeight: FontWeight.bold, fontSize: 14)),
              ),
              const Spacer(),
              const Icon(Icons.auto_awesome, color: Colors.amber, size: 20),
            ],
          ),
          const SizedBox(height: 16),
          const Text('Expert Feedback', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          Text(_result!.feedback, style: TextStyle(color: AxonColors.textSecondary, height: 1.5, fontSize: 14)),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: () async {
                final prefs = await SharedPreferences.getInstance();
                final list = prefs.getStringList('achievements') ?? [];
                list.add(jsonEncode({
                  'title': _titleController.text,
                  'description': _descController.text,
                  'date': DateTime.now().toIso8601String(),
                  'rating': _result!.importanceRating,
                }));
                await prefs.setStringList('achievements', list);
                if (mounted) Navigator.pop(context, true);
              },
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: Colors.white10),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
              child: const Text('Save to Profile', style: TextStyle(color: Colors.white)),
            ),
          ),
        ],
      ),
    );
  }
}

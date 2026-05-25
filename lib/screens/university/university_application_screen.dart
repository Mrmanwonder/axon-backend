// lib/screens/university/university_application_screen.dart

import 'package:flutter/material.dart';
import '../../models/university.dart';
import '../../services/university_service.dart';
import '../../theme/app_theme.dart';
import 'achievement_upload_screen.dart';
import 'university_search_screen.dart';

class UniversityApplicationScreen extends StatefulWidget {
  const UniversityApplicationScreen({super.key});

  @override
  State<UniversityApplicationScreen> createState() => _UniversityApplicationScreenState();
}

class _UniversityApplicationScreenState extends State<UniversityApplicationScreen> {
  final UniversityService _uniService = UniversityService();
  List<UserUniversityPreference> _userPrefs = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() => _isLoading = true);
    final prefs = await _uniService.getPreferences();
    setState(() {
      _userPrefs = prefs;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        title: const Text('University Applications'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.add_circle_outline),
            onPressed: () async {
              await Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const UniversitySearchScreen()),
              );
              _loadData();
            },
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildSectionHeader('Dream Universities', UniversityCategory.dream),
                  _buildUniList(UniversityCategory.dream),
                  const SizedBox(height: 24),
                  _buildSectionHeader('Reach Universities', UniversityCategory.reach),
                  _buildUniList(UniversityCategory.reach),
                  const SizedBox(height: 24),
                  _buildSectionHeader('Safety Universities', UniversityCategory.safety),
                  _buildUniList(UniversityCategory.safety),
                  const SizedBox(height: 32),
                  _buildAchievementSection(),
                ],
              ),
            ),
    );
  }

  Widget _buildSectionHeader(String title, UniversityCategory category) {
    Color color = category == UniversityCategory.dream 
        ? Colors.purpleAccent 
        : category == UniversityCategory.reach 
            ? Colors.orangeAccent 
            : Colors.greenAccent;

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            width: 4,
            height: 20,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUniList(UniversityCategory category) {
    final filtered = _userPrefs.where((p) => p.category == category).toList();
    if (filtered.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
        ),
        child: Center(
          child: Text(
            'No universities added yet.',
            style: TextStyle(color: AxonColors.textSecondary, fontSize: 14),
          ),
        ),
      );
    }

    return Column(
      children: filtered.map((pref) {
        final uni = _uniService.getUniversityById(pref.universityId);
        if (uni == null) return const SizedBox.shrink();
        final degree = uni.degrees.firstWhere((d) => d.id == pref.degreeId);

        return Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AxonColors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
          ),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.network(uni.logoUrl, fit: BoxFit.contain, errorBuilder: (_, __, ___) => const Icon(Icons.school, color: Colors.grey)),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(uni.name, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                    const SizedBox(height: 2),
                    Text(degree.name, style: TextStyle(color: AxonColors.textSecondary, fontSize: 14)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right, color: Colors.grey),
            ],
          ),
        );
      }).toList(),
    );
  }

  Widget _buildAchievementSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Achievements & Profile',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white),
        ),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF3A86FF), Color(0xFF006DFF)],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            children: [
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Boost Your Application', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18)),
                    SizedBox(height: 4),
                    Text('Upload your achievements and get AI-powered importance ratings.', style: TextStyle(color: Colors.white70, fontSize: 13)),
                  ],
                ),
              ),
              ElevatedButton(
                onPressed: () async {
                  final result = await Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const AchievementUploadScreen()),
                  );
                  if (result == true) _loadData();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.blue,
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Upload'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

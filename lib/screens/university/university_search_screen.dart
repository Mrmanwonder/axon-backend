// lib/screens/university/university_search_screen.dart

import 'package:flutter/material.dart';
import '../../models/university.dart';
import '../../services/university_service.dart';
import '../../theme/app_theme.dart';

class UniversitySearchScreen extends StatefulWidget {
  const UniversitySearchScreen({super.key});

  @override
  State<UniversitySearchScreen> createState() => _UniversitySearchScreenState();
}

class _UniversitySearchScreenState extends State<UniversitySearchScreen> {
  final UniversityService _uniService = UniversityService();
  final TextEditingController _searchController = TextEditingController();
  List<University> _searchResults = [];
  bool _isSearching = false;

  @override
  void initState() {
    super.initState();
    _loadSuggestions();
  }

  Future<void> _loadSuggestions() async {
    final suggestions = await _uniService.suggestUniversities();
    setState(() {
      _searchResults = suggestions;
    });
  }

  Future<void> _onSearch(String query) async {
    if (query.isEmpty) {
      _loadSuggestions();
      return;
    }
    setState(() => _isSearching = true);
    final results = await _uniService.searchUniversities(query);
    setState(() {
      _searchResults = results;
      _isSearching = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        title: const Text('Find Universities'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: TextField(
              controller: _searchController,
              onChanged: _onSearch,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                hintText: 'Search by name or location...',
                hintStyle: TextStyle(color: AxonColors.textTertiary),
                prefixIcon: const Icon(Icons.search, color: Colors.grey),
                filled: true,
                fillColor: AxonColors.surface,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
          ),
          Expanded(
            child: _isSearching
                ? const Center(child: CircularProgressIndicator())
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: _searchResults.length,
                    itemBuilder: (context, index) {
                      final uni = _searchResults[index];
                      return _buildUniversityCard(uni);
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildUniversityCard(University uni) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: InkWell(
        onTap: () => _showUniversityDetails(uni),
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(14),
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
                    Text(uni.location, style: TextStyle(color: AxonColors.textSecondary, fontSize: 13)),
                  ],
                ),
              ),
              const Icon(Icons.arrow_forward_ios, color: Colors.grey, size: 16),
            ],
          ),
        ),
      ),
    );
  }

  void _showUniversityDetails(University uni) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _UniversityDetailSheet(uni: uni),
    );
  }
}

class _UniversityDetailSheet extends StatefulWidget {
  final University uni;
  const _UniversityDetailSheet({required this.uni});

  @override
  State<_UniversityDetailSheet> createState() => _UniversityDetailSheetState();
}

class _UniversityDetailSheetState extends State<_UniversityDetailSheet> {
  Degree? _selectedDegree;
  UniversityCategory _selectedCategory = UniversityCategory.reach;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: MediaQuery.of(context).size.height * 0.85,
      decoration: BoxDecoration(
        color: AxonColors.background,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
      ),
      child: Column(
        children: [
          const SizedBox(height: 12),
          Container(width: 40, height: 4, decoration: BoxDecoration(color: Colors.white24, borderRadius: BorderRadius.circular(2))),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 64, height: 64,
                        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: Image.network(widget.uni.logoUrl, fit: BoxFit.contain, errorBuilder: (_, __, ___) => const Icon(Icons.school)),
                        ),
                      ),
                      const SizedBox(width: 16),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(widget.uni.name, style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
                            Text(widget.uni.location, style: TextStyle(color: AxonColors.textSecondary, fontSize: 14)),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  const Text('About', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                  const SizedBox(height: 8),
                  Text(widget.uni.description, style: TextStyle(color: AxonColors.textSecondary, height: 1.5)),
                  const SizedBox(height: 24),
                  const Text('Select Degree', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                  const SizedBox(height: 12),
                  ...widget.uni.degrees.map((d) => _buildDegreeOption(d)),
                  if (_selectedDegree != null) ...[
                    const SizedBox(height: 24),
                    _buildDegreeDetails(),
                    const SizedBox(height: 24),
                    const Text('Category', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 16)),
                    const SizedBox(height: 12),
                    _buildCategorySelector(),
                  ],
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
          if (_selectedDegree != null)
            Padding(
              padding: const EdgeInsets.all(24),
              child: SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: () async {
                    await UniversityService().addToPreferences(UserUniversityPreference(
                      universityId: widget.uni.id,
                      degreeId: _selectedDegree!.id,
                      category: _selectedCategory,
                      addedAt: DateTime.now(),
                    ));
                    if (context.mounted) Navigator.pop(context);
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF3A86FF),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                  ),
                  child: const Text('Add to Applications', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildDegreeOption(Degree d) {
    final isSelected = _selectedDegree?.id == d.id;
    return GestureDetector(
      onTap: () => setState(() => _selectedDegree = d),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF3A86FF).withValues(alpha: 0.1) : AxonColors.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: isSelected ? const Color(0xFF3A86FF) : Colors.white.withValues(alpha: 0.05)),
        ),
        child: Row(
          children: [
            Expanded(child: Text(d.name, style: TextStyle(color: isSelected ? const Color(0xFF3A86FF) : Colors.white, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal))),
            if (isSelected) const Icon(Icons.check_circle, color: Color(0xFF3A86FF), size: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildDegreeDetails() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildDetailItem(Icons.timer_outlined, 'Duration', _selectedDegree!.duration),
          const Divider(color: Colors.white10, height: 24),
          _buildDetailItem(Icons.grade_outlined, 'Grade Requirements', _selectedDegree!.gradeRequirements.entries.map((e) => '${e.key}: ${e.value}').join(', ')),
          const Divider(color: Colors.white10, height: 24),
          _buildDetailItem(Icons.book_outlined, 'Required Subjects', _selectedDegree!.requiredSubjects.join(', ')),
          const Divider(color: Colors.white10, height: 24),
          _buildDetailItem(Icons.list_alt_outlined, 'Syllabus Focus', _selectedDegree!.syllabus),
        ],
      ),
    );
  }

  Widget _buildDetailItem(IconData icon, String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: const Color(0xFF3A86FF), size: 20),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: TextStyle(color: AxonColors.textSecondary, fontSize: 12)),
              const SizedBox(height: 4),
              Text(value, style: const TextStyle(color: Colors.white, fontSize: 14, height: 1.4)),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildCategorySelector() {
    return Row(
      children: [
        _buildCatOption('Dream', UniversityCategory.dream, Colors.purpleAccent),
        const SizedBox(width: 10),
        _buildCatOption('Reach', UniversityCategory.reach, Colors.orangeAccent),
        const SizedBox(width: 10),
        _buildCatOption('Safety', UniversityCategory.safety, Colors.greenAccent),
      ],
    );
  }

  Widget _buildCatOption(String label, UniversityCategory cat, Color color) {
    final isSelected = _selectedCategory == cat;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _selectedCategory = cat),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? color.withValues(alpha: 0.15) : AxonColors.surface,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: isSelected ? color : Colors.white.withValues(alpha: 0.05)),
          ),
          child: Center(
            child: Text(label, style: TextStyle(color: isSelected ? color : AxonColors.textSecondary, fontWeight: isSelected ? FontWeight.bold : FontWeight.normal, fontSize: 13)),
          ),
        ),
      ),
    );
  }
}

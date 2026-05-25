import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:go_router/go_router.dart';

import '../../models/models.dart';
import '../../services/app_state.dart';
import '../../services/resource_crawler_service.dart';
import '../../services/exam_zone_service.dart';
import '../../services/google_drive_downloader.dart';
import '../../theme/app_theme.dart';
import '../../widgets/common/axon_widgets.dart';

class OnboardingScreen extends ConsumerStatefulWidget {
  const OnboardingScreen({super.key});

  @override
  ConsumerState<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends ConsumerState<OnboardingScreen> {
  static const Map<String, String> _boards = {
    'CAIE': 'Cambridge (CAIE)',
    'IB': 'International Baccalaureate (IB)',
    'Edexcel': 'Edexcel',
  };

  static const Map<String, List<String>> _levelOptions = {
    'CAIE': ['IGCSE', 'AS and A Level', 'O Level'],
    'IB': ['IB DP', 'IB CP', 'IB MYP'],
    'Edexcel': ['IGCSE', 'AS and A Level'],
  };

  static const Map<String, List<String>> _subjectsByLevel = {
    'CAIE-IGCSE': [
      'Mathematics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Economics',
      'English',
      'History',
      'Geography',
      'Business Studies',
      'Accounting',
      'Art and Design',
      'Music',
      'Drama',
      'French',
      'German',
      'Spanish',
      'Chinese',
      'Hindi',
      'Thinking Skills',
      'Psychology',
      'Environmental Management',
      'Information Technology',
      'Travel and Tourism',
      'Sociology',
      'Religious Studies',
      'Global Perspectives',
      'Food and Nutrition',
      'Design and Technology',
      'Physical Education',
      'ICT',
      'English as a Second Language',
    ],
    'CAIE-AS and A Level': [
      'Mathematics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Economics',
      'Business Studies',
      'Accounting',
      'English',
      'English Literature',
      'History',
      'Geography',
      'Psychology',
      'Sociology',
      'French',
      'German',
      'Spanish',
      'Chinese',
      'Arabic',
      'Hindi',
      'Art and Design',
      'Music',
      'Drama',
      'Media Studies',
      'Design and Technology',
      'Information Technology',
      'Thinking Skills',
      'Law',
      'Philosophy',
      'Marine Science',
      'Environmental Management',
      'Travel and Tourism',
      'Physical Education',
      'Hinduism',
      'Islamic Studies',
      'Divinity',
      'Biblical Studies',
      'Classical Studies',
      'General Paper',
    ],
    'CAIE-O Level': [
      'Mathematics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Economics',
      'Business Studies',
      'Accounting',
      'English Language',
      'Literature in English',
      'History',
      'Geography',
      'French',
      'German',
      'Spanish',
      'Arabic',
      'Hindi',
      'Urdu',
      'Bengali',
      'Nepali',
      'Art',
      'Art and Design',
      'Design and Technology',
      'Food and Nutrition',
      'Religious Studies',
      'Islamiyat',
      'Global Perspectives',
      'Sociology',
      'Statistics',
      'Commerce',
      'Environmental Management',
      'Agriculture',
    ],
    'IB-IB DP': [
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Mathematics',
      'Economics',
      'Business Management',
      'History',
      'Geography',
      'Psychology',
      'Philosophy',
      'English A',
      'English B',
      'French B',
      'Spanish B',
      'Chinese B',
      'Visual Arts',
      'Music',
      'Theatre',
      'Film',
      'Environmental Systems and Societies',
      'Sports Exercise and Health Science',
      'Design Technology',
      'Information Technology in a Global Society',
      'Social and Cultural Anthropology',
      'World Politics',
      'Mathematics Analysis and Approaches',
      'Mathematics Applications and Interpretation',
      'Literature and Performance',
      'Dance',
    ],
    'IB-IB CP': [
      'Business Management',
      'Applied Psychology',
      'Environmental Systems',
      'Health and Social Care',
      'Sport and Exercise Science',
      'Media',
      'Arts and Design',
      'Information Technology',
      'Language Acquisition',
    ],
    'IB-IB MYP': [
      'Language and Literature',
      'Language Acquisition',
      'Individuals and Societies',
      'Sciences',
      'Mathematics',
      'Arts',
      'Physical and Health Education',
      'Design',
      'Digital Societies',
      'Interdisciplinary',
    ],
    'Edexcel-IGCSE': [
      'Mathematics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Economics',
      'Business Studies',
      'English Language A',
      'English Language B',
      'English Literature',
      'French',
      'German',
      'Spanish',
      'Geography',
      'History',
      'ICT',
      'Art and Design',
    ],
    'Edexcel-AS and A Level': [
      'Mathematics',
      'Physics',
      'Chemistry',
      'Biology',
      'Computer Science',
      'Economics',
      'Business',
      'Geography',
      'History',
      'Psychology',
      'English',
      'French',
      'German',
      'Spanish',
      'Art and Design',
      'Music',
      'Drama',
      'Media Studies',
      'Law',
      'Accounting',
      'Finance',
    ],
  };

  List<String> get _currentSubjects {
    final key = '$_board-$_level';
    return _subjectsByLevel[key] ?? _subjectsByLevel['CAIE-IGCSE']!;
  }

  final _resourceService = ResourceCrawlerService();
  final _downloadService = GoogleDriveDownloader.instance;
  String _board = 'CAIE';
  String _level = 'IGCSE';
  String _country = 'Pakistan';
  String _timezone = 'PKT (UTC+5)';
  List<String> _availableSeries = ['March', 'June', 'November'];
  final Set<String> _selectedSubjects = {'Mathematics', 'Physics'};
  double _targetHours = 4;
  MotivationStyle _motivationStyle = MotivationStyle.logicBased;
  bool _saving = false;
  String? _error;

  void _onCountryChanged(String country) {
    final zone = ExamZoneService.getZone(country);
    setState(() {
      _country = country;
      _timezone = zone?['timezone'] ?? 'UTC';
      _availableSeries =
          List<String>.from(zone?['series'] ?? ['June', 'November']);
    });
  }

  Future<void> _save() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      setState(() => _error = 'Authentication expired. Sign in again.');
      return;
    }
    if (_selectedSubjects.isEmpty) {
      setState(() => _error = 'Select at least one subject.');
      return;
    }

    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      final profileService = ref.read(profileServiceProvider);
      final authNotifier = ref.read(authStateProvider.notifier);

      // Store combined board+level for crawler (e.g., "CAIE-IGCSE")
      final combinedBoard = '$_board-$_level';

      final savedProfile = await profileService
          .saveOnboarding(
            user: user,
            displayName:
                user.displayName ?? user.email?.split('@').first ?? 'Student',
            board:
                combinedBoard, // e.g., "CAIE-IGCSE", "IB-DP", "Edexcel-AS and A Level"
            subjects: _selectedSubjects.toList(),
            targetHours: _targetHours,
            motivationStyle: _motivationStyle,
            country: _country,
            timezone: _timezone,
            examSeries: _availableSeries,
          )
          .timeout(const Duration(seconds: 12));
      await authNotifier.applyOnboardingProfile(savedProfile);
      await authNotifier.updateMotivationStyle(_motivationStyle);
      ref.read(metricsProvider.notifier).updateTargetStudyHours(_targetHours);
      if (_selectedSubjects.isNotEmpty) {
        ref
            .read(metricsProvider.notifier)
            .updateSubject(_selectedSubjects.first);
      }
      await _resourceService
          .scheduleInitialCrawl(
            uid: user.uid,
            board: _board,
            subjects: _selectedSubjects.toList(),
          )
          .timeout(const Duration(seconds: 8));

      try {
        await _downloadService.initialize();
        if (_country.isNotEmpty) {
          _downloadService.downloadPastPapersForSubjects(
            _selectedSubjects.toList(),
            onProgress: (subject, current, total) {
              debugPrint('Downloading $subject ($current/$total)');
            },
          );
        }
      } catch (e) {
        debugPrint('Past papers download failed: $e');
      }
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Profile saved. Axon is preparing resources in the background.',
            style: GoogleFonts.googleSans(color: Colors.white),
          ),
          backgroundColor: AxonColors.accent,
        ),
      );
      context.go('/home');
    } catch (e) {
      setState(() => _error = 'Failed to save onboarding profile.');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      body: Container(
        color: AxonColors.oxfordBlueDark,
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(24, 24, 24, 40),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Prepare the lab.',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 34,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -1.2,
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  'Axon needs your board, subjects, and study capacity before it unlocks the dashboard.',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary,
                    fontSize: 13,
                    height: 1.5,
                  ),
                ),
                const SizedBox(height: 24),
                if (_error != null) ...[
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AxonColors.error.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                          color: AxonColors.error.withValues(alpha: 0.3)),
                    ),
                    child: Text(
                      _error!,
                      style: GoogleFonts.googleSans(
                          color: AxonColors.error, fontSize: 13),
                    ),
                  ),
                  const SizedBox(height: 16),
                ],
                AxonCard(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _SectionTitle(label: 'Exam Board'),
                      const SizedBox(height: 12),
                      Wrap(
                        spacing: 10,
                        runSpacing: 10,
                        children: _boards.entries.map((entry) {
                          final selected = entry.key == _board;
                          return _SelectChip(
                            label: entry.value,
                            selected: selected,
                            onTap: () => setState(() {
                              _board = entry.key;
                              _level = _levelOptions[_board]?.first ?? 'IGCSE';
                              _selectedSubjects.clear();
                              _selectedSubjects
                                  .addAll(_currentSubjects.take(2));
                            }),
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 22),
                      _SectionTitle(label: 'Level'),
                      const SizedBox(height: 12),
                      Wrap(
                        spacing: 10,
                        runSpacing: 10,
                        children: (_levelOptions[_board] ?? []).map((level) {
                          final selected = level == _level;
                          return _SelectChip(
                            label: level,
                            selected: selected,
                            onTap: () => setState(() {
                              _level = level;
                              _selectedSubjects.clear();
                              _selectedSubjects
                                  .addAll(_currentSubjects.take(2));
                            }),
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 22),
                      _SectionTitle(label: 'Country (for exam timeline)'),
                      const SizedBox(height: 12),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 14),
                        decoration: BoxDecoration(
                          color: AxonColors.surfaceElevated,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: AxonColors.divider),
                        ),
                        child: DropdownButtonHideUnderline(
                          child: DropdownButton<String>(
                            value: _country,
                            isExpanded: true,
                            dropdownColor: AxonColors.surfaceElevated,
                            style: GoogleFonts.googleSans(color: Colors.white),
                            icon: Icon(Icons.keyboard_arrow_down,
                                color: AxonColors.textSecondary),
                            items: ExamZoneService.countryList.map((country) {
                              return DropdownMenuItem(
                                value: country,
                                child: Text(country,
                                    style: GoogleFonts.googleSans(
                                        color: Colors.white)),
                              );
                            }).toList(),
                            onChanged: (value) {
                              if (value != null) _onCountryChanged(value);
                            },
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color:
                              AxonColors.electricCyan.withValues(alpha: 0.08),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                              color: AxonColors.electricCyan
                                  .withValues(alpha: 0.2)),
                        ),
                        child: Row(
                          children: [
                            Icon(Icons.schedule,
                                size: 16, color: AxonColors.electricCyan),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Timezone: $_timezone | Series: ${_availableSeries.join(", ")}',
                                style: GoogleFonts.googleSans(
                                    color: AxonColors.electricCyan,
                                    fontSize: 12),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 22),
                      _SectionTitle(label: 'Subjects'),
                      const SizedBox(height: 8),
                      Text(
                        'These seed your study catalog, calendar weights, and background resource crawler.',
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textTertiary, fontSize: 12),
                      ),
                      const SizedBox(height: 12),
                      Wrap(
                        spacing: 10,
                        runSpacing: 10,
                        children: _currentSubjects.map((subject) {
                          final selected = _selectedSubjects.contains(subject);
                          return _SelectChip(
                            label: subject,
                            selected: selected,
                            onTap: () => setState(() {
                              if (selected) {
                                _selectedSubjects.remove(subject);
                              } else {
                                _selectedSubjects.add(subject);
                              }
                            }),
                          );
                        }).toList(),
                      ),
                      const SizedBox(height: 22),
                      _SectionTitle(label: 'Target Study Hours'),
                      const SizedBox(height: 8),
                      Text(
                        (_targetHours % 1).abs() < 0.001
                            ? '${_targetHours.toInt()} hours per day'
                            : '${_targetHours.toStringAsFixed(1)} hours per day',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.electricCyan,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      AxonSpatialSlider(
                        label: 'Target Study Hours',
                        value: ((_targetHours - 1) / 7).clamp(0.0, 1.0),
                        valueFormatter: (_) => (_targetHours % 1).abs() < 0.001
                            ? '${_targetHours.toInt()}h'
                            : '${_targetHours.toStringAsFixed(1)}h',
                        onChanged: (value) => setState(() =>
                            _targetHours = ((1 + (value * 7)) * 2).round() / 2),
                      ),
                      const SizedBox(height: 12),
                      _SectionTitle(label: 'Motivation Style'),
                      const SizedBox(height: 12),
                      ...MotivationStyle.values
                          .map((style) => _MotivationOption(
                                style: style,
                                selected: style == _motivationStyle,
                                onTap: () =>
                                    setState(() => _motivationStyle = style),
                              )),
                    ],
                  ),
                ),
                const SizedBox(height: 22),
                CyberButton(
                  label: 'Save And Continue',
                  onTap: _save,
                  icon: Icons.arrow_forward_rounded,
                  isLoading: _saving,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String label;
  const _SectionTitle({required this.label});

  @override
  Widget build(BuildContext context) {
    return Text(
      label.toUpperCase(),
      style: GoogleFonts.googleSans(
        color: AxonColors.textTertiary,
        fontSize: 11,
        fontWeight: FontWeight.w700,
        letterSpacing: 1.4,
      ),
    );
  }
}

class _SelectChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _SelectChip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: selected
              ? AxonColors.electricCyan.withValues(alpha: 0.12)
              : AxonColors.surfaceElevated,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: selected ? AxonColors.electricCyan : AxonColors.divider,
          ),
        ),
        child: Text(
          label,
          style: GoogleFonts.googleSans(
            color: selected ? AxonColors.electricCyan : AxonColors.textPrimary,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _MotivationOption extends StatelessWidget {
  final MotivationStyle style;
  final bool selected;
  final VoidCallback onTap;

  const _MotivationOption({
    required this.style,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: selected
              ? AxonColors.accent.withValues(alpha: 0.08)
              : AxonColors.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: selected ? AxonColors.accent : AxonColors.divider,
          ),
        ),
        child: Row(
          children: [
            Text(style.icon, style: const TextStyle(fontSize: 18)),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    style.displayName,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 3),
                  Text(
                    style.description,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary,
                      fontSize: 12,
                      height: 1.45,
                    ),
                  ),
                ],
              ),
            ),
            if (selected)
              Icon(Icons.check_circle_rounded,
                  color: AxonColors.accent, size: 18),
          ],
        ),
      ),
    );
  }
}

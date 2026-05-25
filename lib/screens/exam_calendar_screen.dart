import 'dart:async';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:table_calendar/table_calendar.dart';
import 'package:intl/intl.dart';

import '../models/exam_event_model.dart';
import '../theme/app_theme.dart';
import '../services/app_state.dart';
import '../services/exam_repository.dart';
import '../models/daily_plan_task.dart';
import '../widgets/common/daily_plan_panel.dart';

class ExamCalendarScreen extends ConsumerStatefulWidget {
  const ExamCalendarScreen({super.key});

  @override
  ConsumerState<ExamCalendarScreen> createState() => _ExamCalendarScreenState();
}

class _ExamCalendarScreenState extends ConsumerState<ExamCalendarScreen> {
  DateTime _focusedDay = DateTime.now();
  DateTime? _selectedDay;
  CalendarFormat _calendarFormat = CalendarFormat.month;

  List<ExamEventModel> _exams = [];
  bool _isLoading = true;
  String? _userId;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData({bool forceRefresh = false}) async {
    if (!mounted) return;
    print(
        'ExamCalendarScreen: _loadData starting (forceRefresh: $forceRefresh)');
    setState(() => _isLoading = true);

    final auth = ref.read(authStateProvider);
    final user = auth.user;
    if (user == null) {
      print('ExamCalendarScreen: User is null, aborting load');
      setState(() => _isLoading = false);
      return;
    }

    _userId = user.uid;

    try {
      await ExamRepository.instance.initialize(userId: _userId!);

      debugPrint(
          'ExamCalendarScreen: Syncing exams for board: ${user.board}, subjects: ${user.subjects}');

      final supabaseExams = await ExamRepository.instance
          .syncSupabaseExamDates(
        curriculum: user.board,
        subjects: user.subjects,
      )
          .timeout(
        const Duration(seconds: 60),
        onTimeout: () {
          debugPrint('ExamCalendarScreen: Supabase sync timed out after 60s');
          return const <ExamEventModel>[];
        },
      );

      // Generate daily plan after exam sync (so deadlines exist)
      unawaited(
          ref.read(dailyPlanServiceProvider).ensureTodayPlan(_userId!).timeout(
        const Duration(seconds: 35),
        onTimeout: () {
          debugPrint('ExamCalendarScreen: Daily plan timed out');
          return const <DailyPlanTask>[];
        },
      ).catchError((error) {
        debugPrint('Daily plan generation failed: $error');
        return const <DailyPlanTask>[];
      }));

      final exams = supabaseExams.isNotEmpty
          ? supabaseExams
          : ExamRepository.instance.getUpcomingExams();

      debugPrint('ExamCalendarScreen: Loaded ${exams.length} exams');

      if (mounted) {
        setState(() {
          _exams = exams;
          if (_exams.isNotEmpty && _selectedDay == null) {
            // Only focus if we don't have a selection
            _focusedDay = _exams.first.date;
          }
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint('Exam timeline load failed: $e');
      if (mounted) {
        final cachedExams = ExamRepository.instance.getUpcomingExams();
        setState(() {
          _exams = cachedExams;
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF050505),
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          _buildSliverHeader(),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 12),
                  _buildMonthlyCalendar(),
                  const SizedBox(height: 24),
                  _buildDailyPlanPanel(),
                  const SizedBox(height: 32),
                  Text(
                    "UPCOMING_EXAMS",
                    style: GoogleFonts.robotoMono(
                      color: Colors.white24,
                      fontSize: 10,
                      letterSpacing: 2,
                    ),
                  ),
                  const SizedBox(height: 16),
                  _isLoading
                      ? const Center(
                          child: Padding(
                            padding: EdgeInsets.all(40),
                            child: CircularProgressIndicator(
                              color: Color(0xFF3A86FF),
                            ),
                          ),
                        )
                      : _buildExamList(),
                  const SizedBox(height: 50),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSliverHeader() {
    return SliverAppBar(
      expandedHeight: 120,
      backgroundColor: Colors.transparent,
      floating: false,
      pinned: true,
      elevation: 0,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back_ios_rounded,
            color: Colors.white, size: 20),
        onPressed: () => Navigator.of(context).pop(),
      ),
      flexibleSpace: FlexibleSpaceBar(
        titlePadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        centerTitle: false,
        title: Text(
          "Timeline",
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 22,
          ),
        ),
        background: Container(color: const Color(0xFF050505)),
      ),
    );
  }

  Widget _buildMonthlyCalendar() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 32, sigmaY: 32),
        child: Container(
          decoration: BoxDecoration(
            color: SpatialColors.charcoalLight.withValues(alpha: 0.86),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
            boxShadow: SpatialGlow.glassDock,
          ),
          child: TableCalendar(
            firstDay: DateTime.utc(2024, 1, 1),
            lastDay: DateTime.utc(2027, 12, 31),
            focusedDay: _focusedDay,
            calendarFormat: _calendarFormat,
            selectedDayPredicate: (day) => isSameDay(_selectedDay, day),
            onDaySelected: (selectedDay, focusedDay) {
              setState(() {
                _selectedDay = selectedDay;
                _focusedDay = focusedDay;
              });
              HapticFeedback.lightImpact();
            },
            onFormatChanged: (format) {
              setState(() {
                _calendarFormat = format;
              });
            },
            eventLoader: (day) {
              return _exams.where((exam) {
                return isSameDay(exam.date, day);
              }).toList();
            },
            calendarStyle: CalendarStyle(
              defaultTextStyle:
                  GoogleFonts.googleSans(color: Colors.white70, fontSize: 14),
              weekendTextStyle:
                  GoogleFonts.googleSans(color: Colors.white70, fontSize: 14),
              outsideDaysVisible: false,
              markersMaxCount: 3,
              markerDecoration: const BoxDecoration(
                color: Color(0xFF3A86FF),
                shape: BoxShape.circle,
              ),
              markerSize: 6,
              markerMargin: const EdgeInsets.symmetric(horizontal: 1),
            ),
            calendarBuilders: CalendarBuilders<ExamEventModel>(
              defaultBuilder: (context, day, focusedDay) {
                final isToday = isSameDay(day, DateTime.now());
                if (isToday) {
                  return Container(
                    margin: const EdgeInsets.all(4),
                    alignment: Alignment.center,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          '${day.day}',
                          style: GoogleFonts.googleSans(
                            color: Colors.white38,
                            fontSize: 14,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Container(
                          width: 4,
                          height: 4,
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.3),
                            shape: BoxShape.circle,
                          ),
                        ),
                      ],
                    ),
                  );
                }
                return null;
              },
              todayBuilder: (context, day, focusedDay) {
                final isSelected =
                    _selectedDay != null && isSameDay(_selectedDay!, day);
                if (isSelected) {
                  return Container(
                    margin: const EdgeInsets.all(4),
                    alignment: Alignment.center,
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          '${day.day}',
                          style: GoogleFonts.googleSans(
                            color: Colors.white38,
                            fontSize: 14,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Container(
                          width: 4,
                          height: 4,
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.3),
                            shape: BoxShape.circle,
                          ),
                        ),
                      ],
                    ),
                  );
                }
                return Container(
                  margin: const EdgeInsets.all(4),
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: const Color(0xFF3A86FF),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.3),
                      width: 0.5,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFF3A86FF).withValues(alpha: 0.3),
                        blurRadius: 4,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Text(
                    '${day.day}',
                    style: GoogleFonts.googleSans(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                );
              },
              selectedBuilder: (context, day, focusedDay) {
                return TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.8, end: 1.0),
                  duration: const Duration(milliseconds: 200),
                  builder: (context, value, child) {
                    return Transform.scale(
                      scale: value,
                      child: Container(
                        margin: const EdgeInsets.all(4),
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          color: const Color(0xFF3A86FF),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.4),
                            width: 1,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFF3A86FF)
                                  .withValues(alpha: 0.4),
                              blurRadius: 8,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: Text(
                          '${day.day}',
                          style: GoogleFonts.googleSans(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 14,
                          ),
                        ),
                      ),
                    );
                  },
                );
              },
            ),
            headerStyle: HeaderStyle(
              formatButtonVisible: true,
              formatButtonShowsNext: false,
              formatButtonDecoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(12),
              ),
              formatButtonTextStyle:
                  GoogleFonts.googleSans(color: Colors.white54, fontSize: 12),
              titleCentered: true,
              titleTextStyle: GoogleFonts.googleSans(
                color: Colors.white,
                fontWeight: FontWeight.w600,
                fontSize: 16,
              ),
              leftChevronIcon:
                  const Icon(Icons.chevron_left, color: Colors.white24),
              rightChevronIcon:
                  const Icon(Icons.chevron_right, color: Colors.white24),
            ),
            daysOfWeekStyle: DaysOfWeekStyle(
              weekdayStyle:
                  GoogleFonts.robotoMono(color: Colors.white54, fontSize: 11),
              weekendStyle:
                  GoogleFonts.robotoMono(color: Colors.white54, fontSize: 11),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDailyPlanPanel() {
    final auth = ref.watch(authStateProvider);
    if (!auth.isAuthenticated || _userId == null) {
      return const SizedBox.shrink();
    }

    return DailyPlanPanel(
      uid: _userId!,
      service: ref.watch(dailyPlanServiceProvider),
    );
  }

  Widget _buildExamList() {
    if (_exams.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(40),
          child: Column(
            children: [
              Icon(Icons.event_busy_rounded, color: Colors.white24, size: 48),
              const SizedBox(height: 12),
              Text(
                'No upcoming exams',
                style: GoogleFonts.googleSans(color: Colors.white38),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _exams.length,
      itemBuilder: (context, index) {
        final exam = _exams[index];
        return _ExamDetailCard(exam: exam);
      },
    );
  }
}

class _ExamDetailCard extends StatelessWidget {
  final ExamEventModel exam;

  const _ExamDetailCard({required this.exam});

  @override
  Widget build(BuildContext context) {
    final bool isUrgent = exam.daysRemaining <= 14 && exam.daysRemaining > 0;
    final bool isPast = exam.daysRemaining <= 0;

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.02),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isUrgent
              ? Colors.red.withValues(alpha: 0.2)
              : Colors.white.withValues(alpha: 0.05),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: const Color(0xFF3A86FF).withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  exam.code,
                  style: GoogleFonts.robotoMono(
                    color: const Color(0xFF3A86FF),
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Text(
                DateFormat('EEE, MMM d').format(exam.date),
                style:
                    GoogleFonts.googleSans(color: Colors.white38, fontSize: 12),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            exam.subject,
            style: GoogleFonts.googleSans(
              color: isPast ? Colors.white38 : Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              _InfoBadge(
                icon: Icons.access_time_rounded,
                label: exam.time.isNotEmpty ? exam.time : 'TBA',
              ),
              const SizedBox(width: 12),
              _InfoBadge(
                icon: Icons.layers_outlined,
                label: exam.component.isNotEmpty ? exam.component : 'Paper',
              ),
            ],
          ),
          if (isUrgent) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.red.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                "CRITICAL: ${exam.daysRemaining} DAYS REMAINING",
                style: GoogleFonts.robotoMono(
                  color: Colors.redAccent,
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
          if (isPast) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                "COMPLETED",
                style: GoogleFonts.robotoMono(
                  color: Colors.white38,
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _InfoBadge extends StatelessWidget {
  final IconData icon;
  final String label;

  const _InfoBadge({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: Colors.white54),
          const SizedBox(width: 6),
          Text(
            label,
            style: GoogleFonts.googleSans(color: Colors.white70, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

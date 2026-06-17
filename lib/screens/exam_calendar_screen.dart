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
import '../services/daily_plan_service.dart';
import '../components/notion_calendar_view.dart';

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
      backgroundColor: AxonColors.background,
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          _buildSliverHeader(),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: _isLoading 
                  ? const Center(
                      child: Padding(
                        padding: EdgeInsets.all(40),
                        child: CircularProgressIndicator(
                          color: Color(0xFF3A86FF),
                        ),
                      ),
                    )
                  : _buildUnifiedCalendar(),
            ),
          ),
          const SliverToBoxAdapter(child: SizedBox(height: 50)),
        ],
      ),
    );
  }

  Widget _buildUnifiedCalendar() {
    if (_userId == null) return const SizedBox.shrink();

    return StreamBuilder<List<DailyPlanTask>>(
      stream: ref.watch(dailyPlanServiceProvider).watchTodayPlan(_userId!),
      builder: (context, snapshot) {
        final tasks = snapshot.data ?? [];
        
        return NotionStyleCalendar(
          events: _exams,
          dailyPlanTasks: tasks,
          enableDragDrop: true,
          onEventTap: (exam) {
            // Placeholder for exam details
          },
          onDailyPlanTaskTap: (task) {
            // Trigger bottom sheet for focus timer
            _showTaskBottomSheet(task);
          },
          onDailyPlanTaskDrop: (task, droppedTime) async {
            // Reschedule logic goes here
            await _updateTaskTime(task, droppedTime);
          },
        );
      },
    );
  }

  void _showTaskBottomSheet(DailyPlanTask task) {
    showModalBottomSheet(
      context: context,
      backgroundColor: AxonColors.surfaceHighlight,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                task.title,
                style: TextStyle(
                  color: AxonColors.textPrimary,
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Subject: ${task.subject}',
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.7),
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 32),
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFD4E6B5), // Mint Emerald
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  onPressed: () {
                    Navigator.pop(context);
                    // Start Focus Timer logic here
                  },
                  child: const Text(
                    'Start Focus Timer',
                    style: TextStyle(
                      color: Color(0xFF0F172A),
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
            ],
          ),
        );
      },
    );
  }

  Future<void> _updateTaskTime(DailyPlanTask task, DateTime newStartTime) async {
    final duration = task.endTime.difference(task.startTime);
    final newEndTime = newStartTime.add(duration);
    
    // Will implement updateTaskTimes in DailyPlanService next
    await ref.read(dailyPlanServiceProvider).updateTaskTimes(
      _userId!,
      task.id,
      newStartTime,
      newEndTime,
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
        icon: Icon(Icons.arrow_back_ios_rounded,
            color: AxonColors.textPrimary, size: 20),
        onPressed: () => Navigator.of(context).pop(),
      ),
      flexibleSpace: FlexibleSpaceBar(
        titlePadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        centerTitle: false,
        title: Text(
          "Timeline",
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontWeight: FontWeight.bold,
            fontSize: 22,
          ),
        ),
        background: Container(color: AxonColors.background),
      ),
    );
  }
}



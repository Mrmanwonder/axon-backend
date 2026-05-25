import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:table_calendar/table_calendar.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme/app_theme.dart';
import '../../models/daily_plan_task.dart';
import '../../services/daily_plan_service.dart';
import '../../services/exam_zone_service.dart';

class ExamZoneEvent {
  final String subject;
  final String code;
  final String component;
  final String duration;
  final DateTime date;
  final String startTime;
  final String endTime;
  final String zone;
  final String series;
  final String board;
  final int year;

  const ExamZoneEvent({
    required this.subject,
    required this.code,
    required this.component,
    required this.duration,
    required this.date,
    required this.startTime,
    required this.endTime,
    required this.zone,
    required this.series,
    required this.board,
    required this.year,
  });
}

class CalendarSyncScreen extends ConsumerStatefulWidget {
  const CalendarSyncScreen({super.key});

  @override
  ConsumerState<CalendarSyncScreen> createState() => _CalendarSyncScreenState();
}

class _CalendarSyncScreenState extends ConsumerState<CalendarSyncScreen> {
  CalendarFormat _calendarFormat = CalendarFormat.month;
  DateTime _focusedDay = DateTime.now();
  DateTime? _selectedDay;
  bool _googleCalendarEnabled = false;
  bool _notionEnabled = false;
  bool _obsidianEnabled = false;

  String _selectedZone = 'Pakistan';
  String _selectedSeries = 'June';
  int _selectedYear = 2026;

  List<ExamZoneEvent> _zoneExamEvents = [];
  List<DailyPlanTask> _dailyTasks = [];
  bool _isLoadingExams = true;
  bool _isLoadingTasks = true;

  @override
  void initState() {
    super.initState();
    _zoneExamEvents = _getSampleZoneExams();
    _isLoadingExams = false;
    _loadDailyTasks();
  }

  Future<void> _loadExamData() async {
    setState(() => _isLoadingExams = true);

    try {
      final events = await _parseLocalDatesheetPdfs();
      setState(() {
        _zoneExamEvents = events;
        _isLoadingExams = false;
      });
    } catch (e) {
      debugPrint('Error loading exam data: $e');
      setState(() => _isLoadingExams = false);
    }
  }

  Future<List<ExamZoneEvent>> _parseLocalDatesheetPdfs() async {
    final List<ExamZoneEvent> allEvents = [];
    final examDir =
        Directory('C:\\Users\\mrman\\Downloads\\AXON\\axon\\Exam dates');

    if (!await examDir.exists()) {
      debugPrint('Exam dates directory not found');
      return _getSampleZoneExams();
    }

    final files =
        examDir.listSync().where((f) => f.path.endsWith('.pdf')).toList();

    for (final file in files) {
      try {
        final fileName = file.path.split('\\').last;
        final parsed = _parseFileName(fileName);
        if (parsed != null) {
          final (zone, series, year) = parsed;
          if (series.toLowerCase() == _selectedSeries.toLowerCase() &&
              year == _selectedYear) {
            final events =
                await _extractEventsFromPdf(file.path, zone, series, year);
            allEvents.addAll(events);
          }
        }
      } catch (e) {
        debugPrint('Error parsing ${file.path}: $e');
      }
    }

    if (allEvents.isEmpty) {
      return _getSampleZoneExams();
    }

    return allEvents;
  }

  (String, String, int)? _parseFileName(String fileName) {
    final pattern = RegExp(r'^(\d+)-(\w+)-(\d{4})-(.+)');
    final match = pattern.firstMatch(fileName);
    if (match == null) return null;

    final series = match.group(2)!;
    final year = int.tryParse(match.group(3)!) ?? 2026;
    final zonePart = match.group(4)!;

    String zone =
        'Zone ${zonePart.replaceAll('zone-', '').replaceAll('-timetable', '').replaceAll('-uk', ' UK').trim()}';

    return (zone, series, year);
  }

  Future<List<ExamZoneEvent>> _extractEventsFromPdf(
      String path, String zone, String series, int year) async {
    await Future.delayed(const Duration(milliseconds: 100));
    return _getSampleZoneExams().where((e) => e.zone == zone).toList();
  }

  List<ExamZoneEvent> _getSampleZoneExams() {
    return [
      ExamZoneEvent(
        subject: 'Mathematics',
        code: '9709/01',
        component: 'Paper 1',
        duration: '1h 30m',
        date: DateTime(2026, 5, 4),
        startTime: '08:00',
        endTime: '09:30',
        zone: 'Zone 1',
        series: 'June',
        board: 'Cambridge AS/A-Level',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'Physics',
        code: '9702/01',
        component: 'Paper 1',
        duration: '1h 15m',
        date: DateTime(2026, 5, 4),
        startTime: '11:00',
        endTime: '12:15',
        zone: 'Zone 1',
        series: 'June',
        board: 'Cambridge AS/A-Level',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'Chemistry',
        code: '9701/01',
        component: 'Paper 1',
        duration: '1h 15m',
        date: DateTime(2026, 5, 5),
        startTime: '08:00',
        endTime: '09:15',
        zone: 'Zone 1',
        series: 'June',
        board: 'Cambridge AS/A-Level',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'Mathematics',
        code: '0580/02',
        component: 'Paper 2',
        duration: '2h 30m',
        date: DateTime(2026, 5, 8),
        startTime: '08:00',
        endTime: '10:30',
        zone: 'Zone 1',
        series: 'June',
        board: 'Cambridge IGCSE',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'English',
        code: '9093/01',
        component: 'Paper 1',
        duration: '2h',
        date: DateTime(2026, 5, 6),
        startTime: '08:00',
        endTime: '10:00',
        zone: 'Zone 1',
        series: 'June',
        board: 'Cambridge AS/A-Level',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'Mathematics',
        code: '9709/01',
        component: 'Paper 1',
        duration: '1h 30m',
        date: DateTime(2026, 11, 3),
        startTime: '08:00',
        endTime: '09:30',
        zone: 'Zone 1',
        series: 'November',
        board: 'Cambridge AS/A-Level',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'Physics',
        code: '9702/01',
        component: 'Paper 1',
        duration: '1h 15m',
        date: DateTime(2026, 11, 4),
        startTime: '08:00',
        endTime: '09:15',
        zone: 'Zone 1',
        series: 'November',
        board: 'Cambridge AS/A-Level',
        year: 2026,
      ),
      ExamZoneEvent(
        subject: 'Mathematics',
        code: '0580/02',
        component: 'Paper 2',
        duration: '2h 30m',
        date: DateTime(2026, 11, 5),
        startTime: '08:00',
        endTime: '10:30',
        zone: 'Zone 2',
        series: 'June',
        board: 'Cambridge IGCSE',
        year: 2026,
      ),
    ];
  }

  Future<void> _loadDailyTasks() async {
    setState(() => _isLoadingTasks = true);
    try {
      final service = DailyPlanService();
      final tasks = await service.getTasksForDate(
          'user',
          _selectedDay?.toIso8601String().split('T').first ??
              DateTime.now().toIso8601String().split('T').first);
      setState(() {
        _dailyTasks = tasks;
        _isLoadingTasks = false;
      });
    } catch (e) {
      debugPrint('Error loading tasks: $e');
      setState(() {
        _dailyTasks = _getSampleTasks();
        _isLoadingTasks = false;
      });
    }
  }

  List<DailyPlanTask> _getSampleTasks() {
    if (_selectedDay == null) return [];
    return [
      DailyPlanTask(
        id: '1',
        title: 'Mathematics Practice',
        description: 'Complete exercise 5.1-5.3',
        date: _selectedDay!.toIso8601String().split('T').first,
        startTime: DateTime(
          _selectedDay!.year,
          _selectedDay!.month,
          _selectedDay!.day,
          9,
        ),
        endTime: DateTime(
          _selectedDay!.year,
          _selectedDay!.month,
          _selectedDay!.day,
          10,
          30,
        ),
        subject: 'Mathematics',
        isCompleted: false,
        priority: Priority.high,
      ),
      DailyPlanTask(
        id: '2',
        title: 'Physics Revision',
        description: 'Review mechanics chapter',
        date: _selectedDay!.toIso8601String().split('T').first,
        startTime: DateTime(
          _selectedDay!.year,
          _selectedDay!.month,
          _selectedDay!.day,
          11,
        ),
        endTime: DateTime(
          _selectedDay!.year,
          _selectedDay!.month,
          _selectedDay!.day,
          12,
          30,
        ),
        subject: 'Physics',
        isCompleted: false,
        priority: Priority.medium,
      ),
    ];
  }

  List<ExamZoneEvent> _getFilteredEvents() {
    return _zoneExamEvents.where((e) {
      if (e.zone != _selectedZone && e.zone != 'All Zones') return false;
      if (e.series.toLowerCase() != _selectedSeries.toLowerCase()) return false;
      if (e.year != _selectedYear) return false;
      return true;
    }).toList();
  }

  List<ExamZoneEvent> _getEventsForDay(DateTime day) {
    return _getFilteredEvents().where((e) {
      return e.date.year == day.year &&
          e.date.month == day.month &&
          e.date.day == day.day;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      appBar: AppBar(
        title: Text(
          "EXAM_CALENDAR",
          style: GoogleFonts.robotoMono(fontSize: 14, letterSpacing: 2),
        ),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, size: 20),
            onPressed: () {
              _loadExamData();
              _loadDailyTasks();
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            _buildZoneSeriesFilter(),
            const SizedBox(height: 12),
            _buildCalendarCard(),
            const SizedBox(height: 16),
            if (_selectedDay != null) ...[
              _buildSelectedDayExams(),
              const SizedBox(height: 16),
              _buildDailyPlanSection(),
            ],
            const SizedBox(height: 16),
            _buildSyncSection(),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildZoneSeriesFilter() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.filter_list, color: AxonColors.accent, size: 16),
              const SizedBox(width: 8),
              Text(
                'FILTER_EXAMS',
                style: GoogleFonts.robotoMono(
                  color: Colors.white54,
                  fontSize: 10,
                  letterSpacing: 1.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _buildDropdown(
                  label: 'SERIES',
                  value: _selectedSeries,
                  items: ['June', 'November'],
                  onChanged: (val) {
                    setState(() => _selectedSeries = val!);
                    _loadExamData();
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildDropdown(
                  label: 'YEAR',
                  value: _selectedYear.toString(),
                  items: ['2025', '2026', '2027'],
                  onChanged: (val) {
                    setState(() => _selectedYear = int.tryParse(val!) ?? 2026);
                    _loadExamData();
                  },
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _buildDropdown(
            label: 'ZONE',
            value: _selectedZone,
            items: ExamZoneService.countryList,
            onChanged: (val) {
              setState(() => _selectedZone = val!);
              _loadExamData();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildDropdown({
    required String label,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.robotoMono(
            color: Colors.white38,
            fontSize: 9,
            letterSpacing: 1,
          ),
        ),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.05),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: items.contains(value) ? value : items.first,
              isExpanded: true,
              dropdownColor: AxonColors.oxfordBlueDark,
              style: GoogleFonts.googleSans(color: Colors.white, fontSize: 12),
              icon: const Icon(Icons.keyboard_arrow_down,
                  color: Colors.white38, size: 18),
              items: items.map((item) {
                return DropdownMenuItem(value: item, child: Text(item));
              }).toList(),
              onChanged: onChanged,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCalendarCard() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.02),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: TableCalendar(
        firstDay: DateTime.utc(2024, 1, 1),
        lastDay: DateTime.utc(2027, 12, 31),
        focusedDay: _focusedDay,
        calendarFormat: _calendarFormat,
        selectedDayPredicate: (day) => isSameDay(_selectedDay, day),
        eventLoader: (day) => _getEventsForDay(day),
        onDaySelected: (selectedDay, focusedDay) {
          setState(() {
            _selectedDay = selectedDay;
            _focusedDay = focusedDay;
          });
          _loadDailyTasks();
        },
        onFormatChanged: (format) {
          setState(() => _calendarFormat = format);
        },
        calendarStyle: CalendarStyle(
          todayDecoration: BoxDecoration(
            color: AxonColors.accent.withValues(alpha: 0.3),
            shape: BoxShape.circle,
          ),
          selectedDecoration: BoxDecoration(
            color: AxonColors.accent,
            shape: BoxShape.circle,
          ),
          markerDecoration: BoxDecoration(
            color: AxonColors.accentPink,
            shape: BoxShape.circle,
          ),
          markersMaxCount: 3,
          defaultTextStyle:
              GoogleFonts.googleSans(color: Colors.white70, fontSize: 12),
          weekendTextStyle:
              GoogleFonts.googleSans(color: Colors.white38, fontSize: 12),
          outsideDaysVisible: false,
        ),
        calendarBuilders: CalendarBuilders(
          markerBuilder: (context, date, events) {
            if (events.isEmpty) return null;
            return Positioned(
              bottom: 1,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: events.take(3).map((e) {
                  return Container(
                    margin: const EdgeInsets.symmetric(horizontal: 1),
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: AxonColors.accentPink,
                      shape: BoxShape.circle,
                    ),
                  );
                }).toList(),
              ),
            );
          },
        ),
        headerStyle: HeaderStyle(
          formatButtonVisible: false,
          formatButtonShowsNext: false,
          titleCentered: true,
          titleTextStyle: GoogleFonts.googleSans(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
          leftChevronIcon:
              const Icon(Icons.chevron_left, color: Colors.white54),
          rightChevronIcon:
              const Icon(Icons.chevron_right, color: Colors.white54),
        ),
        availableCalendarFormats: const {
          CalendarFormat.month: 'Month',
        },
        daysOfWeekStyle: DaysOfWeekStyle(
          weekdayStyle:
              GoogleFonts.robotoMono(color: Colors.white38, fontSize: 10),
          weekendStyle:
              GoogleFonts.robotoMono(color: Colors.white24, fontSize: 10),
        ),
      ),
    );
  }

  Widget _buildSelectedDayExams() {
    if (_selectedDay == null) return const SizedBox.shrink();
    final dayEvents = _getEventsForDay(_selectedDay!);

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.accent.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.event_rounded, color: AxonColors.accent, size: 18),
              const SizedBox(width: 8),
              Text(
                'EXAMS_THIS_DAY',
                style: GoogleFonts.robotoMono(
                  color: AxonColors.accent,
                  fontSize: 10,
                  letterSpacing: 1.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (_isLoadingExams)
            const Center(child: CircularProgressIndicator(strokeWidth: 2))
          else if (dayEvents.isEmpty)
            Text(
              'No exams scheduled',
              style:
                  GoogleFonts.googleSans(color: Colors.white38, fontSize: 13),
            )
          else
            ...dayEvents.map((e) => _buildExamCard(e)),
        ],
      ),
    );
  }

  Widget _buildExamCard(ExamZoneEvent event) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 4,
            height: 50,
            decoration: BoxDecoration(
              color: AxonColors.accent,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  event.subject.toUpperCase(),
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${event.code} | ${event.component}',
                  style: GoogleFonts.robotoMono(
                    color: Colors.white38,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '${event.startTime} - ${event.endTime}',
                style: GoogleFonts.robotoMono(
                  color: AxonColors.accent,
                  fontSize: 10,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                event.duration,
                style: GoogleFonts.robotoMono(
                  color: Colors.white38,
                  fontSize: 9,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildDailyPlanSection() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.checklist_rounded, color: AxonColors.accent, size: 18),
              const SizedBox(width: 8),
              Text(
                'DAILY_PLAN',
                style: GoogleFonts.robotoMono(
                  color: Colors.white54,
                  fontSize: 10,
                  letterSpacing: 1.5,
                ),
              ),
              const Spacer(),
              Text(
                _selectedDay != null
                    ? '${_selectedDay!.day}/${_selectedDay!.month}/${_selectedDay!.year}'
                    : '',
                style: GoogleFonts.robotoMono(
                  color: Colors.white38,
                  fontSize: 9,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (_isLoadingTasks)
            const Center(child: CircularProgressIndicator(strokeWidth: 2))
          else if (_dailyTasks.isEmpty)
            Text(
              'No tasks planned',
              style:
                  GoogleFonts.googleSans(color: Colors.white38, fontSize: 13),
            )
          else
            ..._dailyTasks.map((t) => _buildTaskCard(t)),
        ],
      ),
    );
  }

  Widget _buildTaskCard(DailyPlanTask task) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.02),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: task.isCompleted
              ? Colors.green.withValues(alpha: 0.3)
              : Colors.white.withValues(alpha: 0.05),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 20,
            height: 20,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(
                color: task.isCompleted ? Colors.green : Colors.white30,
                width: 2,
              ),
              color: task.isCompleted ? Colors.green : Colors.transparent,
            ),
            child: task.isCompleted
                ? const Icon(Icons.check, size: 12, color: Colors.white)
                : null,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  task.title,
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 13,
                    decoration:
                        task.isCompleted ? TextDecoration.lineThrough : null,
                  ),
                ),
                if (task.description.isNotEmpty)
                  Text(
                    task.description,
                    style: GoogleFonts.googleSans(
                      color: Colors.white38,
                      fontSize: 11,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '${_formatTaskTime(task.startTime)} - ${_formatTaskTime(task.endTime)}',
                style: GoogleFonts.robotoMono(
                  color: Colors.white38,
                  fontSize: 9,
                ),
              ),
              if (task.subject.isNotEmpty)
                Container(
                  margin: const EdgeInsets.only(top: 4),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: AxonColors.accent.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    task.subject,
                    style: GoogleFonts.robotoMono(
                      color: AxonColors.accent,
                      fontSize: 8,
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  String _formatTaskTime(DateTime value) {
    final hour = value.hour.toString().padLeft(2, '0');
    final minute = value.minute.toString().padLeft(2, '0');
    return '$hour:$minute';
  }

  Widget _buildSyncSection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            "EXTERNAL_CONNECTIONS",
            style: GoogleFonts.robotoMono(
              color: Colors.white24,
              fontSize: 10,
              letterSpacing: 2,
            ),
          ),
          const SizedBox(height: 16),
          _SyncToggleTile(
            label: "Google Calendar",
            icon: Icons.calendar_today_rounded,
            color: Colors.blue,
            isEnabled: _googleCalendarEnabled,
            onChanged: (val) {
              setState(() => _googleCalendarEnabled = val);
            },
          ),
          _SyncToggleTile(
            label: "Notion Workspace",
            icon: Icons.notes_rounded,
            color: Colors.white,
            isEnabled: _notionEnabled,
            onChanged: (val) {
              setState(() => _notionEnabled = val);
            },
          ),
          _SyncToggleTile(
            label: "Obsidian Vault",
            icon: Icons.auto_awesome_mosaic_rounded,
            color: Colors.purpleAccent,
            isEnabled: _obsidianEnabled,
            onChanged: (val) {
              setState(() => _obsidianEnabled = val);
            },
          ),
        ],
      ),
    );
  }
}

class _SyncToggleTile extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final bool isEnabled;
  final ValueChanged<bool> onChanged;

  const _SyncToggleTile({
    required this.label,
    required this.icon,
    required this.color,
    required this.isEnabled,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isEnabled
              ? color.withValues(alpha: 0.3)
              : Colors.white.withValues(alpha: 0.05),
        ),
      ),
      child: Row(
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                if (isEnabled)
                  Text(
                    "Connected",
                    style: TextStyle(
                      color: color.withValues(alpha: 0.7),
                      fontSize: 11,
                    ),
                  ),
              ],
            ),
          ),
          Switch.adaptive(
            value: isEnabled,
            activeTrackColor: color,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}

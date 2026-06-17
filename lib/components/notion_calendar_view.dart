import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_calendar/calendar.dart';

import '../../models/models.dart';
import '../../models/exam_event_model.dart';
import '../../models/daily_plan_task.dart';
import '../../theme/app_theme.dart';

class NotionStyleCalendar extends ConsumerStatefulWidget {
  final List<ExamEventModel> events;
  final List<DailyPlanTask>? dailyPlanTasks;
  final Function(ExamEventModel)? onEventTap;
  final Function(DailyPlanTask)? onDailyPlanTaskTap;
  final Function(ExamEventModel, DateTime)? onEventDrop;
  final Function(DailyPlanTask, DateTime)? onDailyPlanTaskDrop;
  final Function(DateTime)? onTimeSlotTap;
  final bool enableDragDrop;
  final Widget Function(BuildContext, DateTime, List<ExamEventModel>)? dayBuilder;

  const NotionStyleCalendar({
    super.key,
    this.events = const [],
    this.dailyPlanTasks,
    this.onEventTap,
    this.onDailyPlanTaskTap,
    this.onEventDrop,
    this.onDailyPlanTaskDrop,
    this.onTimeSlotTap,
    this.enableDragDrop = true,
    this.dayBuilder,
  });

  /// Static helper to create a calendar for DailyPlanTasks
  static Widget buildForDailyPlan({
    Key? key,
    required List<DailyPlanTask> tasks,
    Function(DailyPlanTask)? onTaskTap,
    Function(DailyPlanTask, DateTime)? onTaskDrop,
    Function(DateTime)? onTimeSlotTap,
  }) {
    return NotionStyleCalendar(
      key: key,
      dailyPlanTasks: tasks,
      onDailyPlanTaskTap: onTaskTap,
      onDailyPlanTaskDrop: onTaskDrop,
      onTimeSlotTap: onTimeSlotTap,
    );
  }

  @override
  ConsumerState<NotionStyleCalendar> createState() =>
      _NotionStyleCalendarState();
}

class _NotionStyleCalendarState extends ConsumerState<NotionStyleCalendar> {
  late DateTime _selectedDate;
  _CalendarMode _mode = _CalendarMode.month;

  @override
  void initState() {
    super.initState();
    _selectedDate = _initialSelectedDate(widget.events);
  }

  @override
  void didUpdateWidget(covariant NotionStyleCalendar oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.events == widget.events) return;
    if (widget.events.isEmpty) {
      _selectedDate = _normalizeDate(DateTime.now());
      return;
    }

    final hasSelectedDayEvent = _eventsForDay(_selectedDate).isNotEmpty;
    if (!hasSelectedDayEvent) {
      _selectedDate = _initialSelectedDate(widget.events);
    }
  }

  DateTime _initialSelectedDate(List<ExamEventModel> events) {
    if (events.isEmpty) return _normalizeDate(DateTime.now());
    final sorted = [...events]
      ..sort((a, b) => a.date.compareTo(b.date));
    final now = DateTime.now();
    final upcoming = sorted.where((event) => !event.date.isBefore(now));
    return _normalizeDate(
      upcoming.isNotEmpty ? upcoming.first.date : sorted.first.date,
    );
  }

  DateTime _normalizeDate(DateTime date) =>
      DateTime(date.year, date.month, date.day);

  List<ExamEventModel> _eventsForDay(DateTime day) {
    final normalized = _normalizeDate(day);
    final items = widget.events.where((event) {
      final eventDay = _normalizeDate(event.date);
      return eventDay == normalized;
    }).toList()
      ..sort((a, b) => a.date.compareTo(b.date));
    return items;
  }

  List<DailyPlanTask> _dailyPlanTasksForDay(DateTime day) {
    final normalized = _normalizeDate(day);
    final tasks = widget.dailyPlanTasks ?? [];
    return tasks.where((task) {
      final taskDay = _normalizeDate(task.startTime);
      return taskDay == normalized;
    }).toList()
      ..sort((a, b) => a.startTime.compareTo(b.startTime));
  }

  bool _hasEventOnDay(DateTime day) {
    final hasExamEvents = _eventsForDay(day).isNotEmpty;
    final hasDailyPlanTasks = _dailyPlanTasksForDay(day).isNotEmpty;
    return hasExamEvents || hasDailyPlanTasks;
  }

  /// Format agenda metadata for DailyPlanTask
  String _agendaMetaForDailyPlan(DailyPlanTask task) {
    final time = _formatTimeRange(task.startTime, task.endTime);
    if (time.isNotEmpty) {
      final intensity = task.intensityLabel.value.toUpperCase();
      return '$time • ${task.phase.label} • INTENSITY: $intensity';
    }
    return '${task.subject} • ${task.taskType.value}';
  }

  List<DateTime> _daysForMonth(DateTime selectedDate) {
    final firstOfMonth = DateTime(selectedDate.year, selectedDate.month, 1);
    final leadingOffset = firstOfMonth.weekday - DateTime.monday;
    final gridStart = firstOfMonth.subtract(Duration(days: leadingOffset));
    return List<DateTime>.generate(
      35,
      (index) => gridStart.add(Duration(days: index)),
    );
  }

  List<DateTime> _daysForWeek(DateTime selectedDate) {
    final start = _normalizeDate(selectedDate)
        .subtract(Duration(days: selectedDate.weekday - 1));
    return List<DateTime>.generate(
        7, (index) => start.add(Duration(days: index)));
  }

  String _headerLabel(DateTime date) {
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];
    return '${date.day} ${months[date.month - 1]} ${date.year}';
  }

  String _agendaMeta(ExamEventModel event) {
    final endTime = event.date.add(const Duration(hours: 2));
    final time = _formatTimeRange(event.date, endTime);
    if (time.isNotEmpty) return time;
    return '${event.board.toUpperCase()} • ${_headerLabel(event.date)}';
  }

  String _formatTimeRange(DateTime start, DateTime end) {
    final sameDay = start.year == end.year &&
        start.month == end.month &&
        start.day == end.day;
    final hasExplicitTime = start.hour != 0 ||
        start.minute != 0 ||
        end.hour != 0 ||
        end.minute != 0;
    if (!sameDay || !hasExplicitTime) return '';
    return '${_formatTime(start)}-${_formatTime(end)}';
  }

  String _formatTime(DateTime value) {
    final hour = value.hour % 12 == 0 ? 12 : value.hour % 12;
    final minute = value.minute.toString().padLeft(2, '0');
    final suffix = value.hour >= 12 ? 'PM' : 'AM';
    return '$hour:$minute$suffix';
  }

  void _selectDate(DateTime day) {
    final normalized = _normalizeDate(day);
    setState(() => _selectedDate = normalized);
    widget.onTimeSlotTap?.call(normalized);
  }

  @override
  Widget build(BuildContext context) {
    final selectedEvents = _eventsForDay(_selectedDate);
    final selectedDailyPlanTasks = widget.dailyPlanTasks != null
        ? _dailyPlanTasksForDay(_selectedDate)
        : <DailyPlanTask>[];
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final calendarBg =
        AxonColors.surfaceHighlight;

    return Container(
      decoration: BoxDecoration(
        color: calendarBg,
        borderRadius: BorderRadius.circular(32),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.18),
            blurRadius: 28,
            offset: const Offset(0, 18),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 22, 20, 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              _headerLabel(_selectedDate),
              style: TextStyle(
                color: AxonColors.textPrimary,
                fontSize: 22,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.6,
              ),
            ),
            const SizedBox(height: 18),
            Row(
              children: _CalendarMode.values
                  .map(
                    (mode) => Expanded(
                      child: Padding(
                        padding: EdgeInsets.only(
                          right: mode == _CalendarMode.day ? 0 : 10,
                        ),
                        child: _ModePill(
                          label: mode.label,
                          selected: _mode == mode,
                          onTap: () => setState(() => _mode = mode),
                        ),
                      ),
                    ),
                  )
                  .toList(),
            ),
            const SizedBox(height: 20),
            if (_mode == _CalendarMode.month)
              _MonthGrid(
                selectedDate: _selectedDate,
                days: _daysForMonth(_selectedDate),
                hasEventOnDay: _hasEventOnDay,
                onSelectDate: _selectDate,
              ),
            if (_mode == _CalendarMode.week)
              _WeekStrip(
                selectedDate: _selectedDate,
                days: _daysForWeek(_selectedDate),
                hasEventOnDay: _hasEventOnDay,
                onSelectDate: _selectDate,
              ),
            if (_mode == _CalendarMode.day)
              SizedBox(
                height: 500,
                child: _DayTimeline(
                  date: _selectedDate,
                  events: selectedEvents,
                  tasks: selectedDailyPlanTasks,
                  enableDragDrop: widget.enableDragDrop,
                  onEventDrop: widget.onEventDrop,
                  onTaskDrop: widget.onDailyPlanTaskDrop,
                  onEventTap: widget.onEventTap,
                  onTaskTap: widget.onDailyPlanTaskTap,
                ),
              ),
            if (_mode != _CalendarMode.day) ...[
              const SizedBox(height: 18),
              Divider(color: AxonColors.textPrimary.withValues(alpha: 0.22), height: 1),
              const SizedBox(height: 18),
              if (selectedEvents.isEmpty && selectedDailyPlanTasks.isEmpty)
                Container(
                  width: double.infinity,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
                  decoration: BoxDecoration(
                    color: AxonColors.textPrimary.withValues(alpha: 0.94),
                    borderRadius: BorderRadius.circular(22),
                  ),
                  child: Text(
                    widget.dailyPlanTasks != null
                        ? 'No study blocks planned.'
                        : 'No exams on this date.',
                    style: TextStyle(
                      color: AxonColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                )
              else
                Column(
                  children: [
                    // Show DailyPlanTasks first (if present)
                    if (selectedDailyPlanTasks.isNotEmpty)
                      ...selectedDailyPlanTasks.map(
                        (task) => Padding(
                          padding: const EdgeInsets.only(bottom: 14),
                          child: _AgendaCard(
                            title: task.title,
                            subtitle: _agendaMetaForDailyPlan(task),
                            accentColor: _intensityColor(task.intensityLabel),
                            onTap: () =>
                                widget.onDailyPlanTaskTap?.call(task),
                          ),
                        ),
                      ),
                    // Then show ExamEvents
                    ...selectedEvents.map(
                      (event) => Padding(
                        padding: const EdgeInsets.only(bottom: 14),
                        child: _AgendaCard(
                          title: event.component.trim().isNotEmpty
                              ? event.component
                              : event.subject,
                          subtitle: _agendaMeta(event),
                          accentColor: _softEventTint(event.subject),
                          onTap: () => widget.onEventTap?.call(event),
                        ),
                      ),
                    ),
                  ],
                ),
            ],
          ],
        ),
      ),
    );
  }

  Color _softEventTint(String subject) {
    final lower = subject.toLowerCase();
    if (lower.contains('math')) return AxonColors.accent.withValues(alpha: 0.2);
    if (lower.contains('physics')) return AxonColors.electricCyan.withValues(alpha: 0.2);
    if (lower.contains('chemistry')) return AxonColors.warning.withValues(alpha: 0.2);
    if (lower.contains('biology')) return AxonColors.success.withValues(alpha: 0.2);
    return AxonColors.surfaceHighlight;
  }

  /// Color coding for daily plan task intensity
  Color _intensityColor(IntensityLevel intensityLabel) {
    switch (intensityLabel) {
      case IntensityLevel.blue: return AxonColors.accent.withValues(alpha: 0.2);
      case IntensityLevel.orange: return AxonColors.warning.withValues(alpha: 0.2);
      case IntensityLevel.red: return AxonColors.error.withValues(alpha: 0.2);
    }
  }
}

enum _CalendarMode {
  month('Month'),
  week('Week'),
  day('Day');

  const _CalendarMode(this.label);
  final String label;
}

class _ModePill extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _ModePill({
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
        height: 54,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: selected ? Colors.white : AxonColors.textSecondary,
          borderRadius: BorderRadius.circular(999),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: AxonColors.textPrimary,
            fontSize: 15,
            fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
          ),
        ),
      ),
    );
  }
}

class _MonthGrid extends StatelessWidget {
  final DateTime selectedDate;
  final List<DateTime> days;
  final bool Function(DateTime date) hasEventOnDay;
  final ValueChanged<DateTime> onSelectDate;

  const _MonthGrid({
    required this.selectedDate,
    required this.days,
    required this.hasEventOnDay,
    required this.onSelectDate,
  });

  @override
  Widget build(BuildContext context) {
    const labels = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
    return Column(
      children: [
        Row(
          children: labels
              .map(
                (label) => Expanded(
                  child: Center(
                    child: Text(
                      label,
                      style: TextStyle(
                        color: AxonColors.textPrimary,
                        fontSize: 15,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  ),
                ),
              )
              .toList(),
        ),
        const SizedBox(height: 10),
        ...List.generate(5, (row) {
          final rowDays = days.skip(row * 7).take(7).toList();
          return Padding(
            padding: EdgeInsets.only(bottom: row == 4 ? 0 : 8),
            child: Row(
              children: rowDays
                  .map(
                    (day) => Expanded(
                      child: _DateCell(
                        date: day,
                        selectedDate: selectedDate,
                        inCurrentMonth: day.month == selectedDate.month,
                        hasEvent: hasEventOnDay(day),
                        onTap: () => onSelectDate(day),
                      ),
                    ),
                  )
                  .toList(),
            ),
          );
        }),
      ],
    );
  }
}

class _WeekStrip extends StatelessWidget {
  final DateTime selectedDate;
  final List<DateTime> days;
  final bool Function(DateTime date) hasEventOnDay;
  final ValueChanged<DateTime> onSelectDate;

  const _WeekStrip({
    required this.selectedDate,
    required this.days,
    required this.hasEventOnDay,
    required this.onSelectDate,
  });

  @override
  Widget build(BuildContext context) {
    const labels = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];
    return Row(
      children: List.generate(days.length, (index) {
        final day = days[index];
        return Expanded(
          child: Padding(
            padding: EdgeInsets.only(right: index == days.length - 1 ? 0 : 8),
            child: GestureDetector(
              onTap: () => onSelectDate(day),
              child: Container(
                height: 94,
                decoration: BoxDecoration(
                  color: _isSameDay(day, selectedDate)
                      ? AxonColors.accent.withValues(alpha: 0.3)
                      : AxonColors.surfaceHighlight,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      labels[index],
                      style: TextStyle(
                        color: _isSameDay(day, selectedDate)
                            ? AxonColors.textPrimary
                            : Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      '${day.day}',
                      style: TextStyle(
                        color: _isSameDay(day, selectedDate)
                            ? AxonColors.textPrimary
                            : Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                        color: hasEventOnDay(day)
                            ? (_isSameDay(day, selectedDate)
                                ? AxonColors.textPrimary
                                : Colors.white)
                            : Colors.transparent,
                        shape: BoxShape.circle,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      }),
    );
  }
}

class _DayTimeline extends StatefulWidget {
  final DateTime date;
  final List<ExamEventModel> events;
  final List<DailyPlanTask> tasks;
  final bool enableDragDrop;
  final Function(ExamEventModel, DateTime)? onEventDrop;
  final Function(DailyPlanTask, DateTime)? onTaskDrop;
  final Function(ExamEventModel)? onEventTap;
  final Function(DailyPlanTask)? onTaskTap;

  const _DayTimeline({
    required this.date,
    required this.events,
    required this.tasks,
    required this.enableDragDrop,
    this.onEventDrop,
    this.onTaskDrop,
    this.onEventTap,
    this.onTaskTap,
  });

  @override
  State<_DayTimeline> createState() => _DayTimelineState();
}

class _DayTimelineState extends State<_DayTimeline> {
  late CalendarController _calendarController;

  @override
  void initState() {
    super.initState();
    _calendarController = CalendarController();
    _calendarController.displayDate = widget.date;
  }

  @override
  void didUpdateWidget(covariant _DayTimeline oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.date != widget.date) {
      _calendarController.displayDate = widget.date;
    }
  }

  @override
  void dispose() {
    _calendarController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final textColor = AxonColors.textPrimary;

    final appointments = <Appointment>[];

    for (final exam in widget.events) {
      appointments.add(Appointment(
        startTime: exam.date,
        endTime: exam.date.add(const Duration(hours: 2)),
        subject: exam.component.trim().isNotEmpty ? '${exam.subject} ${exam.component}' : exam.subject,
        color: AxonColors.accent.withValues(alpha: 0.15), // Soft tint
        isAllDay: false,
        id: 'exam_${exam.id}',
        notes: 'EXAM',
      ));
    }

    for (final task in widget.tasks) {
      appointments.add(Appointment(
        startTime: task.startTime,
        endTime: task.endTime,
        subject: task.title,
        color: _intensityColorForTimeline(task.intensityLabel),
        id: 'task_${task.id}',
        notes: 'TASK',
      ));
    }

    return SfCalendar(
      controller: _calendarController,
      view: CalendarView.day,
      dataSource: _UnifiedCalendarDataSource(appointments),
      headerHeight: 0,
      viewHeaderHeight: 0,
      allowDragAndDrop: widget.enableDragDrop,
      timeSlotViewSettings: TimeSlotViewSettings(
        startHour: 6,
        endHour: 24,
        timeTextStyle: TextStyle(color: textColor, fontSize: 12),
        timeRulerSize: 50,
      ),
      dragAndDropSettings: const DragAndDropSettings(
        allowScroll: true,
        allowNavigation: false,
      ),
      appointmentBuilder: (context, details) {
        final Appointment app = details.appointments.first;
        final isExam = app.notes == 'EXAM';
        
        return Container(
          decoration: BoxDecoration(
            color: app.color,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
                color: isExam ? AxonColors.accent : Colors.transparent, 
                width: 1),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.05),
                blurRadius: 4,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          padding: const EdgeInsets.all(8),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (isExam) Icon(Icons.lock, size: 14, color: AxonColors.textPrimary),
              if (isExam) const SizedBox(width: 4),
              Expanded(
                child: Text(
                  app.subject,
                  style: TextStyle(
                    color: AxonColors.textPrimary,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        );
      },
      onDragEnd: (AppointmentDragEndDetails details) {
        final Appointment app = details.appointment as Appointment;
        final droppedTime = details.droppingTime;
        if (droppedTime != null) {
          if (app.notes == 'EXAM') {
            // Exams are locked, reset by triggering a rebuild
            final examId = (app.id as String).replaceFirst('exam_', '');
            final exam = widget.events.firstWhere((e) => e.id == examId);
            widget.onEventDrop?.call(exam, exam.date);
          } else if (app.notes == 'TASK') {
            final taskId = (app.id as String).replaceFirst('task_', '');
            final task = widget.tasks.firstWhere((t) => t.id == taskId);
            widget.onTaskDrop?.call(task, droppedTime);
          }
        }
      },
      onTap: (CalendarTapDetails details) {
        if (details.appointments != null && details.appointments!.isNotEmpty) {
          final app = details.appointments!.first as Appointment;
          if (app.notes == 'EXAM') {
            final examId = (app.id as String).replaceFirst('exam_', '');
            final exam = widget.events.firstWhere((e) => e.id == examId);
            widget.onEventTap?.call(exam);
          } else if (app.notes == 'TASK') {
            final taskId = (app.id as String).replaceFirst('task_', '');
            final task = widget.tasks.firstWhere((t) => t.id == taskId);
            widget.onTaskTap?.call(task);
          }
        }
      },
    );
  }

  Color _intensityColorForTimeline(IntensityLevel intensityLabel) {
    switch (intensityLabel) {
      case IntensityLevel.blue: return AxonColors.accent.withValues(alpha: 0.2);
      case IntensityLevel.orange: return AxonColors.warning.withValues(alpha: 0.2);
      case IntensityLevel.red: return AxonColors.error.withValues(alpha: 0.2);
    }
  }
}

class _UnifiedCalendarDataSource extends CalendarDataSource {
  _UnifiedCalendarDataSource(List<Appointment> source) {
    appointments = source;
  }
}

class _DateCell extends StatelessWidget {
  final DateTime date;
  final DateTime selectedDate;
  final bool inCurrentMonth;
  final bool hasEvent;
  final VoidCallback onTap;

  const _DateCell({
    required this.date,
    required this.selectedDate,
    required this.inCurrentMonth,
    required this.hasEvent,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final selected = _isSameDay(date, selectedDate);
    final textColor = selected
        ? AxonColors.textPrimary
        : inCurrentMonth
            ? Colors.white
            : AxonColors.textTertiary;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2),
      child: AspectRatio(
        aspectRatio: 0.74,
        child: GestureDetector(
          onTap: onTap,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            decoration: BoxDecoration(
              color: selected ? AxonColors.accent.withValues(alpha: 0.3) : Colors.transparent,
              borderRadius: BorderRadius.circular(22),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '${date.day}',
                  style: TextStyle(
                    color: textColor,
                    fontSize: 16,
                    fontWeight: selected ? FontWeight.w800 : FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 10),
                Container(
                  width: 8,
                  height: 8,
                  decoration: BoxDecoration(
                    color: hasEvent ? textColor : Colors.transparent,
                    shape: BoxShape.circle,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _AgendaCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final Color accentColor;
  final VoidCallback? onTap;

  const _AgendaCard({
    required this.title,
    required this.subtitle,
    required this.accentColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 18),
        decoration: BoxDecoration(
          color: accentColor,
          borderRadius: BorderRadius.circular(22),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: TextStyle(
                color: AxonColors.textPrimary,
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              subtitle,
              style: TextStyle(
                color: AxonColors.textPrimary.withValues(alpha: 0.58),
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

bool _isSameDay(DateTime a, DateTime b) =>
    a.year == b.year && a.month == b.month && a.day == b.day;

class NotionCalendarBottomSheet extends StatefulWidget {
  final DateTime selectedDate;
  final List<String> availableSubjects;
  final Function(String subject, DateTime date, Duration duration)
      onEventCreated;

  const NotionCalendarBottomSheet({
    super.key,
    required this.selectedDate,
    required this.availableSubjects,
    required this.onEventCreated,
  });

  @override
  State<NotionCalendarBottomSheet> createState() =>
      _NotionCalendarBottomSheetState();
}

class _NotionCalendarBottomSheetState extends State<NotionCalendarBottomSheet> {
  String? _selectedSubject;
  Duration _selectedDuration = const Duration(hours: 1);

  final List<Duration> _durationOptions = const [
    Duration(minutes: 30),
    Duration(hours: 1),
    Duration(hours: 2),
    Duration(hours: 3),
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AxonColors.oxfordBlue,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'New Study Session',
                  style: TextStyle(
                    color: AxonColors.textPrimary,
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: Icon(Icons.close, color: AxonColors.textSecondary),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '${_formatDate(widget.selectedDate)} at ${_formatTime(widget.selectedDate)}',
              style: TextStyle(color: AxonColors.textSecondary, fontSize: 14),
            ),
            const SizedBox(height: 20),
            Text('Subject',
                style: TextStyle(color: AxonColors.textSecondary, fontSize: 12)),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: AxonColors.textPrimary.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: DropdownButton<String>(
                value: _selectedSubject,
                hint: Text('Select subject',
                    style: TextStyle(color: AxonColors.textTertiary)),
                dropdownColor: AxonColors.oxfordBlue,
                isExpanded: true,
                underline: const SizedBox(),
                items: widget.availableSubjects
                    .map((s) => DropdownMenuItem(
                        value: s,
                        child: Text(s,
                            style: TextStyle(color: AxonColors.textPrimary))))
                    .toList(),
                onChanged: (v) => setState(() => _selectedSubject = v),
              ),
            ),
            const SizedBox(height: 20),
            Text('Duration',
                style: TextStyle(color: AxonColors.textSecondary, fontSize: 12)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: _durationOptions.map((d) {
                final isSelected = d == _selectedDuration;
                return ChoiceChip(
                  label: Text(_formatDuration(d)),
                  selected: isSelected,
                  selectedColor: AxonColors.accent,
                  backgroundColor: AxonColors.textPrimary.withValues(alpha: 0.1),
                  labelStyle: TextStyle(
                    color: isSelected ? Colors.white : AxonColors.textSecondary,
                    fontSize: 14,
                  ),
                  onSelected: (_) => setState(() => _selectedDuration = d),
                );
              }).toList(),
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _selectedSubject != null
                    ? () {
                        HapticFeedback.mediumImpact();
                        widget.onEventCreated(
                          _selectedSubject!,
                          widget.selectedDate,
                          _selectedDuration,
                        );
                        Navigator.pop(context);
                      }
                    : null,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                child: const Text(
                  'Create Session',
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime date) {
    const weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec'
    ];
    return '${weekdays[date.weekday - 1]}, ${months[date.month - 1]} ${date.day}';
  }

  String _formatTime(DateTime date) {
    return '${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}';
  }

  String _formatDuration(Duration d) {
    if (d.inMinutes < 60) return '${d.inMinutes}m';
    return '${d.inHours}h';
  }
}

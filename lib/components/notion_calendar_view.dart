import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/models.dart';
import '../../models/daily_plan_task.dart';
import '../../theme/app_theme.dart';

class NotionStyleCalendar extends ConsumerStatefulWidget {
  final List<ExamEvent> events;
  final List<DailyPlanTask>? dailyPlanTasks;
  final Function(ExamEvent)? onEventTap;
  final Function(DailyPlanTask)? onDailyPlanTaskTap;
  final Function(ExamEvent, DateTime)? onEventDrop;
  final Function(DateTime)? onTimeSlotTap;
  final bool enableDragDrop;
  final Widget Function(BuildContext, DateTime, List<ExamEvent>)? dayBuilder;

  const NotionStyleCalendar({
    super.key,
    this.events = const [],
    this.dailyPlanTasks,
    this.onEventTap,
    this.onDailyPlanTaskTap,
    this.onEventDrop,
    this.onTimeSlotTap,
    this.enableDragDrop = true,
    this.dayBuilder,
  });

  /// Static helper to create a calendar for DailyPlanTasks
  static Widget buildForDailyPlan({
    Key? key,
    required List<DailyPlanTask> tasks,
    Function(DailyPlanTask)? onTaskTap,
    Function(DateTime)? onTimeSlotTap,
  }) {
    return NotionStyleCalendar(
      key: key,
      dailyPlanTasks: tasks,
      onDailyPlanTaskTap: onTaskTap,
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

  DateTime _initialSelectedDate(List<ExamEvent> events) {
    if (events.isEmpty) return _normalizeDate(DateTime.now());
    final sorted = [...events]
      ..sort((a, b) => a.startDate.compareTo(b.startDate));
    final now = DateTime.now();
    final upcoming = sorted.where((event) => !event.startDate.isBefore(now));
    return _normalizeDate(
      upcoming.isNotEmpty ? upcoming.first.startDate : sorted.first.startDate,
    );
  }

  DateTime _normalizeDate(DateTime date) =>
      DateTime(date.year, date.month, date.day);

  List<ExamEvent> _eventsForDay(DateTime day) {
    final normalized = _normalizeDate(day);
    final items = widget.events.where((event) {
      final eventDay = _normalizeDate(event.startDate);
      return eventDay == normalized;
    }).toList()
      ..sort((a, b) => a.startDate.compareTo(b.startDate));
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

  String _agendaMeta(ExamEvent event) {
    final time = _formatTimeRange(event.startDate, event.endDate);
    if (time.isNotEmpty) return time;
    return '${event.board.toUpperCase()} • ${_headerLabel(event.startDate)}';
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
        isDark ? const Color(0xFF1C1C1E) : const Color(0xFFF5F5F7);

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
              style: const TextStyle(
                color: Colors.white,
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
              _DaySummary(
                date: _selectedDate,
                events: selectedEvents,
                onTap: () => widget.onTimeSlotTap?.call(_selectedDate),
              ),
            const SizedBox(height: 18),
            Divider(color: Colors.white.withValues(alpha: 0.22), height: 1),
            const SizedBox(height: 18),
            if (selectedEvents.isEmpty && selectedDailyPlanTasks.isEmpty)
              Container(
                width: double.infinity,
                padding:
                    const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.94),
                  borderRadius: BorderRadius.circular(22),
                ),
                child: Text(
                  widget.dailyPlanTasks != null
                      ? 'No study blocks planned.'
                      : 'No exams on this date.',
                  style: TextStyle(
                    color: const Color(0xFF4B4C56),
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
                        title: event.label.trim().isNotEmpty
                            ? event.label
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
        ),
      ),
    );
  }

  Color _softEventTint(String subject) {
    final lower = subject.toLowerCase();
    if (lower.contains('math')) return const Color(0xFFF4F4D9);
    if (lower.contains('physics')) return const Color(0xFFE9F4FF);
    if (lower.contains('chemistry')) return const Color(0xFFFBEAF0);
    if (lower.contains('biology')) return const Color(0xFFE9F6E8);
    return Colors.white;
  }

  /// Color coding for daily plan task intensity
  Color _intensityColor(IntensityLevel intensityLabel) {
    switch (intensityLabel) {
      case IntensityLevel.blue: return const Color(0xFFE3F2FD);
      case IntensityLevel.orange: return const Color(0xFFFFF3E0);
      case IntensityLevel.red: return const Color(0xFFFFEBEE);
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
          color: selected ? Colors.white : Colors.white.withValues(alpha: 0.72),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: const Color(0xFF4B4C56),
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
                      style: const TextStyle(
                        color: Colors.white,
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
                      ? const Color(0xFFF5F7D7)
                      : Colors.white.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(24),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      labels[index],
                      style: TextStyle(
                        color: _isSameDay(day, selectedDate)
                            ? const Color(0xFF4B4C56)
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
                            ? const Color(0xFF4B4C56)
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
                                ? const Color(0xFF4B4C56)
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

class _DaySummary extends StatelessWidget {
  final DateTime date;
  final List<ExamEvent> events;
  final VoidCallback onTap;

  const _DaySummary({
    required this.date,
    required this.events,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: const Color(0xFFF5F7D7),
          borderRadius: BorderRadius.circular(28),
        ),
        child: Row(
          children: [
            Container(
              width: 70,
              height: 82,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    '${date.day}',
                    style: const TextStyle(
                      color: Color(0xFF4B4C56),
                      fontSize: 28,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Container(
                    width: 10,
                    height: 10,
                    decoration: const BoxDecoration(
                      color: Color(0xFF4B4C56),
                      shape: BoxShape.circle,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${events.length} exam${events.length == 1 ? '' : 's'} scheduled',
                    style: const TextStyle(
                      color: Color(0xFF4B4C56),
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Tap a card below to open the receipt for this exam day.',
                    style: TextStyle(
                      color: const Color(0xFF4B4C56).withValues(alpha: 0.72),
                      fontSize: 13,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
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
        ? const Color(0xFF4B4C56)
        : inCurrentMonth
            ? Colors.white
            : Colors.white.withValues(alpha: 0.42);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2),
      child: AspectRatio(
        aspectRatio: 0.74,
        child: GestureDetector(
          onTap: onTap,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            decoration: BoxDecoration(
              color: selected ? const Color(0xFFF5F7D7) : Colors.transparent,
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
              style: const TextStyle(
                color: Color(0xFF4B4C56),
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              subtitle,
              style: TextStyle(
                color: const Color(0xFF4B4C56).withValues(alpha: 0.58),
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
                const Text(
                  'New Study Session',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.close, color: Colors.white54),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              '${_formatDate(widget.selectedDate)} at ${_formatTime(widget.selectedDate)}',
              style: const TextStyle(color: Colors.white54, fontSize: 14),
            ),
            const SizedBox(height: 20),
            const Text('Subject',
                style: TextStyle(color: Colors.white70, fontSize: 12)),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: DropdownButton<String>(
                value: _selectedSubject,
                hint: const Text('Select subject',
                    style: TextStyle(color: Colors.white38)),
                dropdownColor: AxonColors.oxfordBlue,
                isExpanded: true,
                underline: const SizedBox(),
                items: widget.availableSubjects
                    .map((s) => DropdownMenuItem(
                        value: s,
                        child: Text(s,
                            style: const TextStyle(color: Colors.white))))
                    .toList(),
                onChanged: (v) => setState(() => _selectedSubject = v),
              ),
            ),
            const SizedBox(height: 20),
            const Text('Duration',
                style: TextStyle(color: Colors.white70, fontSize: 12)),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: _durationOptions.map((d) {
                final isSelected = d == _selectedDuration;
                return ChoiceChip(
                  label: Text(_formatDuration(d)),
                  selected: isSelected,
                  selectedColor: AxonColors.accent,
                  backgroundColor: Colors.white.withValues(alpha: 0.1),
                  labelStyle: TextStyle(
                    color: isSelected ? Colors.white : Colors.white70,
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

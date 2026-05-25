import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:syncfusion_flutter_calendar/calendar.dart';

import '../../models/daily_plan_task.dart';
import '../../services/calendar_bridge.dart';
import '../../theme/app_theme.dart';

class DailyPlanCalendar extends StatefulWidget {
  const DailyPlanCalendar({
    super.key,
    required this.tasks,
    this.onTaskTap,
    this.onTaskDrop,
  });

  final List<DailyPlanTask> tasks;
  final ValueChanged<DailyPlanTask>? onTaskTap;
  final Function(DailyPlanTask, DateTime)? onTaskDrop;

  @override
  State<DailyPlanCalendar> createState() => _DailyPlanCalendarState();
}

class _DailyPlanCalendarState extends State<DailyPlanCalendar> {
  final CalendarController _controller = CalendarController();
  DateTime _displayDate = DateTime.now();

  @override
  Widget build(BuildContext context) {
    final appointments = widget.tasks.map((task) {
          return Appointment(
            startTime: task.startTime,
            endTime: task.endTime,
            subject: task.title,
            notes: '${task.subject} | ${task.paper}\n'
                '${task.phase.label} | ${task.intensityLabel.value.toUpperCase()} intensity\n'
                '${task.reason}\nObjective: ${task.objectiveId}',
            color: _taskColor(task),
            isAllDay: false,
          );
        }).toList();

    final dataSource = CalendarBridge(seedAppointments: appointments);

    return Container(
      height: 400,
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AxonColors.divider.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        children: [
          _buildNotionHeader(),
          Expanded(
            child: SfCalendar(
              controller: _controller,
              view: CalendarView.day,
              dataSource: dataSource,
              headerHeight: 0,
              viewHeaderHeight: 0,
              allowDragAndDrop: true,
              onDragEnd: (details) {
                HapticFeedback.mediumImpact();
              },
              timeSlotViewSettings: TimeSlotViewSettings(
                startHour: 6,
                endHour: 23,
                timeIntervalHeight: 48,
                timeFormat: 'h a',
                timeInterval: const Duration(minutes: 60),
                timeTextStyle: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary,
                  fontSize: 10,
                ),
              ),
              todayHighlightColor: AxonColors.electricCyan,
              selectionDecoration: BoxDecoration(
                color: AxonColors.electricCyan.withValues(alpha: 0.1),
                border: Border.all(
                  color: AxonColors.electricCyan,
                  width: 1.5,
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              appointmentBuilder: (context, details) {
                final appointment = details.appointments.first as Appointment;
                final task = widget.tasks.firstWhere(
                  (item) =>
                      item.title == appointment.subject &&
                      item.startTime == appointment.startTime,
                  orElse: () => widget.tasks.first,
                );
                return _buildNotionAppointment(appointment, task);
              },
              onTap: (details) {
                if (details.appointments == null ||
                    details.appointments!.isEmpty) {
                  return;
                }
                final appointment = details.appointments!.first as Appointment;
                final task = widget.tasks.firstWhere(
                  (item) =>
                      item.title == appointment.subject &&
                      item.startTime == appointment.startTime,
                  orElse: () => widget.tasks.first,
                );
                widget.onTaskTap?.call(task);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNotionHeader() {
    final months = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December'
    ];
    final weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: AxonColors.divider.withValues(alpha: 0.2),
          ),
        ),
      ),
      child: Row(
        children: [
          Text(
            'Today',
            style: GoogleFonts.googleSans(
              color: AxonColors.electricCyan,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(width: 16),
          IconButton(
            icon: const Icon(Icons.chevron_left, size: 20),
            color: AxonColors.textSecondary,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () {
              HapticFeedback.selectionClick();
              setState(() {
                _displayDate = _displayDate.subtract(const Duration(days: 1));
              });
              _controller.displayDate = _displayDate;
            },
          ),
          const SizedBox(width: 8),
          Text(
            '${weekdays[_displayDate.weekday - 1]} ${_displayDate.day} ${months[_displayDate.month - 1]} ${_displayDate.year}',
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            icon: const Icon(Icons.chevron_right, size: 20),
            color: AxonColors.textSecondary,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () {
              HapticFeedback.selectionClick();
              setState(() {
                _displayDate = _displayDate.add(const Duration(days: 1));
              });
              _controller.displayDate = _displayDate;
            },
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: AxonColors.electricCyan.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              '${widget.tasks.length} tasks',
              style: GoogleFonts.googleSans(
                color: AxonColors.electricCyan,
                fontSize: 11,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNotionAppointment(Appointment appointment, DailyPlanTask task) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 1, vertical: 1),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () {
            HapticFeedback.lightImpact();
            widget.onTaskTap?.call(task);
          },
          child: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: appointment.color.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(10),
              border: Border(
                left: BorderSide(
                  color: appointment.color,
                  width: 3,
                ),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  appointment.subject,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  task.objectiveId.isNotEmpty ? task.objectiveId : task.phase.label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.googleSans(
                    color: Colors.white.withValues(alpha: 0.6),
                    fontSize: 9,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _taskColor(DailyPlanTask task) {
    if (task.intensityLabel == IntensityLevel.red ||
        task.intensityScore >= 2.4) {
      return const Color(0xFFD9534F);
    }
    if (task.intensityLabel == IntensityLevel.orange ||
        task.intensityScore >= 1.6) {
      return const Color(0xFFF0AD4E);
    }
    return AxonColors.electricCyan;
  }
}

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:syncfusion_flutter_calendar/calendar.dart';
import '../../services/exam_data_service.dart';

class ExamCalendarNotion extends StatefulWidget {
  final VoidCallback? onTap;
  final VoidCallback? onLongPress;

  const ExamCalendarNotion({super.key, this.onTap, this.onLongPress});

  @override
  State<ExamCalendarNotion> createState() => _ExamCalendarNotionState();
}

class _ExamCalendarNotionState extends State<ExamCalendarNotion> {
  final CalendarController _controller = CalendarController();
  DateTime _displayDate = DateTime.now();
  List<ExamEvent> _exams = [];

  @override
  void initState() {
    super.initState();
    _loadExams();
  }

  Future<void> _loadExams() async {
    final service = ExamDataService();
    await service.initialize();
    if (mounted) {
      setState(() {
        _exams = service.allExams;
      });
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_exams.isEmpty) {
      return _buildNoExamsWidget();
    }

    final appointments = <Appointment>[];
    for (final exam in _exams) {
      appointments.add(Appointment(
        startTime: _startOfDay(exam.date),
        endTime: _endOfDay(exam.date),
        subject: '${exam.subject} ${exam.component}',
        notes:
            'Board: ${exam.board}\nTime: ${exam.startTime} - ${exam.endTime}\nComponent: ${exam.component}',
        color: const Color(0xFF3A86FF),
        isAllDay: true,
      ));
    }

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF121212),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2A2A2A)),
      ),
      child: Column(
        children: [
          _buildNotionHeader(),
          Expanded(
            child: SfCalendar(
              controller: _controller,
              view: CalendarView.month,
              dataSource: ExamCalendarBridge(appointments: appointments),
              headerHeight: 0,
              viewHeaderHeight: 40,
              monthViewSettings: MonthViewSettings(
                appointmentDisplayMode: MonthAppointmentDisplayMode.indicator,
                showTrailingAndLeadingDates: false,
              ),
              monthCellBuilder: (context, details) {
                return _buildMonthCell(details);
              },
              onViewChanged: (details) {
                if (mounted) {
                  setState(() {
                    _displayDate = details.visibleDates.first;
                  });
                }
              },
              onTap: (details) {
                if (details.appointments != null &&
                    details.appointments!.isNotEmpty) {
                  HapticFeedback.lightImpact();
                  widget.onTap?.call();
                }
              },
              onLongPress: (details) {
                if (details.appointments != null &&
                    details.appointments!.isNotEmpty) {
                  HapticFeedback.heavyImpact();
                  widget.onLongPress?.call();
                }
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

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.chevron_left, size: 20),
            color: const Color(0xFF3A86FF),
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () {
              HapticFeedback.selectionClick();
              setState(() {
                _displayDate =
                    DateTime(_displayDate.year, _displayDate.month - 1, 1);
              });
              _controller.displayDate = _displayDate;
            },
          ),
          const SizedBox(width: 8),
          Text(
            '${months[_displayDate.month - 1]} ${_displayDate.year}',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const Spacer(),
          IconButton(
            icon: const Icon(Icons.chevron_right, size: 20),
            color: const Color(0xFF3A86FF),
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(),
            onPressed: () {
              HapticFeedback.selectionClick();
              setState(() {
                _displayDate =
                    DateTime(_displayDate.year, _displayDate.month + 1, 1);
              });
              _controller.displayDate = _displayDate;
            },
          ),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: const Color(0xFF3A86FF).withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              '${_exams.length} exams',
              style: GoogleFonts.googleSans(
                color: const Color(0xFF3A86FF),
                fontSize: 11,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMonthCell(MonthCellDetails details) {
    final date = details.date;
    final isCurrentMonth = date.month == _displayDate.month;
    final hasExams =
        _exams.any((exam) => _startOfDay(exam.date) == _startOfDay(date));

    return Container(
      decoration: BoxDecoration(
        color: isCurrentMonth ? null : const Color(0xFF0A0A0A),
        borderRadius: BorderRadius.circular(12),
      ),
      margin: const EdgeInsets.all(2),
      child: Stack(
        children: [
          Center(
            child: Text(
              '${date.day}',
              style: GoogleFonts.googleSans(
                color: isCurrentMonth
                    ? Colors.white.withValues(alpha: 0.9)
                    : Colors.white.withValues(alpha: 0.3),
                fontSize: 14,
                fontWeight: isCurrentMonth ? FontWeight.w500 : FontWeight.w400,
              ),
            ),
          ),
          if (hasExams)
            Positioned(
              top: 4,
              child: Container(
                width: 4,
                height: 4,
                decoration: const BoxDecoration(
                  color: Color(0xFF3A86FF),
                  shape: BoxShape.circle,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildNoExamsWidget() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0xFF121212),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2A2A2A)),
      ),
      child: Column(
        children: [
          Icon(Icons.event_busy, color: const Color(0xFF666666), size: 48),
          const SizedBox(height: 12),
          Text(
            'NO UPCOMING EXAMS',
            style: TextStyle(
              color: const Color(0xFF666666),
              fontSize: 12,
              letterSpacing: 2,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  DateTime _startOfDay(DateTime date) {
    return DateTime(date.year, date.month, date.day);
  }

  DateTime _endOfDay(DateTime date) {
    return DateTime(date.year, date.month, date.day, 23, 59, 59);
  }
}

class ExamCalendarBridge extends CalendarDataSource {
  ExamCalendarBridge({required List<Appointment> appointments}) {
    this.appointments = appointments;
  }
}

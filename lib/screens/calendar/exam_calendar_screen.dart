import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../services/exam_repository.dart';
import '../../services/exam_schedule_generator.dart';
import '../../services/daily_plan_service.dart';
import '../../services/app_state.dart';
import '../../models/daily_plan_task.dart';
import '../../models/exam_event_model.dart';
import '../../theme/app_theme.dart';

class ExamCalendarScreen extends ConsumerStatefulWidget {
  const ExamCalendarScreen({super.key});

  @override
  ConsumerState<ExamCalendarScreen> createState() => _ExamCalendarScreenState();
}

class _ExamCalendarScreenState extends ConsumerState<ExamCalendarScreen> {
  String _subjectFilter = 'All';
  List<String> _userSubjects = [];
  List<ExamEventModel> _exams = [];
  List<DailyPlanTask> _todayTasks = [];
  List<DailyPlanTask> _weekTasks = [];
  bool _isLoading = true;
  String? _userId;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    final auth = ref.read(authStateProvider);
    final user = auth.user;
    
    if (user != null) {
      _userId = user.uid;
      _userSubjects = user.subjects;
      
      await ExamRepository.instance.initialize(userId: _userId);
      
      if (ExamRepository.instance.totalUpcomingCount == 0 && _userSubjects.isNotEmpty) {
        await ExamScheduleGenerator.instance.saveGeneratedExams();
      }
      
      final dailyPlanService = DailyPlanService();
      final today = DateTime.now();
      final todayStr = today.toIso8601String().split('T').first;
      
      try {
        final todayTasks = await dailyPlanService.getTasksForDate(_userId!, todayStr);
        
        final weekTasks = <DailyPlanTask>[];
        for (int i = 0; i < 7; i++) {
          final date = today.add(Duration(days: i));
          final dateStr = date.toIso8601String().split('T').first;
          final tasks = await dailyPlanService.getTasksForDate(_userId!, dateStr);
          weekTasks.addAll(tasks);
        }
        
        if (mounted) {
          setState(() {
            _exams = ExamRepository.instance.getUpcomingExams();
            _todayTasks = todayTasks;
            _weekTasks = weekTasks;
            _isLoading = false;
          });
        }
      } catch (e) {
        debugPrint('Error loading daily plan: $e');
        if (mounted) {
          setState(() {
            _exams = ExamRepository.instance.getUpcomingExams();
            _isLoading = false;
          });
        }
      }
    } else {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      body: _isLoading
          ? Center(child: CircularProgressIndicator(color: AxonColors.textTertiary))
          : CustomScrollView(
              slivers: [
                _buildHeader(),
                SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
                  sliver: SliverList(
                    delegate: SliverChildBuilderDelegate(
                      (context, index) {
                        if (index == 0) return _buildTodaySection();
                        if (index == 1) return _buildWeekSection();
                        if (index == 2) return _buildFilterBar();
                        if (index == 3 && _exams.isEmpty) return const SizedBox.shrink();
                        
                        final examIndex = index - (_exams.isEmpty ? 3 : 4);
                        if (examIndex < 0 || examIndex >= _exams.length) return null;
                        return _TimelineEventCard(exam: _exams[examIndex]);
                      },
                      childCount: 3 + (_exams.isEmpty ? 0 : _exams.length),
                    ),
                  ),
                ),
              ],
            ),
    );
  }

  Widget _buildHeader() {
    return SliverPersistentHeader(
      pinned: true,
      delegate: _MinimalHeaderDelegate(
        minHeight: 80,
        maxHeight: 140,
        child: Container(
          color: const Color(0xFF0A0A0A).withValues(alpha: 0.95),
          padding: const EdgeInsets.fromLTRB(24, 48, 24, 16),
          alignment: Alignment.bottomLeft,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Flexible(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("EXAM TIMELINE",
                        style: TextStyle(color: Colors.white38, fontSize: 10, letterSpacing: 1.5)),
                    const SizedBox(height: 4),
                    Text(_exams.isEmpty ? "No Exams Scheduled" : "Cambridge CAIE",
                        style: const TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.w500)),
                  ],
                ),
              ),
              Flexible(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text("${_exams.length} EXAMS",
                        style: const TextStyle(color: Colors.white, fontSize: 12, fontFamily: 'monospace')),
                    const SizedBox(height: 4),
                    Text("${_todayTasks.length} TASKS TODAY",
                        style: TextStyle(color: Colors.white.withValues(alpha: 0.5), fontSize: 10)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTodaySection() {
    final now = DateTime.now();
    final greeting = _getGreeting();
    
    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            const Color(0xFF3A86FF).withValues(alpha: 0.15),
            const Color(0xFF3A86FF).withValues(alpha: 0.05),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF3A86FF).withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.wb_sunny_outlined, color: Colors.amber, size: 20),
              const SizedBox(width: 8),
              Text(greeting, style: const TextStyle(color: Colors.white70, fontSize: 14)),
              const Spacer(),
              Text(_formatDate(now), style: const TextStyle(color: Colors.white38, fontSize: 12)),
            ],
          ),
          const SizedBox(height: 16),
          if (_todayTasks.isEmpty)
            Column(
              children: [
                Icon(Icons.event_available_outlined, color: Colors.white24, size: 40),
                const SizedBox(height: 8),
                Text("No tasks scheduled for today",
                    style: TextStyle(color: Colors.white54, fontSize: 14)),
              ],
            )
          else
            ...(_todayTasks.take(3).map((task) => _TodayTaskCard(task: task))),
        ],
      ),
    );
  }

  Widget _buildWeekSection() {
    return Container(
      margin: const EdgeInsets.only(bottom: 24),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF111111),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.calendar_view_week, color: Colors.white38, size: 18),
              const SizedBox(width: 8),
              const Text('THIS WEEK',
                  style: TextStyle(color: Colors.white38, fontSize: 10, letterSpacing: 1.5)),
              const Spacer(),
              Text('${_weekTasks.length} tasks',
                  style: const TextStyle(color: Colors.white54, fontSize: 11)),
            ],
          ),
          const SizedBox(height: 16),
          ...List.generate(7, (i) {
            final date = DateTime.now().add(Duration(days: i));
            final dayTasks = _weekTasks.where((t) => t.date == date.toIso8601String().split('T').first).toList();
            final isToday = DateUtils.isSameDay(date, DateTime.now());
            return _WeekDayRow(date: date, tasks: dayTasks, isToday: isToday);
          }),
        ],
      ),
    );
  }

  Widget _buildFilterBar() {
    if (_userSubjects.isEmpty) return const SizedBox.shrink();

    final subjectNames = _userSubjects.map((code) => _getSubjectName(code)).toList();

    return Container(
      height: 40,
      margin: const EdgeInsets.only(bottom: 16),
      child: ListView(
        scrollDirection: Axis.horizontal,
        children: ['All', ...subjectNames].map((subject) {
          final isActive = _subjectFilter == subject;
          return GestureDetector(
            onTap: () {
              HapticFeedback.selectionClick();
              setState(() {
                _subjectFilter = subject;
                _applyFilters();
              });
            },
            child: Container(
              margin: const EdgeInsets.only(right: 8),
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: isActive ? const Color(0xFF3A86FF).withValues(alpha: 0.2) : Colors.transparent,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: isActive ? const Color(0xFF3A86FF).withValues(alpha: 0.5) : Colors.white.withValues(alpha: 0.1),
                ),
              ),
              child: Center(
                child: Text(subject,
                    style: TextStyle(color: isActive ? const Color(0xFF3A86FF) : Colors.white54, fontSize: 12)),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  void _applyFilters() {
    setState(() {
      _exams = ExamRepository.instance.getUpcomingExams(
        filterSubject: _subjectFilter == 'All' ? null : _subjectFilter,
      );
    });
  }

  String _getGreeting() {
    final hour = DateTime.now().hour;
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  }

  String _formatDate(DateTime date) {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    return '${days[date.weekday - 1]}, ${months[date.month - 1]} ${date.day}';
  }

  String _getSubjectName(String code) {
    const names = {
      '0580': 'Math IGCSE', '0625': 'Physics', '0620': 'Chemistry', '0610': 'Biology',
      '0478': 'Computer Science', '0455': 'Economics', '0450': 'Business', '0500': 'English Lang',
      '4024': 'Maths O Level', '5054': 'Physics O Level', '9709': 'Mathematics',
      '9231': 'Further Maths', '9702': 'Physics', '9701': 'Chemistry', '9700': 'Biology',
      '9618': 'Comp Science', '9708': 'Economics', '9609': 'Business',
    };
    return names[code] ?? code;
  }
}

class _TodayTaskCard extends StatelessWidget {
  final DailyPlanTask task;
  const _TodayTaskCard({required this.task});

  @override
  Widget build(BuildContext context) {
    final startTime = task.startTime;
    final timeStr = '${startTime.hour.toString().padLeft(2, '0')}:${startTime.minute.toString().padLeft(2, '0')}';
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 50,
            child: Text(timeStr, style: const TextStyle(color: Colors.white38, fontSize: 12)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  task.title,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${task.subject} · ${task.taskType.value.replaceAll('_', ' ')}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.5),
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
          _TaskStatusChip(status: task.status),
        ],
      ),
    );
  }
}

class _TaskStatusChip extends StatelessWidget {
  final TaskStatus status;
  const _TaskStatusChip({required this.status});

  @override
  Widget build(BuildContext context) {
    Color color;
    String label;
    switch (status) {
      case TaskStatus.completed:
        color = const Color(0xFF4CAF50);
        label = 'DONE';
      case TaskStatus.rescheduled:
        color = const Color(0xFFFF9800);
        label = 'MOVED';
      default:
        color = Colors.white38;
        label = 'PENDING';
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(label, style: TextStyle(color: color, fontSize: 9, fontWeight: FontWeight.bold)),
    );
  }
}

class _WeekDayRow extends StatelessWidget {
  final DateTime date;
  final List<DailyPlanTask> tasks;
  final bool isToday;

  const _WeekDayRow({required this.date, required this.tasks, required this.isToday});

  @override
  Widget build(BuildContext context) {
    final dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: Colors.white.withValues(alpha: 0.05))),
      ),
      child: Row(
        children: [
          Container(
            width: 45,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(dayNames[date.weekday - 1],
                    style: TextStyle(color: isToday ? const Color(0xFF3A86FF) : Colors.white38, fontSize: 11)),
                Text("${date.day}",
                    style: TextStyle(color: isToday ? Colors.white : Colors.white54, fontSize: 16, fontWeight: FontWeight.w600)),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: tasks.isEmpty
                ? Text('No tasks', style: TextStyle(color: Colors.white.withValues(alpha: 0.3), fontSize: 12))
                : Wrap(
                    spacing: 6,
                    runSpacing: 4,
                    children: tasks.take(3).map((t) => Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.05),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(t.subject, style: const TextStyle(color: Colors.white54, fontSize: 10)),
                    )).toList(),
                  ),
          ),
          if (isToday)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(6),
              ),
              child: const Text('TODAY', style: TextStyle(color: Color(0xFF3A86FF), fontSize: 9, fontWeight: FontWeight.bold)),
            ),
        ],
      ),
    );
  }
}

class _TimelineEventCard extends StatelessWidget {
  final ExamEventModel exam;
  const _TimelineEventCard({required this.exam});

  @override
  Widget build(BuildContext context) {
    final bool isUrgent = exam.daysRemaining <= 14;
    final color = isUrgent ? Colors.white : Colors.white54;

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF111111),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 60,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("${exam.date.day}", style: TextStyle(color: color, fontSize: 24, fontWeight: FontWeight.w400)),
                Text(_getMonth(exam.date.month), style: const TextStyle(color: Colors.white38, fontSize: 12, letterSpacing: 1.0)),
              ],
            ),
          ),
          Container(width: 1, height: 40, color: Colors.white.withValues(alpha: 0.1), margin: const EdgeInsets.only(right: 16)),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(exam.subject, style: TextStyle(color: color, fontSize: 16, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(exam.component, style: const TextStyle(color: Colors.white54, fontSize: 13)),
                if (exam.daysRemaining > 0) ...[
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: isUrgent ? Colors.red.withValues(alpha: 0.2) : Colors.white.withValues(alpha: 0.05),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text("${exam.daysRemaining} DAYS LEFT",
                        style: TextStyle(color: isUrgent ? Colors.red : Colors.white38, fontSize: 10)),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _getMonth(int m) {
    const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
    return m >= 1 && m <= 12 ? months[m - 1] : '';
  }
}

class _MinimalHeaderDelegate extends SliverPersistentHeaderDelegate {
  final double minHeight;
  final double maxHeight;
  final Widget child;
  _MinimalHeaderDelegate({required this.minHeight, required this.maxHeight, required this.child});

  @override
  double get minExtent => minHeight;
  @override
  double get maxExtent => maxHeight;
  @override
  Widget build(BuildContext context, double shrinkOffset, bool overlapsContent) => child;
  @override
  bool shouldRebuild(_MinimalHeaderDelegate oldDelegate) => maxHeight != oldDelegate.maxHeight;
}

// lib/services/exam_planner_service.dart
// ─────────────────────────────────────────────────────────────────
// Exam Planning Service
// Countdown, milestones, past papers, strategy checklists
//
// Fixed vs original:
//  1. getRevisionPlan: raw.isEmpty → raw.isNotEmpty (plan never loaded before)
//  2. _calculatePhases: endDay values are now fixed constants, not live vars
//  3. getCountdown: firstWhere condition had startDay/endDay reversed
//  4. getCountdown: daysRemaining > 90 now correctly returns Foundation phase
//  5. getCountdown: isOver uses <= 0, not < 0
//  6. generateCompressionPlan: cycles through all subjects per day properly
//  7. SharedPreferences cached as _prefs to avoid repeated getInstance() calls
//  8. totalDays derives from config.examStartDate if available, else 90-day window
// ─────────────────────────────────────────────────────────────────

import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'widget_service.dart';

class ExamPlannerService {
  static final ExamPlannerService _instance = ExamPlannerService._internal();
  factory ExamPlannerService() => _instance;
  ExamPlannerService._internal();

  static const String _examConfigKey    = 'exam_config';
  static const String _milestonesKey    = 'exam_milestones';
  static const String _pastPapersKey    = 'past_papers';
  static const String _checklistsKey    = 'strategy_checklists';
  static const String _formulasKey      = 'formula_sheets';
  static const String _revisionPlanKey  = 'revision_plan';

  // Cached prefs – avoids repeated getInstance() across every method call.
  SharedPreferences? _prefs;
  Future<SharedPreferences> get _p async => _prefs ??= await SharedPreferences.getInstance();

  // ─────────────────────────────────────────────────────────────────
  // EXAM CONFIGURATION
  // ─────────────────────────────────────────────────────────────────

  Future<ExamConfig> getExamConfig() async {
    final prefs = await _p;
    final raw = prefs.getString(_examConfigKey);
    if (raw == null || raw.isEmpty) return _defaultConfig();
    try {
      final data = jsonDecode(raw) as Map<String, dynamic>;
      DateTime? examStartDate;
      if (data['examStartDate'] != null) {
        try { examStartDate = DateTime.parse(data['examStartDate']); } catch (_) {}
      }
      return ExamConfig(
        board:         data['board'] ?? '',
        examStartDate: examStartDate,
        subjects:      List<String>.from(data['subjects'] ?? []),
        targetScore:   data['targetScore'] ?? 85,
      );
    } catch (_) {
      return _defaultConfig();
    }
  }

  ExamConfig _defaultConfig() => ExamConfig(
        board:         '',
        examStartDate: null,
        subjects:      [],
        targetScore:   85,
      );

  Future<void> saveExamConfig(ExamConfig config) async {
    final prefs = await _p;
    await prefs.setString(
      _examConfigKey,
      jsonEncode({
        'board':         config.board,
        'examStartDate': config.examStartDate?.toIso8601String(),
        'subjects':      config.subjects,
        'targetScore':   config.targetScore,
      }),
    );
    await WidgetService().syncFromExamConfig(config);
  }

  Future<ExamCountdown> getCountdown() async {
    final config = await getExamConfig();
    if (config.examStartDate == null) {
      return ExamCountdown(
        totalDays:      0,
        daysRemaining:  0,
        weeksRemaining: 0,
        phases:         [],
        currentPhase:   null,
        isOver:         false,
      );
    }

    final now           = DateTime.now();
    final examDate      = config.examStartDate!;
    final daysRemaining = examDate.difference(now).inDays;
    final weeksRemaining = (daysRemaining / 7).floor();

    // FIX: phases now have fixed boundary constants (see _calculatePhases).
    final phases = _calculatePhases();

    // FIX: condition was reversed (endDay/startDay swapped).
    // Each phase covers: endDay ≤ daysRemaining ≤ startDay.
    RevisionPhase? currentPhase;
    if (daysRemaining > 0) {
      currentPhase = phases.firstWhere(
        (p) => daysRemaining <= p.startDay && daysRemaining >= p.endDay,
        // FIX: > 90 days → Foundation (not Final Prep which was the bugged fallback).
        orElse: () => phases.first,
      );
    }

    // FIX: isOver should include the exam day itself (<= 0).
    return ExamCountdown(
      // FIX: total window = distance from "90 days before exam" to exam date,
      // OR from today if today is already within the window.
      totalDays:      examDate
          .difference(
            now.isBefore(examDate.subtract(const Duration(days: 90)))
                ? examDate.subtract(const Duration(days: 90))
                : now,
          )
          .inDays
          .abs(),
      daysRemaining:  daysRemaining,
      weeksRemaining: weeksRemaining.clamp(0, 9999),
      phases:         phases,
      currentPhase:   currentPhase,
      isOver:         daysRemaining <= 0,
    );
  }

  // FIX: phase boundaries are fixed constants, not derived from live daysRemaining.
  // Phase matching uses: daysRemaining ∈ [endDay, startDay].
  List<RevisionPhase> _calculatePhases() {
    return [
      RevisionPhase(
        name:        'Foundation',
        startDay:    90,
        endDay:      61,
        description: 'Build strong foundations. Cover all chapters.',
        dailyHours:  4,
        focus:       'Learning new concepts',
        color:       0xFF4CAF50,
      ),
      RevisionPhase(
        name:        'Intensive Revision',
        startDay:    60,
        endDay:      31,
        description: 'Deep dive into weak areas. Practice problems.',
        dailyHours:  5,
        focus:       'Problem solving',
        color:       0xFF2196F3,
      ),
      RevisionPhase(
        name:        'Mock Tests',
        startDay:    30,
        endDay:      8,
        description: 'Full-length papers. Time management practice.',
        dailyHours:  6,
        focus:       'Exam simulation',
        color:       0xFFFF9800,
      ),
      RevisionPhase(
        name:        'Final Prep',
        startDay:    7,
        endDay:      0,
        description: 'Light review. Formula memorization. Rest.',
        dailyHours:  3,
        focus:       'Consolidation',
        color:       0xFFE91E63,
      ),
    ];
  }

  // ─────────────────────────────────────────────────────────────────
  // MILESTONES
  // ─────────────────────────────────────────────────────────────────

  Future<List<Milestone>> getMilestones() async {
    final prefs = await _p;
    final raw = prefs.getString(_milestonesKey);
    if (raw == null || raw.isEmpty) return _generateAutoMilestones();
    try {
      final list = jsonDecode(raw) as List;
      return list.map((e) => Milestone.fromJson(e)).toList();
    } catch (_) {
      return _generateAutoMilestones();
    }
  }

  List<Milestone> _generateAutoMilestones() {
    return [
      Milestone(id: '1', title: 'First Mock Test', description: 'Complete your first mock exam', targetDay: 75, isCompleted: false, type: MilestoneType.mockTest, metricKey: 'mockCount', targetValue: 1),
      Milestone(id: '2', title: '3 Past Papers', description: 'Complete 3 mock exams', targetDay: 60, isCompleted: false, type: MilestoneType.practice, metricKey: 'mockCount', targetValue: 3),
      Milestone(id: '3', title: '10 Hours Studied', description: 'Log 10 hours of study time', targetDay: 50, isCompleted: false, type: MilestoneType.syllabus, metricKey: 'studySeconds', targetValue: 36000),
      Milestone(id: '4', title: '50 Hours Studied', description: 'Log 50 hours of study time', targetDay: 30, isCompleted: false, type: MilestoneType.syllabus, metricKey: 'studySeconds', targetValue: 180000),
      Milestone(id: '5', title: 'Weak Points Fixed', description: 'Improve 5 weak chapters', targetDay: 40, isCompleted: false, type: MilestoneType.analysis, metricKey: 'improvements', targetValue: 5),
      Milestone(id: '6', title: 'Speed Practice', description: 'Complete a timed mock', targetDay: 21, isCompleted: false, type: MilestoneType.timing, metricKey: 'timedMock', targetValue: 1),
      Milestone(id: '7', title: 'Consolidation', description: 'Final review phase', targetDay: 7, isCompleted: false, type: MilestoneType.consolidation, metricKey: 'daysRemaining', targetValue: 7),
    ];
  }

  Future<List<Milestone>> getAutoMeasuredMilestones() async {
    final milestones = await getMilestones();
    final updated = <Milestone>[];
    
    for (final m in milestones) {
      final isComplete = _checkMetricCompletion(m);
      updated.add(m.copyWith(
        isCompleted: isComplete,
        completedAt: isComplete ? DateTime.now() : null,
      ));
    }
    return updated;
  }

  bool _checkMetricCompletion(Milestone m) {
    switch (m.metricKey) {
      case 'mockCount':
        return _mockCount >= m.targetValue;
      case 'studySeconds':
        return _totalStudySeconds >= m.targetValue;
      case 'improvements':
        return _weakChapterImprovements >= m.targetValue;
      case 'timedMock':
        return _timedMocksCompleted >= m.targetValue;
      case 'daysRemaining':
        return _daysRemaining <= m.targetValue;
      default:
        return m.isCompleted;
    }
  }

  int get _mockCount => _metricsSnapshot.mockExamsCompleted;
  int get _totalStudySeconds => _metricsSnapshot.totalStudySeconds;
  int get _weakChapterImprovements => _metricsSnapshot.chaptersImproved;
  int get _timedMocksCompleted => _metricsSnapshot.timedMocksCompleted;
  int get _daysRemaining => _countdownDaysRemaining;

  int _countdownDaysRemaining = 0;
  _MetricsSnapshot _metricsSnapshot = _MetricsSnapshot.empty();

  Future<void> syncMetrics({
    required int mockCount,
    required int studySeconds,
    required int improvements,
    required int timedMocks,
    required int daysRemaining,
  }) async {
    _metricsSnapshot = _MetricsSnapshot(
      mockExamsCompleted: mockCount,
      totalStudySeconds: studySeconds,
      chaptersImproved: improvements,
      timedMocksCompleted: timedMocks,
    );
    _countdownDaysRemaining = daysRemaining;
    
    final autoMilestones = await getAutoMeasuredMilestones();
    final prefs = await _p;
    await prefs.setString(_milestonesKey, jsonEncode(autoMilestones.map((m) => m.toJson()).toList()));
  }

  Future<void> completeMilestone(String id) async {
    final milestones = await getMilestones();
    final updated = milestones.map((m) => m.id == id ? m.copyWith(isCompleted: true, completedAt: DateTime.now()) : m).toList();
    final prefs = await _p;
    await prefs.setString(_milestonesKey, jsonEncode(updated.map((m) => m.toJson()).toList()));
  }

  Future<void> addMilestone(Milestone milestone) async {
    final milestones = await getMilestones();
    milestones.add(milestone);
    final prefs = await _p;
    await prefs.setString(_milestonesKey, jsonEncode(milestones.map((m) => m.toJson()).toList()));
  }

  Future<void> toggleMilestone(String id) async {
    final milestones = await getMilestones();
    final updated = milestones.map((m) => m.id == id ? m.copyWith(isCompleted: !m.isCompleted) : m).toList();
    final prefs = await _p;
    await prefs.setString(_milestonesKey, jsonEncode(updated.map((m) => m.toJson()).toList()));
  }

  Future<void> deleteMilestone(String id) async {
    final milestones = await getMilestones();
    milestones.removeWhere((m) => m.id == id);
    final prefs = await _p;
    await prefs.setString(_milestonesKey, jsonEncode(milestones.map((m) => m.toJson()).toList()));
  }

  // ─────────────────────────────────────────────────────────────────
  // PAST PAPERS
  // ─────────────────────────────────────────────────────────────────

  Future<List<PastPaperPack>> getPastPapers() async {
    final config = await getExamConfig();
    final prefs  = await _p;
    final raw    = prefs.getString(_pastPapersKey);

    if (raw != null && raw.isNotEmpty) {
      try {
        final list = jsonDecode(raw) as List;
        return list.map((e) => PastPaperPack.fromJson(e)).toList();
      } catch (_) {}
    }

    if (config.board.isNotEmpty) {
      return _generateDefaultPapers(config.board, config.subjects);
    }
    return [];
  }

  List<PastPaperPack> _generateDefaultPapers(String board, List<String> subjects) {
    final papers   = <PastPaperPack>[];
    final years    = [2024, 2023, 2022, 2021, 2020];
    final variants = ['Main', 'Compartment', 'Model'];

    for (final subject in subjects) {
      for (final year in years) {
        for (final variant in variants) {
          papers.add(PastPaperPack(
            id:          '${board}_${subject}_${year}_$variant'.toLowerCase().replaceAll(' ', '_'),
            subject:     subject,
            board:       board,
            year:        year,
            variant:     variant,
            paperNumber: 1,
            maxMarks:    100,
            duration:    180,
            isCompleted: false,
          ));
        }
      }
    }
    return papers;
  }

  Future<void> savePastPaperResult(String id, int score, String? filePath) async {
    final papers  = await getPastPapers();
    final updated = papers.map((p) => p.id == id
        ? p.copyWith(isCompleted: true, score: score, attemptedAt: DateTime.now(), filePath: filePath)
        : p).toList();
    final prefs = await _p;
    await prefs.setString(_pastPapersKey, jsonEncode(updated.map((p) => p.toJson()).toList()));
  }

  Future<List<PastPaperPack>> getPapersBySubject(String subject) async {
    final papers = await getPastPapers();
    return papers.where((p) => p.subject == subject).toList()
      ..sort((a, b) {
        final yearCompare = b.year.compareTo(a.year);
        return yearCompare != 0 ? yearCompare : a.variant.compareTo(b.variant);
      });
  }

  // ─────────────────────────────────────────────────────────────────
  // STRATEGY CHECKLISTS
  // ─────────────────────────────────────────────────────────────────

  Future<Map<String, StrategyChecklist>> getChecklists() async {
    final prefs = await _p;
    final raw   = prefs.getString(_checklistsKey);
    if (raw != null && raw.isNotEmpty) {
      try {
        final map = jsonDecode(raw) as Map<String, dynamic>;
        return map.map((k, v) => MapEntry(k, StrategyChecklist.fromJson(v)));
      } catch (_) {}
    }
    return _getDefaultChecklists();
  }

  Map<String, StrategyChecklist> _getDefaultChecklists() {
    return {
      'general': StrategyChecklist(subject: 'General', items: [
        ChecklistItem(id: '1',  text: 'Know exam pattern and marking scheme',  isChecked: false),
        ChecklistItem(id: '2',  text: 'Practice time management',              isChecked: false),
        ChecklistItem(id: '3',  text: 'Read all questions before answering',   isChecked: false),
        ChecklistItem(id: '4',  text: 'Attempt easier questions first',        isChecked: false),
        ChecklistItem(id: '5',  text: 'Keep 10 mins for revision',             isChecked: false),
        ChecklistItem(id: '6',  text: 'Write legibly and neatly',              isChecked: false),
        ChecklistItem(id: '7',  text: 'Draw diagrams where applicable',        isChecked: false),
        ChecklistItem(id: '8',  text: 'Show all working steps',                isChecked: false),
        ChecklistItem(id: '9',  text: 'Double-check calculations',             isChecked: false),
        ChecklistItem(id: '10', text: 'Attempt all questions (no blanks)',     isChecked: false),
      ]),
      'mathematics': StrategyChecklist(subject: 'Mathematics', items: [
        ChecklistItem(id: '1', text: 'Memorize all formulas',                  isChecked: false),
        ChecklistItem(id: '2', text: 'Practice derivation of formulas',        isChecked: false),
        ChecklistItem(id: '3', text: 'Solve 10 questions daily',               isChecked: false),
        ChecklistItem(id: '4', text: 'Focus on trigonometry and calculus',     isChecked: false),
        ChecklistItem(id: '5', text: 'Practice graph plotting',                isChecked: false),
        ChecklistItem(id: '6', text: 'Learn shortcut methods',                 isChecked: false),
        ChecklistItem(id: '7', text: 'Attempt proof questions last',           isChecked: false),
      ]),
      'physics': StrategyChecklist(subject: 'Physics', items: [
        ChecklistItem(id: '1', text: 'Understand concepts, not just formulas', isChecked: false),
        ChecklistItem(id: '2', text: 'Practice numerical problems daily',      isChecked: false),
        ChecklistItem(id: '3', text: 'Learn SI units and dimensions',          isChecked: false),
        ChecklistItem(id: '4', text: 'Draw free body diagrams',                isChecked: false),
        ChecklistItem(id: '5', text: 'Memorize derivations',                   isChecked: false),
        ChecklistItem(id: '6', text: 'Practice unit conversions',              isChecked: false),
      ]),
      'chemistry': StrategyChecklist(subject: 'Chemistry', items: [
        ChecklistItem(id: '1', text: 'Learn all reactions and equations',      isChecked: false),
        ChecklistItem(id: '2', text: 'Practice balancing equations',           isChecked: false),
        ChecklistItem(id: '3', text: 'Memorize periodic table trends',         isChecked: false),
        ChecklistItem(id: '4', text: 'Know named reactions',                   isChecked: false),
        ChecklistItem(id: '5', text: 'Practice organic chemistry mechanisms',  isChecked: false),
        ChecklistItem(id: '6', text: 'Learn mole concept thoroughly',          isChecked: false),
      ]),
    };
  }

  Future<void> updateChecklistItem(String subject, String itemId, bool isChecked) async {
    final checklists = await getChecklists();
    if (checklists.containsKey(subject)) {
      final checklist    = checklists[subject]!;
      final updatedItems = checklist.items
          .map((item) => item.id == itemId ? item.copyWith(isChecked: isChecked) : item)
          .toList();
      checklists[subject] = StrategyChecklist(subject: subject, items: updatedItems);
    }
    final prefs = await _p;
    await prefs.setString(_checklistsKey, jsonEncode(checklists.map((k, v) => MapEntry(k, v.toJson()))));
  }

  Future<void> addChecklistItem(String subject, ChecklistItem item) async {
    final checklists = await getChecklists();
    if (checklists.containsKey(subject)) {
      final checklist = checklists[subject]!;
      final updatedItems = [...checklist.items, item];
      checklists[subject] = StrategyChecklist(subject: subject, items: updatedItems);
    } else {
      checklists[subject] = StrategyChecklist(subject: subject, items: [item]);
    }
    final prefs = await _p;
    await prefs.setString(_checklistsKey, jsonEncode(checklists.map((k, v) => MapEntry(k, v.toJson()))));
  }

  Future<void> toggleChecklistItem(String subject, String itemId) async {
    final checklists = await getChecklists();
    if (checklists.containsKey(subject)) {
      final checklist = checklists[subject]!;
      final updatedItems = checklist.items
          .map((item) => item.id == itemId ? item.copyWith(isChecked: !item.isChecked) : item)
          .toList();
      checklists[subject] = StrategyChecklist(subject: subject, items: updatedItems);
    }
    final prefs = await _p;
    await prefs.setString(_checklistsKey, jsonEncode(checklists.map((k, v) => MapEntry(k, v.toJson()))));
  }

  Future<void> deleteChecklistItem(String subject, String itemId) async {
    final checklists = await getChecklists();
    if (checklists.containsKey(subject)) {
      final checklist = checklists[subject]!;
      final updatedItems = checklist.items.where((item) => item.id != itemId).toList();
      checklists[subject] = StrategyChecklist(subject: subject, items: updatedItems);
    }
    final prefs = await _p;
    await prefs.setString(_checklistsKey, jsonEncode(checklists.map((k, v) => MapEntry(k, v.toJson()))));
  }

  // ─────────────────────────────────────────────────────────────────
  // FORMULA SHEETS
  // ─────────────────────────────────────────────────────────────────

  Future<Map<String, FormulaSheet>> getFormulaSheets() async {
    final prefs = await _p;
    final raw   = prefs.getString(_formulasKey);
    if (raw != null && raw.isNotEmpty) {
      try {
        final map = jsonDecode(raw) as Map<String, dynamic>;
        return map.map((k, v) => MapEntry(k, FormulaSheet.fromJson(v)));
      } catch (_) {}
    }
    return _getDefaultFormulas();
  }

  Map<String, FormulaSheet> _getDefaultFormulas() {
    return {
      'algebra': FormulaSheet(subject: 'Mathematics', topic: 'Algebra', formulas: [
        FormulaItem(name: 'Quadratic Formula',  expression: r'x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}'),
        FormulaItem(name: 'Sum of AP',          expression: r'S = \frac{n}{2}(2a + (n-1)d)'),
        FormulaItem(name: 'Sum of GP',          expression: r'S = \frac{a(r^n - 1)}{r - 1}'),
        FormulaItem(name: 'Binomial',           expression: r'(a+b)^n = \sum_{r=0}^n C(n,r) a^{n-r} b^r'),
        FormulaItem(name: 'Permutation',        expression: r'P(n,r) = \frac{n!}{(n-r)!}'),
        FormulaItem(name: 'Combination',        expression: r'C(n,r) = \frac{n!}{r!(n-r)!}'),
      ]),
      'trigonometry': FormulaSheet(subject: 'Mathematics', topic: 'Trigonometry', formulas: [
        FormulaItem(name: 'sin²θ + cos²θ',   expression: r'= 1'),
        FormulaItem(name: '1 + tan²θ',        expression: r'= \sec^2\theta'),
        FormulaItem(name: 'sin(A+B)',          expression: r'= \sin A \cos B + \cos A \sin B'),
        FormulaItem(name: 'cos(A+B)',          expression: r'= \cos A \cos B - \sin A \sin B'),
        FormulaItem(name: 'tan(A+B)',          expression: r'= \frac{\tan A + \tan B}{1 - \tan A \tan B}'),
        FormulaItem(name: 'sin 2A',            expression: r'= 2\sin A \cos A'),
        FormulaItem(name: 'cos 2A',            expression: r'= \cos^2 A - \sin^2 A'),
      ]),
      'calculus': FormulaSheet(subject: 'Mathematics', topic: 'Calculus', formulas: [
        FormulaItem(name: 'd/dx(xⁿ)',          expression: r'= nx^{n-1}'),
        FormulaItem(name: 'd/dx(eˣ)',           expression: r'= e^x'),
        FormulaItem(name: 'd/dx(ln x)',         expression: r'= \frac{1}{x}'),
        FormulaItem(name: '∫ eˣ dx',           expression: r'= e^x + C'),
        FormulaItem(name: '∫ 1/x dx',          expression: r'= \ln|x| + C'),
        FormulaItem(name: '∫ sin x dx',        expression: r'= -\cos x + C'),
      ]),
      'physics_mechanics': FormulaSheet(subject: 'Physics', topic: 'Mechanics', formulas: [
        FormulaItem(name: 'Velocity',           expression: r'v = u + at'),
        FormulaItem(name: 'Displacement',       expression: r's = ut + \frac{1}{2}at^2'),
        FormulaItem(name: 'Momentum',           expression: r'p = mv'),
        FormulaItem(name: 'Force',              expression: r'F = ma'),
        FormulaItem(name: 'Kinetic Energy',     expression: r'KE = \frac{1}{2}mv^2'),
        FormulaItem(name: 'Potential Energy',   expression: r'PE = mgh'),
        FormulaItem(name: 'Work',               expression: r'W = Fd \cos\theta'),
        FormulaItem(name: 'Power',              expression: r'P = \frac{W}{t}'),
      ]),
      'physics_electricity': FormulaSheet(subject: 'Physics', topic: 'Electricity', formulas: [
        FormulaItem(name: 'Ohm\'s Law',         expression: r'V = IR'),
        FormulaItem(name: 'Power',              expression: r'P = VI = I^2R = \frac{V^2}{R}'),
        FormulaItem(name: 'Resistance',         expression: r'R = \frac{\rho L}{A}'),
        FormulaItem(name: 'Capacitance',        expression: r'C = \frac{Q}{V}'),
        FormulaItem(name: 'Energy',             expression: r'E = \frac{1}{2}CV^2'),
      ]),
    };
  }

  Future<void> toggleFormulaFavorite(String topic, String formulaName) async {
    final sheets = await getFormulaSheets();
    if (sheets.containsKey(topic)) {
      final sheet          = sheets[topic]!;
      final updatedFormulas = sheet.formulas.map((f) => f.name == formulaName
          ? f.copyWith(isFavorite: !f.isFavorite) : f).toList();
      sheets[topic] = FormulaSheet(subject: sheet.subject, topic: sheet.topic, formulas: updatedFormulas);
    }
    final prefs = await _p;
    await prefs.setString(_formulasKey, jsonEncode(sheets.map((k, v) => MapEntry(k, v.toJson()))));
  }

  // ─────────────────────────────────────────────────────────────────
  // REVISION PLAN
  // ─────────────────────────────────────────────────────────────────

  Future<RevisionPlan> getRevisionPlan() async {
    final prefs = await _p;
    final raw   = prefs.getString(_revisionPlanKey);

    // FIX: was `raw.isEmpty` — condition was inverted, plan never loaded.
    if (raw != null && raw.isNotEmpty) {
      try {
        return RevisionPlan.fromJson(jsonDecode(raw));
      } catch (_) {}
    }

    return RevisionPlan(
      compressionMode: true,
      dailyTargets:    [],
      completedDays:   [],
      lastUpdated:     null,
    );
  }

  // FIX: original algorithm broke after chaptersPerDay entries and only ever
  // assigned the first subject. Now it cycles round-robin across all subjects.
  Future<void> generateCompressionPlan({
    required List<String> subjects,
    required int daysRemaining,
    required Map<String, int> weakChapterCount,
  }) async {
    final dailyTargets  = <DailyRevisionTarget>[];
    final totalChapters = weakChapterCount.values.fold(0, (a, b) => a + b);
    if (totalChapters == 0 || daysRemaining <= 0) return;

    final chaptersPerDay = (totalChapters / daysRemaining).ceil().clamp(1, 10);

    // Build a flat list of (subject, chapter) pairs to distribute.
    final allChapters = <({String subject, int chapterNum})>[];
    int maxChapters = 0;
    for (final val in weakChapterCount.values) {
      if (val > maxChapters) maxChapters = val;
    }
    for (int c = 1; c <= maxChapters; c++) {
      for (final entry in weakChapterCount.entries) {
        if (c <= entry.value) {
          allChapters.add((subject: entry.key, chapterNum: c));
        }
      }
    }

    int chapterIndex = 0;
    for (int day = 0; day < daysRemaining && day < 30; day++) {
      final date       = DateTime.now().add(Duration(days: day));
      final dayTargets = <ChapterRevision>[];

      // FIX: pull chaptersPerDay items from the flat list, cycling correctly.
      for (int i = 0; i < chaptersPerDay && chapterIndex < allChapters.length; i++) {
        final item = allChapters[chapterIndex++];
        dayTargets.add(ChapterRevision(
          subject:  item.subject,
          chapter:  'Chapter ${item.chapterNum}',
          priority: chaptersPerDay - i,
          status:   RevisionStatus.pending,
        ));
      }

      if (dayTargets.isEmpty) break;

      dailyTargets.add(DailyRevisionTarget(
        date:        date,
        targetHours: day < 7 ? 6 : (day < 14 ? 5 : 4),
        topics:      dayTargets,
        notes:       '',
      ));
    }

    final plan = RevisionPlan(
      compressionMode: true,
      dailyTargets:    dailyTargets,
      completedDays:   [],
      lastUpdated:     DateTime.now(),
    );
    final prefs = await _p;
    await prefs.setString(_revisionPlanKey, jsonEncode(plan.toJson()));
  }

  Future<void> markDayComplete(int dayIndex, bool isComplete) async {
    final plan          = await getRevisionPlan();
    final completedDays = List<int>.from(plan.completedDays);

    if (isComplete && !completedDays.contains(dayIndex)) {
      completedDays.add(dayIndex);
    } else if (!isComplete) {
      completedDays.remove(dayIndex);
    }

    final updatedPlan = RevisionPlan(
      compressionMode: plan.compressionMode,
      dailyTargets:    plan.dailyTargets,
      completedDays:   completedDays,
      lastUpdated:     DateTime.now(),
    );
    final prefs = await _p;
    await prefs.setString(_revisionPlanKey, jsonEncode(updatedPlan.toJson()));
  }

  Future<void> saveRevisionPlan(RevisionPlan plan) async {
    final prefs = await _p;
    await prefs.setString(_revisionPlanKey, jsonEncode(plan.toJson()));
  }
}

// ─────────────────────────────────────────────────────────────────
// DATA MODELS
// ─────────────────────────────────────────────────────────────────

class ExamConfig {
  final String       board;
  final DateTime?    examStartDate;
  final List<String> subjects;
  final int          targetScore;

  const ExamConfig({
    required this.board,
    this.examStartDate,
    required this.subjects,
    required this.targetScore,
  });
}

class ExamCountdown {
  final int            totalDays;
  final int            daysRemaining;
  final int            weeksRemaining;
  final List<RevisionPhase> phases;
  final RevisionPhase? currentPhase;
  final bool           isOver;

  const ExamCountdown({
    required this.totalDays,
    required this.daysRemaining,
    required this.weeksRemaining,
    required this.phases,
    this.currentPhase,
    required this.isOver,
  });
}

class RevisionPhase {
  final String name;
  final int    startDay;
  final int    endDay;
  final String description;
  final int    dailyHours;
  final String focus;
  final int    color;

  const RevisionPhase({
    required this.name,
    required this.startDay,
    required this.endDay,
    required this.description,
    required this.dailyHours,
    required this.focus,
    required this.color,
  });
}

class Milestone {
  final String        id;
  final String        title;
  final String        description;
  final int           targetDay;
  final bool          isCompleted;
  final MilestoneType type;
  final DateTime?     completedAt;
  final String?      metricKey;
  final int          targetValue;

  const Milestone({
    required this.id,
    required this.title,
    required this.description,
    required this.targetDay,
    required this.isCompleted,
    required this.type,
    this.completedAt,
    this.metricKey,
    this.targetValue = 0,
  });

  Milestone copyWith({bool? isCompleted, DateTime? completedAt, String? metricKey, int? targetValue}) => Milestone(
        id:          id,
        title:       title,
        description: description,
        targetDay:   targetDay,
        isCompleted: isCompleted ?? this.isCompleted,
        type:        type,
        completedAt: completedAt ?? this.completedAt,
      );

  factory Milestone.fromJson(Map<String, dynamic> json) {
    DateTime? completedAt;
    if (json['completedAt'] != null) {
      try { completedAt = DateTime.parse(json['completedAt']); } catch (_) {}
    }
    return Milestone(
      id:          json['id'],
      title:       json['title'],
      description: json['description'],
      targetDay:   json['targetDay'],
      isCompleted: json['isCompleted'] ?? false,
      type: MilestoneType.values.firstWhere(
        (t) => t.name == json['type'],
        orElse: () => MilestoneType.general,
      ),
      completedAt: completedAt,
      metricKey: json['metricKey'],
      targetValue: json['targetValue'] ?? 0,
    );
  }

  Map<String, dynamic> toJson() => {
        'id':          id,
        'title':       title,
        'description': description,
        'targetDay':   targetDay,
        'isCompleted': isCompleted,
        'type':        type.name,
        'completedAt': completedAt?.toIso8601String(),
        'metricKey':   metricKey,
        'targetValue': targetValue,
      };
}

enum MilestoneType { syllabus, mockTest, analysis, practice, formula, timing, consolidation, general }

class _MetricsSnapshot {
  final int mockExamsCompleted;
  final int totalStudySeconds;
  final int chaptersImproved;
  final int timedMocksCompleted;

  _MetricsSnapshot({
    required this.mockExamsCompleted,
    required this.totalStudySeconds,
    required this.chaptersImproved,
    required this.timedMocksCompleted,
  });

  factory _MetricsSnapshot.empty() => _MetricsSnapshot(
    mockExamsCompleted: 0,
    totalStudySeconds: 0,
    chaptersImproved: 0,
    timedMocksCompleted: 0,
  );
}

class PastPaper {
  final String id;
  final String subject;
  final String board;
  final int year;
  final String variant;
  final int paperNumber;
  final int maxMarks;
  final int duration;
  final bool isCompleted;
  final int? score;
  final DateTime? attemptedAt;
  final String? filePath;

  const PastPaper({
    required this.id,
    required this.subject,
    required this.board,
    required this.year,
    required this.variant,
    required this.paperNumber,
    required this.maxMarks,
    required this.duration,
    this.isCompleted = false,
    this.score,
    this.attemptedAt,
    this.filePath,
  });

  PastPaper copyWith({
    bool? isCompleted,
    int? score,
    DateTime? attemptedAt,
    String? filePath,
  }) => PastPaper(
    id: id,
    subject: subject,
    board: board,
    year: year,
    variant: variant,
    paperNumber: paperNumber,
    maxMarks: maxMarks,
    duration: duration,
    isCompleted: isCompleted ?? this.isCompleted,
    score: score ?? this.score,
    attemptedAt: attemptedAt ?? this.attemptedAt,
    filePath: filePath ?? this.filePath,
  );
}

class RevisionDay {
  final DateTime date;
  final List<String> subjects;
  final List<String> topics;
  final bool isComplete;

  const RevisionDay({
    required this.date,
    required this.subjects,
    this.topics = const [],
    this.isComplete = false,
  });

  RevisionDay copyWith({
    DateTime? date,
    List<String>? subjects,
    List<String>? topics,
    bool? isComplete,
  }) => RevisionDay(
    date: date ?? this.date,
    subjects: subjects ?? this.subjects,
    topics: topics ?? this.topics,
    isComplete: isComplete ?? this.isComplete,
  );
}

class PastPaperPack {
  final String    id;
  final String    subject;
  final String    board;
  final int       year;
  final String    variant;
  final int       paperNumber;
  final int       maxMarks;
  final int       duration;
  final bool      isCompleted;
  final int?      score;
  final DateTime? attemptedAt;
  final String?   filePath;

  const PastPaperPack({
    required this.id,
    required this.subject,
    required this.board,
    required this.year,
    required this.variant,
    required this.paperNumber,
    required this.maxMarks,
    required this.duration,
    required this.isCompleted,
    this.score,
    this.attemptedAt,
    this.filePath,
  });

  PastPaperPack copyWith({
    bool?     isCompleted,
    int?      score,
    DateTime? attemptedAt,
    String?   filePath,
  }) =>
      PastPaperPack(
        id:          id,
        subject:     subject,
        board:       board,
        year:        year,
        variant:     variant,
        paperNumber: paperNumber,
        maxMarks:    maxMarks,
        duration:    duration,
        isCompleted: isCompleted ?? this.isCompleted,
        score:       score ?? this.score,
        attemptedAt: attemptedAt ?? this.attemptedAt,
        filePath:    filePath ?? this.filePath,
      );

  factory PastPaperPack.fromJson(Map<String, dynamic> json) {
    DateTime? attemptedAt;
    if (json['attemptedAt'] != null) {
      try { attemptedAt = DateTime.parse(json['attemptedAt']); } catch (_) {}
    }
    return PastPaperPack(
      id:          json['id'],
      subject:     json['subject'],
      board:       json['board'],
      year:        json['year'],
      variant:     json['variant'],
      paperNumber: json['paperNumber'] ?? 1,
      maxMarks:    json['maxMarks']    ?? 100,
      duration:    json['duration']    ?? 180,
      isCompleted: json['isCompleted'] ?? false,
      score:       json['score'],
      attemptedAt: attemptedAt,
      filePath:    json['filePath'],
    );
  }

  Map<String, dynamic> toJson() => {
        'id':          id,
        'subject':     subject,
        'board':       board,
        'year':        year,
        'variant':     variant,
        'paperNumber': paperNumber,
        'maxMarks':    maxMarks,
        'duration':    duration,
        'isCompleted': isCompleted,
        'score':       score,
        'attemptedAt': attemptedAt?.toIso8601String(),
        'filePath':    filePath,
      };
}

class StrategyChecklist {
  final String            subject;
  final List<ChecklistItem> items;

  const StrategyChecklist({required this.subject, required this.items});

  factory StrategyChecklist.fromJson(Map<String, dynamic> json) => StrategyChecklist(
        subject: json['subject'],
        items: (json['items'] as List).map((i) => ChecklistItem.fromJson(i)).toList(),
      );

  Map<String, dynamic> toJson() => {
        'subject': subject,
        'items':   items.map((i) => i.toJson()).toList(),
      };
}

class ChecklistItem {
  final String id;
  final String text;
  final bool   isChecked;

  const ChecklistItem({required this.id, required this.text, required this.isChecked});

  ChecklistItem copyWith({bool? isChecked}) =>
      ChecklistItem(id: id, text: text, isChecked: isChecked ?? this.isChecked);

  factory ChecklistItem.fromJson(Map<String, dynamic> json) =>
      ChecklistItem(id: json['id'], text: json['text'], isChecked: json['isChecked'] ?? false);

  Map<String, dynamic> toJson() => {'id': id, 'text': text, 'isChecked': isChecked};
}

class FormulaSheet {
  final String           subject;
  final String           topic;
  final List<FormulaItem> formulas;

  const FormulaSheet({required this.subject, required this.topic, required this.formulas});

  factory FormulaSheet.fromJson(Map<String, dynamic> json) => FormulaSheet(
        subject:  json['subject'],
        topic:    json['topic'],
        formulas: (json['formulas'] as List).map((f) => FormulaItem.fromJson(f)).toList(),
      );

  Map<String, dynamic> toJson() => {
        'subject':  subject,
        'topic':    topic,
        'formulas': formulas.map((f) => f.toJson()).toList(),
      };
}

class FormulaItem {
  final String name;
  final String expression;
  final bool   isFavorite;
  final String? explanation;
  final String? topicName;

  const FormulaItem({
    required this.name, 
    required this.expression, 
    this.isFavorite = false,
    this.explanation,
    this.topicName,
  });

  String get latexExpression {
    if (expression.startsWith(r'\') || expression.contains(r'\frac')) {
      return expression;
    }
    return _convertToLatex(expression);
  }

  static String _convertToLatex(String expr) {
    String result = expr;
    result = result.replaceAllMapped(
      RegExp(r'√\(([^)]+)\)'),
      (m) => r'\sqrt{${m.group(1)}}',
    );
    result = result.replaceAllMapped(
      RegExp(r'√(\d+)'),
      (m) => r'\sqrt{${m.group(1)}}',
    );
    result = result.replaceAllMapped(
      RegExp(r'\(([^)]+)\)\s*/\s*(\d+|[a-zA-Z])(\s|$)'),
      (m) => r'\frac{${m.group(1)}}{${m.group(2)}}${m.group(3)}',
    );
    return result;
  }

  FormulaItem copyWith({bool? isFavorite}) =>
      FormulaItem(
        name: name, 
        expression: expression, 
        isFavorite: isFavorite ?? this.isFavorite,
        explanation: explanation,
        topicName: topicName,
      );

  factory FormulaItem.fromJson(Map<String, dynamic> json) => FormulaItem(
        name:       json['name'],
        expression: json['expression'],
        isFavorite: json['isFavorite'] ?? false,
        explanation: json['explanation'],
        topicName: json['topicName'],
      );

  Map<String, dynamic> toJson() => {
    'name': name, 
    'expression': expression, 
    'isFavorite': isFavorite,
    'explanation': explanation,
    'topicName': topicName,
  };
}

class RevisionPlan {
  final bool                    compressionMode;
  final List<DailyRevisionTarget> dailyTargets;
  final List<int>               completedDays;
  final DateTime?               lastUpdated;

  const RevisionPlan({
    required this.compressionMode,
    required this.dailyTargets,
    required this.completedDays,
    this.lastUpdated,
  });

  factory RevisionPlan.fromJson(Map<String, dynamic> json) {
    DateTime? lastUpdated;
    if (json['lastUpdated'] != null) {
      try { lastUpdated = DateTime.parse(json['lastUpdated']); } catch (_) {}
    }
    return RevisionPlan(
      compressionMode: json['compressionMode'] ?? true,
      dailyTargets: (json['dailyTargets'] as List?)
              ?.map((d) => DailyRevisionTarget.fromJson(d)).toList() ?? [],
      completedDays: List<int>.from(json['completedDays'] ?? []),
      lastUpdated:   lastUpdated,
    );
  }

  Map<String, dynamic> toJson() => {
        'compressionMode': compressionMode,
        'dailyTargets':    dailyTargets.map((d) => d.toJson()).toList(),
        'completedDays':   completedDays,
        'lastUpdated':     lastUpdated?.toIso8601String(),
      };
}

class DailyRevisionTarget {
  final DateTime            date;
  final int                 targetHours;
  final List<ChapterRevision> topics;
  final String              notes;

  const DailyRevisionTarget({
    required this.date,
    required this.targetHours,
    required this.topics,
    required this.notes,
  });

  factory DailyRevisionTarget.fromJson(Map<String, dynamic> json) {
    DateTime date;
    try { date = DateTime.parse(json['date']); } catch (_) { date = DateTime.now(); }
    return DailyRevisionTarget(
      date:        date,
      targetHours: json['targetHours'] ?? 5,
      topics: (json['topics'] as List?)?.map((t) => ChapterRevision.fromJson(t)).toList() ?? [],
      notes:       json['notes'] ?? '',
    );
  }

  Map<String, dynamic> toJson() => {
        'date':        date.toIso8601String(),
        'targetHours': targetHours,
        'topics':      topics.map((t) => t.toJson()).toList(),
        'notes':       notes,
      };
}

class ChapterRevision {
  final String         subject;
  final String         chapter;
  final int            priority;
  final RevisionStatus status;

  const ChapterRevision({
    required this.subject,
    required this.chapter,
    required this.priority,
    required this.status,
  });

  factory ChapterRevision.fromJson(Map<String, dynamic> json) => ChapterRevision(
        subject:  json['subject'],
        chapter:  json['chapter'],
        priority: json['priority'] ?? 1,
        status: RevisionStatus.values.firstWhere(
          (s) => s.name == json['status'],
          orElse: () => RevisionStatus.pending,
        ),
      );

  Map<String, dynamic> toJson() => {
        'subject':  subject,
        'chapter':  chapter,
        'priority': priority,
        'status':   status.name,
      };
}

enum RevisionStatus { pending, inProgress, completed, skipped }

// Provider
final examPlannerServiceProvider = ExamPlannerService();

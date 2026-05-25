// lib/services/exam_planner_repository.dart
// ─────────────────────────────────────────────────────────────────
// Reactive Repository Pattern for Exam Planner
// Parallel data fetching, reactive state, offline caching
// ─────────────────────────────────────────────────────────────────

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'exam_planner_service.dart';

class ExamPlannerState {
  final ExamConfig? config;
  final ExamCountdown? countdown;
  final List<Milestone> milestones;
  final List<PastPaperPack> pastPapers;
  final Map<String, StrategyChecklist> checklists;
  final Map<String, FormulaSheet> formulas;
  final bool isLoading;
  final String? error;
  final bool needsRetry;

  const ExamPlannerState({
    this.config,
    this.countdown,
    this.milestones = const [],
    this.pastPapers = const [],
    this.checklists = const {},
    this.formulas = const {},
    this.isLoading = true,
    this.error,
    this.needsRetry = false,
  });

  List<ChecklistItem> get allChecklistItems {
    return checklists.values.expand((c) => c.items).toList();
  }

  List<FormulaSheet> get allFormulaSheets {
    return formulas.values.toList();
  }

  ExamPlannerState copyWith({
    ExamConfig? config,
    ExamCountdown? countdown,
    List<Milestone>? milestones,
    List<PastPaperPack>? pastPapers,
    Map<String, StrategyChecklist>? checklists,
    Map<String, FormulaSheet>? formulas,
    bool? isLoading,
    String? error,
    bool? needsRetry,
  }) {
    return ExamPlannerState(
      config: config ?? this.config,
      countdown: countdown ?? this.countdown,
      milestones: milestones ?? this.milestones,
      pastPapers: pastPapers ?? this.pastPapers,
      checklists: checklists ?? this.checklists,
      formulas: formulas ?? this.formulas,
      isLoading: isLoading ?? this.isLoading,
      error: error,
      needsRetry: needsRetry ?? this.needsRetry,
    );
  }
}

class ExamPlannerRepository {
  final ExamPlannerService _service;

  ExamPlannerRepository({ExamPlannerService? service})
      : _service = service ?? ExamPlannerService();
  
  ExamPlannerService get service => _service;

  Future<ExamPlannerState> fetchAll() async {
    try {
      final configResult = _service.getExamConfig();
      final countdownResult = _service.getCountdown();
      final milestonesResult = _service.getMilestones();
      final papersResult = _service.getPastPapers();
      final checklistsResult = _service.getChecklists();
      final formulasResult = _service.getFormulaSheets();

      final results = await Future.wait([
        configResult,
        countdownResult,
        milestonesResult,
        papersResult,
        checklistsResult,
        formulasResult,
      ]);

      return ExamPlannerState(
        config: results[0] as ExamConfig,
        countdown: results[1] as ExamCountdown,
        milestones: results[2] as List<Milestone>,
        pastPapers: results[3] as List<PastPaperPack>,
        checklists: results[4] as Map<String, StrategyChecklist>,
        formulas: results[5] as Map<String, FormulaSheet>,
        isLoading: false,
      );
    } catch (e) {
      return ExamPlannerState(
        isLoading: false,
        error: e.toString(),
        needsRetry: true,
      );
    }
  }

  Future<void> saveConfig(ExamConfig config) async {
    await _service.saveExamConfig(config);
  }

  Future<void> completeMilestone(String id) async {
    await _service.completeMilestone(id);
  }

  Future<void> savePastPaperResult(String id, int score, String? filePath) async {
    await _service.savePastPaperResult(id, score, filePath);
  }

  Future<void> updateChecklistItem(String subject, String itemId, bool isChecked) async {
    await _service.updateChecklistItem(subject, itemId, isChecked);
  }

  Future<void> toggleFormulaFavorite(String topic, String formulaName) async {
    await _service.toggleFormulaFavorite(topic, formulaName);
  }

  Future<void> generateCompressionPlan({
    required List<String> subjects,
    required int daysRemaining,
    required Map<String, int> weakChapterCount,
  }) async {
    await _service.generateCompressionPlan(
      subjects: subjects,
      daysRemaining: daysRemaining,
      weakChapterCount: weakChapterCount,
    );
  }

  Future<void> markDayComplete(int dayIndex, bool isComplete) async {
    await _service.markDayComplete(dayIndex, isComplete);
  }
}

final examPlannerRepositoryProvider = Provider<ExamPlannerRepository>((ref) {
  return ExamPlannerRepository();
});

class ExamPlannerNotifier extends AsyncNotifier<ExamPlannerState> {
  ExamPlannerRepository get _repo => ref.read(examPlannerRepositoryProvider);

  @override
  Future<ExamPlannerState> build() async {
    return _repo.fetchAll();
  }

  Future<void> refresh() async {
    state = const AsyncValue.loading();
    state = AsyncValue.data(await _repo.fetchAll());
  }

  Future<void> retry() async {
    await refresh();
  }

  Future<void> saveConfig(ExamConfig config) async {
    await _repo.saveConfig(config);
    await refresh();
  }

  Future<void> completeMilestone(String id) async {
    await _repo.completeMilestone(id);
    ref.invalidateSelf();
  }

  Future<void> savePastPaperResult(String id, int score, String? filePath) async {
    await _repo.savePastPaperResult(id, score, filePath);
    ref.invalidateSelf();
  }

  Future<void> updateChecklistItem(String subject, String itemId, bool isChecked) async {
    await _repo.updateChecklistItem(subject, itemId, isChecked);
    ref.invalidateSelf();
  }

  Future<void> toggleFormulaFavorite(String topic, String formulaName) async {
    await _repo.toggleFormulaFavorite(topic, formulaName);
    ref.invalidateSelf();
  }

Future<void> generateCompressionPlan({
    required List<String> subjects,
    required DateTime examStart,
  }) async {
    final daysRemaining = examStart.difference(DateTime.now()).inDays;
    await _repo.service.generateCompressionPlan(
      subjects: subjects,
      daysRemaining: daysRemaining.clamp(1, 90),
      weakChapterCount: { for (var s in subjects) s: 3 },
    );
  }

  Future<void> markDayComplete(int dayIndex, bool isComplete) async {
    await _repo.markDayComplete(dayIndex, isComplete);
    ref.invalidateSelf();
  }

  Future<void> toggleMilestone(String id) async {
    await _repo.service.toggleMilestone(id);
    ref.invalidateSelf();
  }

  Future<void> addMilestone(Milestone milestone) async {
    await _repo.service.addMilestone(milestone);
    ref.invalidateSelf();
  }

  Future<void> deleteMilestone(String id) async {
    await _repo.service.deleteMilestone(id);
    ref.invalidateSelf();
  }

  Future<void> addChecklistItem(String subject, ChecklistItem item) async {
    await _repo.service.addChecklistItem(subject, item);
    ref.invalidateSelf();
  }

  Future<void> toggleChecklistItem(String subject, String itemId) async {
    await _repo.service.toggleChecklistItem(subject, itemId);
    ref.invalidateSelf();
  }

  Future<void> deleteChecklistItem(String subject, String itemId) async {
    await _repo.service.deleteChecklistItem(subject, itemId);
    ref.invalidateSelf();
  }

  Future<void> saveRevisionPlan(RevisionPlan plan) async {
    await _repo.service.saveRevisionPlan(plan);
    ref.invalidateSelf();
  }
}

final examPlannerProvider = AsyncNotifierProvider<ExamPlannerNotifier, ExamPlannerState>(
  ExamPlannerNotifier.new,
);

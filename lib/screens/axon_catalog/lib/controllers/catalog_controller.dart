import 'package:flutter/material.dart';
import '../models/paper.dart';
import '../models/paper_series.dart';
import '../data/series_data.dart';

// ─── Never-firing Listenable placeholder ──────────────────────────────────

class _StaticListenable extends Listenable {
  const _StaticListenable();
  @override
  void addListener(VoidCallback listener) {}
  @override
  void removeListener(VoidCallback listener) {}
}

// ─── CatalogController ─────────────────────────────────────────────────────

class CatalogController extends ChangeNotifier {
  // ── Data ──────────────────────────────────────────────────────────────────
  final List<PaperSeries> allSeries = buildAllSeries();

  // ── Animation controllers (null until initialize) ─────────────────────────
  AnimationController? _startupCtrl;
  AnimationController? _breathCtrl;
  AnimationController? _snapCtrl;

  Animation<double>? _opacityAnim;
  Animation<double>? _zoomAnim;
  Animation<double>? _breathAnim;

  // ── View state ────────────────────────────────────────────────────────────
  double _viewOffset;
  double _snapStart = 0;
  double _snapTarget = 0;

  // ── Interaction state ─────────────────────────────────────────────────────
  int? openFolderIndex;
  Offset? openFolderScreenPos;
  Paper? selectedPaper;
  SubjectGroup? filterSubject;

  bool _initialized = false;

  // ── Constructor ───────────────────────────────────────────────────────────
  CatalogController() : _viewOffset = 7.5; // center of 16 series

  // ── Getters ───────────────────────────────────────────────────────────────
  bool get initialized => _initialized;
  bool get startupComplete => _startupCtrl?.isCompleted ?? false;
  double get viewOffset => _viewOffset;
  double get zoomValue => _zoomAnim?.value ?? 0.21;
  double get opacityValue => _opacityAnim?.value ?? 0.0;
  double get breathValue => _breathAnim?.value ?? 1.0;

  Listenable get animationListenable {
    if (!_initialized) return const _StaticListenable();
    return Listenable.merge([_startupCtrl!, _breathCtrl!]);
  }

  PaperSeries? get openSeries =>
      openFolderIndex != null ? allSeries[openFolderIndex!] : null;

  List<Paper> get displayedPapers {
    final s = openSeries;
    if (s == null) return [];
    if (filterSubject == null) return s.papers;
    return s.papers.where((p) => p.subjectGroup == filterSubject).toList();
  }

  // ── Initialize ────────────────────────────────────────────────────────────
  void initialize(TickerProvider vsync) {
    if (_initialized) return;

    // Startup: fade in + zoom in
    _startupCtrl = AnimationController(
      vsync: vsync,
      duration: const Duration(milliseconds: 3200),
    );
    _opacityAnim = CurvedAnimation(
      parent: _startupCtrl!,
      curve: const Interval(0.0, 0.32, curve: Curves.easeIn),
    );
    _zoomAnim = Tween<double>(begin: 0.21, end: 1.0).animate(
      CurvedAnimation(
        parent: _startupCtrl!,
        curve: const Interval(0.12, 0.88, curve: Curves.easeInOutCubic),
      ),
    );
    _startupCtrl!.addListener(notifyListeners);

    // Breath: gentle idle pulse on center folder
    _breathCtrl = AnimationController(
      vsync: vsync,
      duration: const Duration(milliseconds: 2400),
    );
    _breathAnim = Tween<double>(begin: 1.0, end: 1.018).animate(
      CurvedAnimation(parent: _breathCtrl!, curve: Curves.easeInOut),
    );
    _breathCtrl!.repeat(reverse: true);
    _breathCtrl!.addListener(notifyListeners);

    // Snap: smooth deceleration after pan release
    _snapCtrl = AnimationController(
      vsync: vsync,
      duration: const Duration(milliseconds: 420),
    );
    _snapCtrl!.addListener(_onSnapTick);

    _initialized = true;
    notifyListeners();

    // Small delay before starting — let the screen render first
    Future.delayed(const Duration(milliseconds: 350), () {
      if (_startupCtrl != null) _startupCtrl!.forward();
    });
  }

  // ── Snap tick ─────────────────────────────────────────────────────────────
  void _onSnapTick() {
    final t = Curves.easeOutCubic.transform(_snapCtrl!.value);
    _viewOffset = _snapStart + (_snapTarget - _snapStart) * t;
    notifyListeners();
  }

  // ── Pan ───────────────────────────────────────────────────────────────────
  void panBy(double dx) {
    _snapCtrl?.stop();
    final maxOffset = (allSeries.length - 1).toDouble();
    _viewOffset = (_viewOffset - dx / 215.0).clamp(0.0, maxOffset);
    notifyListeners();
  }

  void snapToNearest() {
    _snapStart = _viewOffset;
    _snapTarget = _viewOffset
        .roundToDouble()
        .clamp(0.0, (allSeries.length - 1).toDouble());
    _snapCtrl?.reset();
    _snapCtrl?.forward();
  }

  // ── Folder open / close ───────────────────────────────────────────────────
  void openFolder(int index, Offset screenCenter) {
    if (openFolderIndex == index) {
      closeFolder();
      return;
    }
    openFolderIndex = index;
    openFolderScreenPos = screenCenter;
    selectedPaper = null;
    filterSubject = null;
    notifyListeners();
  }

  void closeFolder() {
    openFolderIndex = null;
    openFolderScreenPos = null;
    selectedPaper = null;
    notifyListeners();
  }

  // ── Paper detail ──────────────────────────────────────────────────────────
  void selectPaper(Paper paper) {
    selectedPaper = paper;
    notifyListeners();
  }

  void deselectPaper() {
    selectedPaper = null;
    notifyListeners();
  }

  // ── Subject filter ────────────────────────────────────────────────────────
  void setFilter(SubjectGroup? group) {
    filterSubject = group;
    notifyListeners();
  }

  void reset() {
    _startupCtrl?.dispose();
    _startupCtrl = null;
    _breathCtrl?.dispose();
    _breathCtrl = null;
    _snapCtrl?.dispose();
    _snapCtrl = null;
    _initialized = false;
  }

  // ── Dispose ───────────────────────────────────────────────────────────────
  @override
  void dispose() {
    reset();
    super.dispose();
  }
}

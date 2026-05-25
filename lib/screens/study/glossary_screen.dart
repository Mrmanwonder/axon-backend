import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class GlossaryScreen extends StatefulWidget {
  const GlossaryScreen({super.key});

  @override
  State<GlossaryScreen> createState() => _GlossaryScreenState();
}

class _GlossaryScreenState extends State<GlossaryScreen>
    with TickerProviderStateMixin {
  late final TransformationController _controller;
  late final AnimationController _introController;
  late final TextEditingController _searchController;
  late final FocusNode _searchFocus;
  late final TextLayoutCache _textCache;

  SpatialGlossaryEngine? _engine;
  AnimationController? _cameraController;
  Timer? _searchDebounce;

  bool _loading = true;
  bool _introDone = false;
  bool _usingAsset = false;
  String _query = '';
  int? _selectedId;
  List<int> _matches = const [];
  Size _viewport = Size.zero;

  @override
  void initState() {
    super.initState();
    _controller = TransformationController();
    _introController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 2200));
    _searchController = TextEditingController();
    _searchFocus = FocusNode();
    _textCache = TextLayoutCache(maxEntries: 7000);
    _load();
    Future<void>.delayed(const Duration(milliseconds: 1400), () async {
      if (!mounted) return;
      _introController.forward(from: 0);
      await Future<void>.delayed(const Duration(milliseconds: 900));
      if (!mounted) return;
      setState(() => _introDone = true);
      _controller.value = Matrix4.identity();
    });
  }

  Future<void> _load() async {
    final result = await SpatialGlossaryEngine.loadOrBuild(
      'assets/glossary.json',
      fallbackCount: 10000,
    );
    if (!mounted) return;
    setState(() {
      _engine = result.engine;
      _usingAsset = result.usedAsset;
      _loading = false;
    });
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _searchDebounce?.cancel();
    _introController.dispose();
    _searchController.dispose();
    _searchFocus.dispose();
    _controller.dispose();
    _textCache.dispose();
    super.dispose();
  }

  void _onSearch(String value) {
    _searchDebounce?.cancel();
    _searchDebounce = Timer(const Duration(milliseconds: 400), () {
      final engine = _engine;
      if (engine == null) return;
      final trimmed = value.trim();
      setState(() {
        _query = trimmed;
        _matches = engine.search(_query);
      });
      if (_matches.isNotEmpty) {
        _centerWord(_matches.first, scale: 2.5);
      }
    });
  }

  Future<void> _centerWord(int id, {double scale = 2.0}) async {
    final engine = _engine;
    if (engine == null || _viewport == Size.zero) return;
    final word = engine.wordsById[id]!;
    final center = Offset(_viewport.width / 2, _viewport.height / 2);
    final target = Matrix4.identity()
      ..translateByDouble(center.dx, center.dy, 0.0, 1.0)
      ..scaleByDouble(scale, scale, scale, 1.0)
      ..translateByDouble(-word.position.dx, -word.position.dy, 0.0, 1.0);
    setState(() {
      _selectedId = id;
    });
    await _animateCamera(target);
  }

  Future<void> _animateCamera(Matrix4 target) async {
    _cameraController?.dispose();
    final c = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 540),
    );
    final anim = Matrix4Tween(
      begin: _controller.value.clone(),
      end: target,
    ).animate(CurvedAnimation(parent: c, curve: Curves.easeOutCubic));
    _cameraController = c;
    void tick() => _controller.value = anim.value;
    anim.addListener(tick);
    await c.forward(from: 0);
    anim.removeListener(tick);
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        _viewport = Size(constraints.maxWidth, constraints.maxHeight);
        final engine = _engine;
        return Scaffold(
          body: Stack(
            children: [
              Positioned.fill(
                child: DecoratedBox(
                  decoration: const BoxDecoration(
                    gradient: RadialGradient(
                      center: Alignment.center,
                      radius: 1.35,
                      colors: [Color(0xFF0F172A), Color(0xFF05070C)],
                    ),
                  ),
                  child: _loading || engine == null
                      ? const Center(child: CircularProgressIndicator())
                      : InteractiveViewer.builder(
                          transformationController: _controller,
                          boundaryMargin: const EdgeInsets.all(double.infinity),
                          minScale: 0.06,
                          maxScale: 8,
                          builder: (context, viewport) {
                            final camera = CameraState.fromMatrix(
                                _controller.value, _viewport);
                            final visible = engine.visibleWords(camera);
                            final results =
                                _matches.isEmpty ? null : _matches.toSet();
                            return SizedBox(
                              width: _viewport.width,
                              height: _viewport.height,
                              child: GestureDetector(
                                behavior: HitTestBehavior.opaque,
                                onTapUp: (details) {
                                  final hit = engine.hitTest(
                                      camera
                                          .screenToWorld(details.localPosition),
                                      camera,
                                      visible);
                                  if (hit != null) {
                                    _centerWord(hit.id,
                                        scale: math.max(camera.scale, 2.0));
                                  } else {
                                    setState(() {
                                      _selectedId = null;
                                    });
                                  }
                                },
                                child: AnimatedBuilder(
                                  animation: _introController,
                                  builder: (context, _) {
                                    return CustomPaint(
                                      painter: GlossaryFieldPainter(
                                        camera: camera,
                                        words: visible,
                                        selectedId: _selectedId,
                                        resultIds: results,
                                        query: _query,
                                        textCache: _textCache,
                                        introProgress: Curves.easeOutCubic
                                            .transform(_introController.value),
                                        introDone: _introDone,
                                      ),
                                    );
                                  },
                                ),
                              ),
                            );
                          },
                        ),
                ),
              ),
              if (!_loading && engine != null) ...[
                if (!_introDone)
                  Positioned.fill(
                    child: AnimatedBuilder(
                      animation: _introController,
                      builder: (context, _) {
                        return IgnorePointer(
                          ignoring: true,
                          child: Center(
                            child: Opacity(
                              opacity: (1 -
                                      Curves.easeOut
                                          .transform(_introController.value))
                                  .clamp(0.0, 1.0),
                              child: Transform.scale(
                                scale: 1.0 +
                                    0.06 *
                                        Curves.easeOut
                                            .transform(_introController.value),
                                child: ShaderMask(
                                  shaderCallback: (bounds) =>
                                      const LinearGradient(
                                    colors: [
                                      Color(0xFFFFFFFF),
                                      Color(0xFF8BE8F5)
                                    ],
                                  ).createShader(bounds),
                                  child: const Text(
                                    'GLOSSARY',
                                    style: TextStyle(
                                      fontSize: 54,
                                      letterSpacing: 8,
                                      fontWeight: FontWeight.w900,
                                      color: Colors.white,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                Positioned(
                  left: 18,
                  right: 18,
                  top: 18 + MediaQuery.of(context).padding.top,
                  child: _SearchBar(
                    query: _query,
                    usingAsset: _usingAsset,
                    controller: _searchController,
                    focusNode: _searchFocus,
                    onChanged: _onSearch,
                    onClear: () {
                      _searchDebounce?.cancel();
                      _searchController.clear();
                      setState(() {
                        _query = '';
                        _matches = const [];
                      });
                    },
                    searchCount: _matches.length,
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }
}

class _SearchBar extends StatelessWidget {
  const _SearchBar({
    required this.query,
    required this.usingAsset,
    required this.controller,
    required this.focusNode,
    required this.onChanged,
    required this.onClear,
    required this.searchCount,
  });
  final String query;
  final bool usingAsset;
  final TextEditingController controller;
  final FocusNode focusNode;
  final ValueChanged<String> onChanged;
  final VoidCallback onClear;
  final int searchCount;

  @override
  Widget build(BuildContext context) => ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: const Color(0xCC0E1626),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          ),
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                const Icon(Icons.search_rounded,
                    color: Colors.white70, size: 22),
                const SizedBox(width: 10),
                Expanded(
                  child: TextField(
                    controller: controller,
                    focusNode: focusNode,
                    onChanged: onChanged,
                    style: const TextStyle(color: Colors.white, fontSize: 14),
                    decoration: const InputDecoration(
                      hintText: 'Search glossary',
                      hintStyle: TextStyle(color: Colors.white38, fontSize: 14),
                      border: InputBorder.none,
                      isDense: true,
                      contentPadding: EdgeInsets.zero,
                    ),
                  ),
                ),
                if (query.isNotEmpty)
                  IconButton(
                    onPressed: onClear,
                    icon: const Icon(Icons.close_rounded,
                        color: Colors.white54, size: 18),
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                  ),
                if (query.isNotEmpty) const SizedBox(width: 10),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                  decoration: BoxDecoration(
                    color: usingAsset
                        ? const Color(0xFF143420)
                        : const Color(0xFF312012),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Text(
                    '$searchCount',
                    style: const TextStyle(
                        fontSize: 11, fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
}

class CameraState {
  CameraState({
    required this.scale,
    required this.translation,
    required this.viewport,
  }) : worldRect = Rect.fromPoints(
          Offset(-translation.dx / scale, -translation.dy / scale),
          Offset(
            (viewport.width - translation.dx) / scale,
            (viewport.height - translation.dy) / scale,
          ),
        );

  final double scale;
  final Offset translation;
  final Size viewport;
  final Rect worldRect;

  factory CameraState.fromMatrix(Matrix4 matrix, Size viewport) => CameraState(
        scale: matrix.getMaxScaleOnAxis(),
        translation: Offset(matrix.storage[12], matrix.storage[13]),
        viewport: viewport,
      );

  Offset worldToScreen(Offset world) => Offset(
      world.dx * scale + translation.dx, world.dy * scale + translation.dy);

  Offset screenToWorld(Offset screen) => Offset(
      (screen.dx - translation.dx) / scale,
      (screen.dy - translation.dy) / scale);
}

class GlossaryWord {
  GlossaryWord({
    required this.id,
    required this.text,
    required this.meaning,
    required this.bucket,
    required this.position,
    required this.z,
  });

  final int id;
  final String text;
  final String meaning;
  final String bucket;
  final Offset position;
  final double z;

  factory GlossaryWord.fromJson(int id, Map<String, dynamic> json) =>
      GlossaryWord(
        id: id,
        text: '${json['text'] ?? 'word-$id'}',
        meaning: '${json['meaning'] ?? ''}',
        bucket: '${json['bucket'] ?? 'general'}',
        position: Offset(
          ((json['x'] as num?)?.toDouble() ?? 0),
          ((json['y'] as num?)?.toDouble() ?? 0),
        ),
        z: ((json['z'] as num?)?.toDouble() ?? 0),
      );
}

class EngineResult {
  EngineResult({required this.engine, required this.usedAsset});
  final SpatialGlossaryEngine engine;
  final bool usedAsset;
}

class SpatialGlossaryEngine {
  SpatialGlossaryEngine._(this.words, this.chunkSize)
      : wordsById = {for (final w in words) w.id: w} {
    for (final w in words) {
      final key = _key(w.position);
      (_chunks[key] ??= []).add(w);
      final t = w.text.toLowerCase();
      for (var i = 1; i <= math.min(t.length, 12); i++) {
        (_prefix[t.substring(0, i)] ??= []).add(w.id);
      }
      (_token[w.bucket.toLowerCase()] ??= []).add(w.id);
      for (final tok in t.split(RegExp(r'[^a-z0-9]+'))) {
        if (tok.isNotEmpty) (_token[tok] ??= []).add(w.id);
      }
    }
  }

  final List<GlossaryWord> words;
  final double chunkSize;
  final Map<int, GlossaryWord> wordsById;
  final Map<String, List<GlossaryWord>> _chunks = {};
  final Map<String, List<int>> _prefix = {};
  final Map<String, List<int>> _token = {};

  int get totalCount => words.length;

  static Future<EngineResult> loadOrBuild(String path,
      {required int fallbackCount}) async {
    try {
      final raw = await rootBundle.loadString(path);
      final decoded = jsonDecode(raw);
      if (decoded is List) {
        final words = <GlossaryWord>[];
        for (var i = 0; i < decoded.length; i++) {
          final item = decoded[i];
          if (item is Map<String, dynamic>) {
            words.add(GlossaryWord.fromJson(i, item));
          } else if (item is Map) {
            words.add(GlossaryWord.fromJson(i, item.cast<String, dynamic>()));
          }
        }
        if (words.isNotEmpty) {
          return EngineResult(
              engine: SpatialGlossaryEngine._(words, 480), usedAsset: true);
        }
      }
    } catch (_) {}
    return EngineResult(
      engine: SpatialGlossaryEngine.generate(fallbackCount),
      usedAsset: false,
    );
  }

  static SpatialGlossaryEngine generate(int count) {
    final r = math.Random(42);
    const buckets = [
      'logic',
      'systems',
      'data',
      'security',
      'networks',
      'programming',
      'theory',
      'hardware'
    ];
    final words = <GlossaryWord>[];
    for (var i = 0; i < count; i++) {
      final arm = i % 8;
      final radius = math.sqrt(i + 1) * 72 + r.nextDouble() * 100;
      final angle = arm * (math.pi / 4) + i * 0.058;
      final x = math.cos(angle) * radius + r.nextDouble() * 140 - 70;
      final y = math.sin(angle) * radius + r.nextDouble() * 140 - 70;
      final b = buckets[i % buckets.length];
      final text = '${b.substring(0, 2)}-${i.toString().padLeft(4, '0')}';
      words.add(GlossaryWord(
        id: i,
        text: text,
        meaning: 'Glossary concept for $text in the $b cluster.',
        bucket: b,
        position: Offset(x, y),
        z: (i % 24) / 6,
      ));
    }
    return SpatialGlossaryEngine._(words, 480);
  }

  String _key(Offset p) =>
      '${(p.dx / chunkSize).floor()}:${(p.dy / chunkSize).floor()}';

  List<int> search(String q) {
    q = q.toLowerCase().trim();
    if (q.isEmpty) return const [];
    final p = _prefix[q];
    if (p != null && p.isNotEmpty) return p.take(200).toList();
    final out = <int, int>{};
    for (final tok
        in q.split(RegExp(r'[^a-z0-9]+')).where((e) => e.isNotEmpty)) {
      for (final id in _token[tok] ?? const []) {
        out[id] = (out[id] ?? 0) + 4;
      }
      for (final w in words) {
        final t = w.text.toLowerCase();
        if (t.contains(tok) || w.bucket.contains(tok)) {
          out[w.id] = (out[w.id] ?? 0) + 1;
        }
      }
    }
    final ids = out.keys.toList()..sort((a, b) => (out[b]!.compareTo(out[a]!)));
    return ids.take(200).toList();
  }

  List<GlossaryWord> visibleWords(CameraState camera) {
    final margin = 260 / camera.scale;
    final rect = camera.worldRect.inflate(margin);
    final minX = (rect.left / chunkSize).floor();
    final maxX = (rect.right / chunkSize).floor();
    final minY = (rect.top / chunkSize).floor();
    final maxY = (rect.bottom / chunkSize).floor();
    final res = <GlossaryWord>[];
    for (var x = minX; x <= maxX; x++) {
      for (var y = minY; y <= maxY; y++) {
        final c = _chunks['$x:$y'];
        if (c != null) res.addAll(c);
      }
    }
    res.sort((a, b) => a.z.compareTo(b.z));
    return res;
  }

  GlossaryWord? hitTest(
      Offset p, CameraState camera, List<GlossaryWord> visible) {
    GlossaryWord? best;
    var bestD = double.infinity;
    final th = 24 / camera.scale;
    for (final w in visible) {
      final d = (w.position - p).distance;
      if (d < th && d < bestD) {
        best = w;
        bestD = d;
      }
    }
    return best;
  }
}

class TextLayoutCache {
  TextLayoutCache({required this.maxEntries});
  final int maxEntries;
  final Map<String, TextPainter> _cache = {};
  final List<String> _order = [];

  int get entryCount => _cache.length;

  TextPainter getPainter({
    required String text,
    required double fontSize,
    required Color color,
    required FontWeight fontWeight,
    double? maxWidth,
    int maxLines = 1,
    String? ellipsis,
  }) {
    final key =
        '$text|${fontSize.toStringAsFixed(2)}|${color.toARGB32()}|${fontWeight.value}|${maxWidth ?? -1}|$maxLines|${ellipsis ?? ''}';
    final existing = _cache[key];
    if (existing != null) return existing;
    final p = TextPainter(
      text: TextSpan(
        text: text,
        style:
            TextStyle(color: color, fontSize: fontSize, fontWeight: fontWeight),
      ),
      textDirection: TextDirection.ltr,
      maxLines: maxLines,
      ellipsis: ellipsis,
    )..layout(maxWidth: maxWidth ?? double.infinity);
    _cache[key] = p;
    _order.add(key);
    if (_order.length > maxEntries) _cache.remove(_order.removeAt(0));
    return p;
  }

  void dispose() {
    _cache.clear();
    _order.clear();
  }
}

class GlossaryFieldPainter extends CustomPainter {
  GlossaryFieldPainter({
    required this.camera,
    required this.words,
    required this.selectedId,
    required this.resultIds,
    required this.query,
    required this.textCache,
    required this.introProgress,
    required this.introDone,
  });

  final CameraState camera;
  final List<GlossaryWord> words;
  final int? selectedId;
  final Set<int>? resultIds;
  final String query;
  final TextLayoutCache textCache;
  final double introProgress;
  final bool introDone;

  @override
  void paint(Canvas canvas, Size size) {
    final bg = Paint()
      ..color = const Color(0xFF9FB3D1).withValues(alpha: 0.05)
      ..strokeWidth = 1;
    _grid(canvas, size, bg);
    final hasQuery = query.isNotEmpty;
    final showLabels = camera.scale > 0.22;
    final dot = Paint()..style = PaintingStyle.fill;
    final intro = Curves.easeOutCubic.transform(introProgress);

    for (final w in words) {
      final screen = camera.worldToScreen(w.position);
      if (screen.dx < -180 ||
          screen.dx > size.width + 180 ||
          screen.dy < -120 ||
          screen.dy > size.height + 120) {
        continue;
      }
      final dist = (w.position - Offset.zero).distance;
      final radial = (1.0 - ((1 - intro) * (dist / 1200))).clamp(0.0, 1.0);
      final isSel = selectedId == w.id;
      final isMatch = resultIds?.contains(w.id) ?? !hasQuery;
      final depth = _depthFactor(camera.scale, w.z);
      final emphasis = isSel
          ? 1.0
          : selectedId != null
              ? 0.3
              : (isMatch ? 0.9 : 0.14);
      final a = (depth * emphasis * radial).clamp(0.0, 1.0);
      final color = Colors.white.withValues(alpha: a);

      if (!showLabels) {
        dot.color = color;
        canvas.drawCircle(screen, isSel ? 4.0 : 1.2 + depth * 2.1, dot);
        continue;
      }

      final fontSize = ((12 + depth * 18) * math.sqrt(camera.scale))
          .clamp(10.0, isSel ? 44.0 : 30.0);
      final p = textCache.getPainter(
        text: w.text,
        fontSize: fontSize,
        color: color,
        fontWeight: isSel ? FontWeight.w800 : FontWeight.w600,
      );
      final offset = screen - Offset(p.width / 2, p.height / 2);

      if (!isMatch && hasQuery && selectedId == null) {
        final blur = Paint()
          ..color = Colors.transparent
          ..imageFilter = ui.ImageFilter.blur(sigmaX: 2, sigmaY: 2);
        canvas.saveLayer(
            Rect.fromLTWH(
                offset.dx - 6, offset.dy - 6, p.width + 12, p.height + 12),
            blur);
        p.paint(canvas, offset);
        canvas.restore();
      } else {
        p.paint(canvas, offset);
      }

      if (isSel) {
        final mp = textCache.getPainter(
          text: w.meaning,
          fontSize: 14,
          color: Colors.white70,
          fontWeight: FontWeight.w400,
          maxWidth: 320,
          maxLines: 3,
          ellipsis: '…',
        );
        mp.paint(canvas, offset + Offset(0, p.height + 8));
      }
    }

    if (!introDone) {
      final alpha = (1 - intro).clamp(0.0, 1.0);
      final tp = TextPainter(
        text: const TextSpan(
          text: 'GLOSSARY',
          style: TextStyle(
              color: Colors.white,
              fontSize: 56,
              letterSpacing: 8,
              fontWeight: FontWeight.w900),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final off = Offset(
          size.width / 2 - tp.width / 2, size.height / 2 - tp.height / 2);
      tp.paint(canvas, off.translate(0, 12 * (1 - intro)));
      final fade = Paint()..color = Colors.black.withValues(alpha: 1 - alpha);
      canvas.drawRect(Offset.zero & size, fade);
    }
  }

  void _grid(Canvas canvas, Size size, Paint paint) {
    final spacing = camera.scale < 0.2
        ? 1200
        : camera.scale < 0.5
            ? 600
            : camera.scale < 1.2
                ? 300
                : 120;
    final tl = camera.screenToWorld(Offset.zero);
    final br = camera.screenToWorld(Offset(size.width, size.height));
    final sx = ((tl.dx / spacing).floor() * spacing).toDouble();
    final ex = ((br.dx / spacing).ceil() * spacing).toDouble();
    final sy = ((tl.dy / spacing).floor() * spacing).toDouble();
    final ey = ((br.dy / spacing).ceil() * spacing).toDouble();
    for (double x = sx; x <= ex; x += spacing) {
      canvas.drawLine(
        Offset(camera.worldToScreen(Offset(x, 0)).dx, 0.0),
        Offset(camera.worldToScreen(Offset(x, 0)).dx, size.height),
        paint,
      );
    }
    for (double y = sy; y <= ey; y += spacing) {
      canvas.drawLine(
        Offset(0.0, camera.worldToScreen(Offset(0, y)).dy),
        Offset(size.width, camera.worldToScreen(Offset(0, y)).dy),
        paint,
      );
    }
  }

  double _depthFactor(double zoom, double z) {
    final d = ((zoom * 0.9) - z).abs();
    return (1.18 - d * 0.42).clamp(0.0, 1.0);
  }

  @override
  bool shouldRepaint(covariant GlossaryFieldPainter old) =>
      old.camera.scale != camera.scale ||
      old.camera.translation != camera.translation ||
      old.selectedId != selectedId ||
      old.resultIds != resultIds ||
      old.query != query ||
      old.textCache != textCache ||
      old.introProgress != introProgress ||
      old.introDone != introDone;
}

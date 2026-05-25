import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const FluidTimerApp());
}

class FluidTimerApp extends StatelessWidget {
  const FluidTimerApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fluid Timer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.black,
      ),
      home: const TimerScreen(),
    );
  }
}

enum TimerMode { timer, stopwatch, pomodoro }

class TimerScreen extends StatefulWidget {
  const TimerScreen({Key? key}) : super(key: key);

  @override
  State<TimerScreen> createState() => _TimerScreenState();
}

class _TimerScreenState extends State<TimerScreen> {
  TimerMode _mode = TimerMode.timer;
  bool _isRunning = false;
  bool _isEnded = false;
  bool _zenMode = false;
  bool _isDockVisible = true;
  bool _isNightMode = false;

  // Active digits values (hT: hours tens, hO: hours ones, mT: mins tens, mO: mins ones, sT: secs tens, sO: secs ones)
  int _hT = 0, _hO = 1, _mT = 0, _mO = 0, _sT = 0, _sO = 0;
  
  // Pomodoro settings
  final List<int> _pomoPresets = [25, 5, 15];
  int _pomoIdx = 0;

  double _remaining = 3600;
  double _elapsed = 0;
  double _total = 3600;

  Timer? _ticker;
  Timer? _dockHideTimer;
  int? _lastSecond;

  // Drag tracking state variables
  double _dragStartY = 0;
  int _dragStartVal = 0;
  String? _dragKey;
  int _dragMax = 9;

  @override
  void initState() {
    super.initState();
    _resetInactivityTimer();
  }

  @override
  void dispose() {
    _ticker?.cancel();
    _dockHideTimer?.cancel();
    super.dispose();
  }

  void _resetInactivityTimer() {
    if (_isEnded) return;
    setState(() {
      _isDockVisible = true;
    });
    _dockHideTimer?.cancel();
    if (_isRunning) {
      _dockHideTimer = Timer(const Duration(seconds: 3), () {
        setState(() {
          _isDockVisible = false;
        });
      });
    }
  }

  void _syncTotal(TimerMode currentMode) {
    double secs = 1;
    if (currentMode == TimerMode.timer) {
      secs = ((_hT * 10 + _hO) * 3600 + (_mT * 10 + _mO) * 60 + (_sT * 10 + _sO)).toDouble();
    } else if (currentMode == TimerMode.pomodoro) {
      secs = (_pomoPresets[_pomoIdx] * 60).toDouble();
    }
    setState(() {
      _total = secs > 0 ? secs : 1;
      _remaining = _total;
    });
  }

  void _startTimer() {
    _ticker?.cancel();
    _ticker = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      setState(() {
        if (_mode == TimerMode.stopwatch) {
          _elapsed += 0.1;
        } else {
          _remaining = (_remaining - 0.1).clamp(0, _total);
          
          final currentCeil = _remaining.ceil();
          if (currentCeil != _lastSecond) {
            _lastSecond = currentCeil;
            if (currentCeil > 0 && currentCeil <= 3) {
              _playCountdownTick();
            } else if (currentCeil == 0) {
              _isRunning = false;
              _isEnded = true;
              _zenMode = false;
              _isDockVisible = true;
              _ticker?.cancel();
              _playCompletionSound();
            }
          }
        }
      });
    });
  }

  void _playCountdownTick() {
    // 3, 2, 1 Countdown Haptics
    HapticFeedback.mediumImpact();
  }

  void _playCompletionSound() {
    // Continuous majestic vibration pattern to accompany final chord sequence
    HapticFeedback.vibrate();
    Future.delayed(const Duration(milliseconds: 150), () => HapticFeedback.vibrate());
    Future.delayed(const Duration(milliseconds: 400), () => HapticFeedback.mediumImpact());
    Future.delayed(const Duration(milliseconds: 700), () => HapticFeedback.vibrate());
  }

  void _reset() {
    _ticker?.cancel();
    setState(() {
      _isRunning = false;
      _isEnded = false;
      _lastSecond = null;
      if (_mode == TimerMode.stopwatch) {
        _elapsed = 0;
      } else {
        _remaining = _total;
        if (_mode == TimerMode.timer) {
          int h = (_total / 3600).floor();
          int m = ((_total % 3600) / 60).floor();
          int s = (_total % 60).floor();
          _hT = h ~/ 10; _hO = h % 10;
          _mT = m ~/ 10; _mO = m % 10;
          _sT = s ~/ 10; _sO = s % 10;
        }
      }
    });
    HapticFeedback.lightImpact();
  }

  void _handlePlayPause() {
    setState(() {
      _isRunning = !_isRunning;
      if (_isRunning) {
        _startTimer();
      } else {
        _ticker?.cancel();
      }
    });
    HapticFeedback.lightImpact();
    _resetInactivityTimer();
  }

  void _handleModeSwitch(TimerMode newMode) {
    if (_isRunning) return;
    setState(() {
      _mode = newMode;
      _elapsed = 0;
      _lastSecond = null;
      _syncTotal(newMode);
    });
    HapticFeedback.mediumImpact();
  }

  // Exact replication of React's quadrant drag detection
  void _onDragStart(DragStartDetails details, BoxConstraints constraints) {
    if (_isRunning || _isEnded) return;
    final localX = details.localPosition.dx;
    final localY = details.localPosition.dy;
    final halfWidth = constraints.maxWidth / 2;
    final halfHeight = constraints.maxHeight / 2;

    _dragStartY = localY;

    if (_mode == TimerMode.pomodoro) {
      _dragKey = "pomo";
      _dragStartVal = _pomoIdx;
      _dragMax = _pomoPresets.length - 1;
    } else if (_mode == TimerMode.timer) {
      bool isShowingHours = _remaining >= 3600;
      if (localY < halfHeight) {
        // Upper row digits
        if (localX < halfWidth) {
          _dragKey = isShowingHours ? "hT" : "mT";
          _dragStartVal = isShowingHours ? _hT : _mT;
          _dragMax = 5;
        } else {
          _dragKey = isShowingHours ? "hO" : "mO";
          _dragStartVal = isShowingHours ? _hO : _mO;
          _dragMax = 9;
        }
      } else {
        // Lower row digits
        if (localX < halfWidth) {
          _dragKey = isShowingHours ? "mT" : "sT";
          _dragStartVal = isShowingHours ? _mT : _sT;
          _dragMax = 5;
        } else {
          _dragKey = isShowingHours ? "mO" : "sO";
          _dragStartVal = isShowingHours ? _mO : _sO;
          _dragMax = 9;
        }
      }
    }
  }

  void _onDragUpdate(DragUpdateDetails details) {
    if (_dragKey == null || _isRunning || _isEnded) return;

    final currentY = details.localPosition.dy;
    final deltaY = _dragStartY - currentY;
    final steps = (deltaY / 40.0).round();

    if (steps != 0) {
      setState(() {
        if (_dragKey == "pomo") {
          int nextIdx = (_dragStartVal + steps) % _pomoPresets.length;
          if (nextIdx < 0) nextIdx += _pomoPresets.length;
          _pomoIdx = nextIdx;
          _syncTotal(_mode);
        } else {
          int nextValue = (_dragStartVal + steps) % (_dragMax + 1);
          if (nextValue < 0) nextValue += (_dragMax + 1);

          Map<String, int> tempDigits = {
            "hT": _hT, "hO": _hO, "mT": _mT, "mO": _mO, "sT": _sT, "sO": _sO
          };
          tempDigits[_dragKey!] = nextValue;

          int h = tempDigits["hT"]! * 10 + tempDigits["hO"]!;
          int m = tempDigits["mT"]! * 10 + tempDigits["mO"]!;
          int s = tempDigits["sT"]! * 10 + tempDigits["sO"]!;

          if (h < 24 || (h == 24 && m == 0 && s == 0)) {
            _hT = tempDigits["hT"]!;
            _hO = tempDigits["hO"]!;
            _mT = tempDigits["mT"]!;
            _mO = tempDigits["mO"]!;
            _sT = tempDigits["sT"]!;
            _sO = tempDigits["sO"]!;
            _syncTotal(_mode);
          }
        }
      });
      HapticFeedback.selectionClick();
    }
  }

  void _onDragEnd(DragEndDetails details) {
    _dragKey = null;
  }

  List<String> _getDisplayDigits() {
    int val = _mode == TimerMode.stopwatch ? _elapsed.floor() : _remaining.ceil();
    bool showingHours = val >= 3600 && _mode != TimerMode.pomodoro;

    if (showingHours) {
      int h = val ~/ 3600;
      int m = (val % 3600) ~/ 60;
      return [
        (h ~/ 10).toString(),
        (h % 10).toString(),
        (m ~/ 10).toString(),
        (m % 10).toString()
      ];
    } else {
      int m = val ~/ 60;
      int s = val % 60;
      return [
        (m ~/ 10).toString(),
        (m % 10).toString(),
        (s ~/ 10).toString(),
        (s % 10).toString()
      ];
    }
  }

  @override
  Widget build(BuildContext context) {
    final displayDigits = _getDisplayDigits();
    final double progress = _mode == TimerMode.stopwatch 
        ? (_elapsed % 60) / 60 
        : (_total > 0 ? _remaining / _total : 0);

    final double bottomPadding = (_isDockVisible && !_zenMode) ? 104.0 : 0.0;

    return Scaffold(
      body: LayoutBuilder(
        builder: (context, constraints) {
          return GestureDetector(
            onDoubleTap: () {
              setState(() {
                _isNightMode = !_isNightMode;
              });
            },
            onVerticalDragStart: (details) => _onDragStart(details, constraints),
            onVerticalDragUpdate: _onDragUpdate,
            onVerticalDragEnd: _onDragEnd,
            behavior: HitTestBehavior.opaque,
            child: Stack(
              children: [
                // 1. Dynamic Rising Liquid Background
                Positioned(
                  left: 0,
                  right: 0,
                  bottom: 0,
                  height: MediaQuery.of(context).size.height * progress,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 1000),
                    curve: const ElasticInCurve(0.9),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: _isNightMode
                            ? [const Color(0xFF400000), const Color(0xFF110000)]
                            : [const Color(0xFF76ABFF), const Color(0xFF2A66C9)],
                      ),
                    ),
                  ),
                ),

                // 2. Base Dry Layer (Inactive Solid Grey/Apple Dim Red Text - No Gradient)
                Positioned.fill(
                  child: AnimatedPadding(
                    duration: const Duration(milliseconds: 500),
                    curve: Curves.easeOutCubic,
                    padding: EdgeInsets.only(bottom: bottomPadding),
                    child: _buildDigitsGrid(displayDigits, active: false),
                  ),
                ),

                // 3. Sliding Masked Wet Layer (Solid White/Bright Apple Red Text)
                if (!_zenMode)
                  Positioned.fill(
                    child: ClipPath(
                      clipper: LiquidClipper(progress),
                      child: AnimatedPadding(
                        duration: const Duration(milliseconds: 500),
                        curve: Curves.easeOutCubic,
                        padding: EdgeInsets.only(bottom: bottomPadding),
                        child: _buildDigitsGrid(displayDigits, active: true),
                      ),
                    ),
                  ),

                // 4. Floating control glassmorphic dock
                if (!_zenMode)
                  AnimatedPositioned(
                    duration: const Duration(milliseconds: 500),
                    curve: Curves.easeOutCubic,
                    bottom: _isDockVisible ? 24.0 : -80.0,
                    left: 16.0,
                    right: 16.0,
                    child: Center(
                      child: _buildGlassDock(),
                    ),
                  ),

                // 5. Zen-mode eye reset button
                if (_zenMode && !_isEnded)
                  Positioned(
                    bottom: 40,
                    right: 24,
                    child: GestureDetector(
                      onTap: () => setState(() => _zenMode = false),
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _isNightMode ? const Color(0xFF150000).withOpacity(0.4) : Colors.white.withOpacity(0.05),
                          border: Border.all(
                            color: _isNightMode ? const Color(0xFF800000).withOpacity(0.3) : Colors.white.withOpacity(0.1),
                          ),
                        ),
                        child: Icon(
                          Icons.visibility,
                          color: _isNightMode ? const Color(0xFFFF3B30) : Colors.white70,
                          size: 24,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          );
        }
      ),
    );
  }

  Widget _buildDigitsGrid(List<String> displayDigits, {required bool active}) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        children: [
          Expanded(
            child: Row(
              children: [
                Expanded(child: _buildDigitWidget(displayDigits[0], active)),
                Expanded(child: _buildDigitWidget(displayDigits[1], active)),
              ],
            ),
          ),
          const SizedBox(height: 8.0),
          Expanded(
            child: Row(
              children: [
                Expanded(child: _buildDigitWidget(displayDigits[2], active)),
                Expanded(child: _buildDigitWidget(displayDigits[3], active)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDigitWidget(String val, bool active) {
    Color textColor;
    if (_isNightMode) {
      textColor = active ? const Color(0xFFFF3B30) : const Color(0xFFFF3B30).withOpacity(0.35);
    } else {
      textColor = active ? Colors.white : Colors.white.withOpacity(0.09);
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        return Center(
          // FittedBox forces the font horizontally to maximize standard screen space
          child: FittedBox(
            fit: BoxFit.fill,
            child: Transform(
              alignment: Alignment.center,
              transform: Matrix4.identity()..scale(0.93, 1.35), 
              child: Text(
                val,
                style: TextStyle(
                  fontWeight: FontWeight.w800,
                  color: textColor,
                  fontFamily: 'Montserrat',
                  height: 0.7,
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  void handleDone() => Navigator.maybePop(context);

  Widget _buildGlassDock() {
    if (_isEnded) {
      return GestureDetector(
        onTap: handleDone,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(32.0),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
            child: Container(
              height: 64.0,
              width: double.infinity,
              decoration: BoxDecoration(
                color: _isNightMode ? const Color(0xFF150000).withOpacity(0.2) : Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(32.0),
                border: Border.all(
                  color: _isNightMode ? const Color(0xFF400000).withOpacity(0.2) : Colors.white.withOpacity(0.3),
                ),
              ),
              child: Center(
                child: Text(
                  'Done',
                  style: TextStyle(
                    color: _isNightMode ? const Color(0xFFFF3B30) : Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 18.0,
                  ),
                ),
              ),
            ),
          ),
        ),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(32.0),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          height: 64.0,
          padding: const EdgeInsets.symmetric(horizontal: 16.0),
          decoration: BoxDecoration(
            color: _isNightMode ? const Color(0xFF150000).withOpacity(0.15) : Colors.white.withOpacity(0.1),
            borderRadius: BorderRadius.circular(32.0),
            border: Border.all(
              color: _isNightMode ? const Color(0xFF400000).withOpacity(0.15) : Colors.white.withOpacity(0.2),
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              IconButton(
                icon: Icon(_isRunning ? Icons.pause : Icons.play_arrow),
                iconSize: 24,
                color: _isNightMode ? const Color(0xFFFF3B30) : Colors.white,
                onPressed: _handlePlayPause,
              ),
              IconButton(
                icon: const Icon(Icons.refresh),
                iconSize: 22,
                color: _isNightMode ? const Color(0xFFFF3B30) : Colors.white,
                onPressed: _reset,
              ),
              VerticalDivider(
                color: _isNightMode ? const Color(0xFF400000).withOpacity(0.15) : Colors.white.withOpacity(0.2),
                indent: 18,
                endIndent: 18,
              ),
              IconButton(
                icon: const Icon(Icons.timer_outlined),
                iconSize: 22,
                color: _mode == TimerMode.stopwatch
                    ? (_isNightMode ? const Color(0xFFFF3B30) : Colors.white)
                    : (_isNightMode ? const Color(0xFFFF3B30).withOpacity(0.4) : Colors.white30),
                onPressed: _isRunning ? null : () => _handleModeSwitch(TimerMode.stopwatch),
              ),
              IconButton(
                icon: const Icon(Icons.coffee_outlined),
                iconSize: 22,
                color: _mode == TimerMode.pomodoro
                    ? (_isNightMode ? const Color(0xFFFF3B30) : Colors.white)
                    : (_isNightMode ? const Color(0xFFFF3B30).withOpacity(0.4) : Colors.white30),
                onPressed: _isRunning ? null : () => _handleModeSwitch(TimerMode.pomodoro),
              ),
              IconButton(
                icon: const Icon(Icons.visibility_off_outlined),
                iconSize: 22,
                color: _isNightMode ? const Color(0xFFFF3B30).withOpacity(0.4) : Colors.white30,
                onPressed: () => setState(() => _zenMode = true),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// Slit clip mask logic (replicates CSS clip-path inset perfectly)
class LiquidClipper extends CustomClipper<Path> {
  final double progress;
  LiquidClipper(this.progress);

  @override
  Path getClip(Size size) {
    final path = Path();
    final double yOffset = size.height * (1.0 - progress);
    path.addRect(Rect.fromLTRB(0, yOffset, size.width, size.height));
    return path;
  }

  @override
  bool shouldReclip(covariant LiquidClipper oldClipper) => oldClipper.progress != progress;
}
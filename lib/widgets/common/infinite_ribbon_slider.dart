import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

class InfiniteRibbonSlider extends StatefulWidget {
  final int minMinutes;
  final int maxMinutes;
  final int initialMinutes;
  final ValueChanged<int> onChanged;
  final double height;
  final bool showValue;
  final Color accentColor;

  const InfiniteRibbonSlider({
    super.key,
    this.minMinutes = 5,
    this.maxMinutes = 120,
    this.initialMinutes = 25,
    required this.onChanged,
    this.height = 48.0,
    this.showValue = true,
    this.accentColor = const Color(0xFF3A86FF),
  });

  @override
  State<InfiniteRibbonSlider> createState() => _InfiniteRibbonSliderState();
}

class _InfiniteRibbonSliderState extends State<InfiniteRibbonSlider> {
  late FixedExtentScrollController _scrollController;
  late int _selectedMinutes;
  int _lastNotifiedIndex = -1;

  @override
  void initState() {
    super.initState();
    _selectedMinutes =
        widget.initialMinutes.clamp(widget.minMinutes, widget.maxMinutes);
    _scrollController = FixedExtentScrollController(
      initialItem: _selectedMinutes - widget.minMinutes,
    );
    _lastNotifiedIndex = _selectedMinutes - widget.minMinutes;
  }

  @override
  void didUpdateWidget(InfiniteRibbonSlider oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.initialMinutes != oldWidget.initialMinutes) {
      final clamped =
          widget.initialMinutes.clamp(widget.minMinutes, widget.maxMinutes);
      final newIndex = clamped - widget.minMinutes;
      _selectedMinutes = clamped;
      _lastNotifiedIndex = newIndex;
      _scrollController.jumpToItem(newIndex);
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final tickCount = widget.maxMinutes - widget.minMinutes + 1;

    return SizedBox(
      height: widget.height,
      child: ClipRect(
        child: Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              width: double.infinity,
              child: ListWheelScrollView.useDelegate(
                controller: _scrollController,
                itemExtent: 20.0,
                diameterRatio: 10.0,
                physics: const FixedExtentScrollPhysics(),
                perspective: 0.0005,
                onSelectedItemChanged: (index) {
                  final minutes = index + widget.minMinutes;
                  setState(() => _selectedMinutes = minutes);
                  if (index != _lastNotifiedIndex) {
                    _lastNotifiedIndex = index;
                    HapticFeedback.selectionClick();
                    widget.onChanged(minutes);
                  }
                },
                childDelegate: ListWheelChildBuilderDelegate(
                  childCount: tickCount,
                  builder: (context, index) {
                    final minutes = index + widget.minMinutes;
                    final distance = (minutes - _selectedMinutes).abs();
                    const maxDistance = 30;
                    final normalizedDistance =
                        distance > maxDistance ? 1.0 : distance / maxDistance;

                    final isMajor = minutes % 5 == 0;
                    final isCenter = minutes == _selectedMinutes;

                    double tickHeight;
                    double opacity;

                    if (isCenter) {
                      tickHeight = 40.0;
                      opacity = 1.0;
                    } else if (distance > maxDistance) {
                      tickHeight = 4.0;
                      opacity = 0.1;
                    } else {
                      tickHeight = 4.0 + (36.0 * (1 - normalizedDistance));
                      opacity = 0.1 + (0.7 * (1 - normalizedDistance));
                    }

                    return Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          if (isMajor && isCenter && widget.showValue)
                            Padding(
                              padding: const EdgeInsets.only(bottom: 4),
                              child: Text(
                                '$minutes',
                                style: GoogleFonts.googleSans(
                                  color: widget.accentColor
                                      .withValues(alpha: opacity),
                                  fontSize: 14,
                                  fontWeight: FontWeight.w700,
                                  height: 1,
                                ),
                              ),
                            ),
                          Container(
                            width: isMajor ? 2.0 : 1.0,
                            height: tickHeight,
                            decoration: BoxDecoration(
                              color: isCenter
                                  ? widget.accentColor
                                  : Colors.white.withValues(alpha: opacity),
                              borderRadius: BorderRadius.circular(1),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ),
            Container(
              width: 2.0,
              height: 30.0,
              decoration: BoxDecoration(
                color: widget.accentColor,
                boxShadow: [
                  BoxShadow(
                    color: widget.accentColor.withValues(alpha: 0.5),
                    blurRadius: 8,
                    spreadRadius: 2,
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

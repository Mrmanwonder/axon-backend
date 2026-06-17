import 'dart:math' as math;

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';

import '../../../models/models.dart';
import '../../../theme/app_theme.dart';
import '../../../utils/layout_utils.dart';
import 'common_components.dart';

class TrendCard extends StatelessWidget {
  final List<DailyMetrics> history;

  const TrendCard({required this.history});

  @override
  Widget build(BuildContext context) {
    return Panel(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SectionTitle(
              title: '7-Day Trajectory', trailing: 'Focus score'),
          const SizedBox(height: 16),
          SizedBox(
            height: 210,
            child: history.isEmpty ? const EmptyTrend() : TrendChart(history),
          ),
        ],
      ),
    ).animateIf(PageIntroService.shouldAnimate('analysis_trend'), [
      (a) => a.fadeIn(delay: 180.ms),
    ]);
  }
}

class EmptyTrend extends StatelessWidget {
  const EmptyTrend();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.show_chart_rounded,
              color: AxonColors.textTertiary, size: 28),
          const SizedBox(height: 8),
          Text(
            'Start a study session to build the trend line.',
            textAlign: TextAlign.center,
            style: GoogleFonts.inter(
              color: AxonColors.textTertiary,
              fontSize: 12,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }
}

class TrendChart extends StatelessWidget {
  final List<DailyMetrics> history;

  const TrendChart(this.history);

  @override
  Widget build(BuildContext context) {
    final visible =
        history.length > 7 ? history.sublist(history.length - 7) : history;
    final spots = <FlSpot>[];
    final labels = <int, String>{};
    final formatter = DateFormat('EEE');

    for (var i = 0; i < visible.length; i++) {
      final entry = visible[i];
      spots.add(FlSpot(
        i.toDouble(),
        entry.predictedPerformance.clamp(0.0, 1.0),
      ));
      labels[i] = formatter.format(entry.date);
    }

    return LineChart(
      LineChartData(
        minX: 0,
        maxX: math.max(1, visible.length - 1).toDouble(),
        minY: 0,
        maxY: 1,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 0.25,
          getDrawingHorizontalLine: (_) => FlLine(
            color: AxonColors.divider.withValues(alpha: 0.3),
            strokeWidth: 0.8,
          ),
        ),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 34,
              interval: 0.5,
              getTitlesWidget: (value, _) => Text(
                '${(value * 100).round()}',
                style: GoogleFonts.inter(
                  color: AxonColors.textTertiary,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              interval: 1,
              getTitlesWidget: (value, _) {
                final label = labels[value.round()];
                if (label == null) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    label,
                    style: GoogleFonts.inter(
                      color: AxonColors.textTertiary,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                );
              },
            ),
          ),
        ),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipItems: (items) => items
                .map(
                  (item) => LineTooltipItem(
                    '${(item.y * 100).round()}%',
                    GoogleFonts.inter(
                      color: AxonColors.accent,
                      fontSize: 12,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                )
                .toList(),
          ),
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.28,
            color: AxonColors.accent,
            barWidth: 3,
            dotData: FlDotData(
              show: true,
              getDotPainter: (spot, _, __, ___) => FlDotCirclePainter(
                radius: 4,
                color: AxonColors.performanceColor(spot.y),
                strokeWidth: 1.5,
                strokeColor: AxonColors.surfaceElevated,
              ),
            ),
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AxonColors.accent.withValues(alpha: 0.18),
                  AxonColors.accent.withValues(alpha: 0.02),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

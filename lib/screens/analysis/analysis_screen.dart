// lib/screens/analysis/analysis_screen.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../router/app_router.dart';
import '../../services/app_state.dart';
import '../../services/app_usage_service.dart';
import '../../services/ask_axon_context_service.dart';
import '../../services/report_card_service.dart';
import '../../models/models.dart';
import '../../theme/app_theme.dart';
import '../../utils/layout_utils.dart';
import '../../widgets/ask_axon_logo.dart';
import 'widgets/action_strip.dart';
import 'widgets/analytics_header.dart';
import 'widgets/empty_analytics.dart';
import 'widgets/insight_card.dart';
import 'widgets/readiness_card.dart';
import 'widgets/trend_chart.dart';
import 'widgets/university_tracker.dart';
import 'widgets/vitals_panel.dart';

class AnalysisScreen extends ConsumerWidget {
  const AnalysisScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final metrics = ref.watch(metricsProvider);
    final auth = ref.watch(authStateProvider);
    final user = auth.user;
    final style =
        user?.motivationStyle ?? MotivationStyle.positiveReinforcement;
    final score = metrics.predictedPerformance.clamp(0.0, 1.0);
    final hasData = metrics.hasData;

    return Scaffold(
      backgroundColor: Colors.transparent,
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
      floatingActionButton: Padding(
        padding: EdgeInsets.only(
          bottom: bottomDockClearance(context, extra: 0) -
              MediaQuery.of(context).padding.bottom -
              16,
        ),
        child: AskAxonOrbButton(
          isWorking: false,
          enableLongPressBuildUp: true,
          onTap: () => context.push(
            AppRoutes.ai,
            extra: {
              'title': 'Ask Axon',
              'contextFuture': AskAxonContextService.instance.buildContext(
                metrics: metrics,
                currentSubject: metrics.primarySubject,
              ),
              'motivationStyle': style,
            },
          ),
        ),
      ),
      body: Container(
        decoration: BoxDecoration(gradient: AxonGradients.backgroundGradient),
        child: SafeArea(
          child: RefreshIndicator(
            color: AxonColors.accent,
            onRefresh: () => ref.read(metricsProvider.notifier).refreshMetrics(),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final isWide = constraints.maxWidth >= 900;
                final horizontalPadding = isWide ? 32.0 : 20.0;

                return ListView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  padding: EdgeInsets.fromLTRB(
                    horizontalPadding,
                    20,
                    horizontalPadding,
                    bottomDockClearance(context),
                  ),
                  children: [
                    AnalyticsHeader(
                      metrics: metrics,
                      score: score,
                      style: style,
                      hasData: hasData,
                    ),
                    const SizedBox(height: 20),
                    if (!hasData) ...[
                      EmptyAnalytics(
                        onStartTimer: () => context.push('/timer'),
                        onImportReport: () =>
                            ReportCardService().importReportCardAndUpdateScore(
                          context,
                          ref,
                        ),
                      ),
                      const SizedBox(height: 20),
                    ],
                    if (isWide)
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            flex: 6,
                            child: Column(
                              children: [
                                ReadinessCard(metrics: metrics, score: score),
                                const SizedBox(height: 16),
                                TrendCard(history: metrics.weekHistory),
                              ],
                            ),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            flex: 4,
                            child: Column(
                              children: [
                                VitalsPanel(metrics: metrics, hasData: hasData),
                                const SizedBox(height: 16),
                                InsightCard(
                                  score: score,
                                  style: style,
                                  metrics: metrics,
                                ),
                                const SizedBox(height: 16),
                                UniversityTracker(uid: user?.uid),
                              ],
                            ),
                          ),
                        ],
                      )
                    else ...[
                      ReadinessCard(metrics: metrics, score: score),
                      const SizedBox(height: 16),
                      VitalsPanel(metrics: metrics, hasData: hasData),
                      const SizedBox(height: 16),
                      TrendCard(history: metrics.weekHistory),
                      const SizedBox(height: 16),
                      InsightCard(score: score, style: style, metrics: metrics),
                      const SizedBox(height: 16),
                      UniversityTracker(uid: user?.uid),
                    ],
                    const SizedBox(height: 16),
                    ActionStrip(
                      hasData: hasData,
                      onTimeline: () => context.push('/calendar'),
                      onTimer: () => context.push('/timer'),
                      onReportCard: () =>
                          ReportCardService().importReportCardAndUpdateScore(
                        context,
                        ref,
                      ),
                      onUsageAccess: () => AppUsageService().requestUsageAccess(),
                    ),
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }
}

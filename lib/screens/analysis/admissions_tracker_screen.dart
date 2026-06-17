import 'dart:async';
import 'dart:ui';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:file_picker/file_picker.dart';

import '../../models/admissions_models.dart';
import '../../services/alert_service.dart';
import '../../services/app_state.dart';
import '../../services/admissions_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/admissions/milestone_progress_bar.dart';
import '../../utils/layout_utils.dart';

TextDecoration? _lineThrough(bool condition) =>
    condition ? TextDecoration.lineThrough : null;

class AdmissionsTrackerScreen extends ConsumerWidget {
  const AdmissionsTrackerScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final authUser = ref.watch(authStateProvider).user;
    final uid = authUser?.uid ?? '';
    final metrics = ref.watch(metricsProvider);
    final service = AdmissionsService();

    return Scaffold(
      backgroundColor: AxonColors.background,
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          Positioned(
            top: -100,
            right: -50,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AxonColors.accent.withValues(alpha: 0.08),
              ),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
                child: Container(),
              ),
            ),
          ),
          SafeArea(
            child: Column(
              children: [
                _buildHeader(context, ref),
                Expanded(
                  child: StreamBuilder<List<AdmissionsMilestone>>(
                    stream: service.watchMilestones(uid),
                    builder: (context, milestoneSnapshot) {
                      final milestones = milestoneSnapshot.data ??
                          const <AdmissionsMilestone>[];
                      return StreamBuilder<List<AdmissionsVaultAsset>>(
                        stream: service.watchVaultAssets(uid),
                        builder: (context, vaultSnapshot) {
                          final assets = vaultSnapshot.data ??
                              const <AdmissionsVaultAsset>[];
                          final progress =
                              service.buildProgress(milestones, assets: assets);
                          final nodes = service.buildMilestoneMap(milestones);
                          return StreamBuilder<List<AdmissionsTarget>>(
                            stream: service.watchTargets(uid),
                            builder: (context, targetSnapshot) {
                              final targets = targetSnapshot.data ??
                                  const <AdmissionsTarget>[];
                              final targetsById = {
                                for (final target in targets) target.id: target
                              };
                              return _buildContent(
                                context,
                                ref,
                                uid,
                                metrics,
                                authUser,
                                service,
                                targets,
                                targetsById,
                                nodes,
                                assets,
                                const <AdmissionsMarkEntry>[],
                                progress,
                              );
                            },
                          );
                        },
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      floatingActionButton:
          _buildFab(context, ref, uid, metrics, authUser, service),
    );
  }

  Widget _buildHeader(BuildContext context, WidgetRef ref) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      child: Row(
        children: [
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AxonColors.surface,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AxonColors.divider, width: 1),
              ),
              child: Icon(Icons.arrow_back_rounded,
                  color: AxonColors.textPrimary, size: 20),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('ADMISSIONS',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.accent,
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 2)),
                const SizedBox(height: 4),
                Text('University Applications',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 20,
                        fontWeight: FontWeight.w700)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildContent(
    BuildContext context,
    WidgetRef ref,
    String uid,
    MetricsState metrics,
    dynamic authUser,
    AdmissionsService service,
    List<AdmissionsTarget> targets,
    Map<String, AdmissionsTarget> targetsById,
    List<AdmissionsMilestoneMapNode> nodes,
    List<AdmissionsVaultAsset> assets,
    List<AdmissionsMarkEntry> marks,
    AdmissionsMilestoneProgress progress,
  ) {
    return StreamBuilder<List<AdmissionsMarkEntry>>(
      stream: service.watchMarks(uid),
      builder: (context, markSnapshot) {
        final markEntries = markSnapshot.data ?? marks;
        final subjects = _readSubjects(authUser);
        final board = _readBoard(authUser);
        return ListView(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          children: [
            _TrajectoryBanner(metrics: metrics),
            const SizedBox(height: 20),
            _UniversityDiscoveryPanel(
              service: service,
              uid: uid,
              board: board,
              subjects: subjects,
              readinessScore: metrics.predictedPerformance.clamp(0.0, 1.0),
            ),
            const SizedBox(height: 20),
            MilestoneProgressBar(progress: progress),
            const SizedBox(height: 24),
            _buildSectionHeader('TARGET_UNIVERSITIES', Icons.school_rounded),
            const SizedBox(height: 12),
            if (targets.isEmpty)
              _buildEmptyState(
                icon: Icons.school_outlined,
                title: 'No targets generated yet',
                subtitle:
                    'Generate a university fit or save a program from discovery.',
                actionLabel: 'START',
                onAction: authUser == null
                    ? null
                    : () => _showGenerateFitSheet(
                        context, ref, uid, service, metrics, subjects),
              )
            else
              for (final target in targets) _buildTargetCard(context, target),
            const SizedBox(height: 24),
            _buildSectionHeader('MARKS_DASHBOARD', Icons.query_stats_rounded),
            const SizedBox(height: 12),
            _MarksDashboard(
                service: service,
                uid: uid,
                targets: targets,
                marks: markEntries),
            const SizedBox(height: 24),
            _buildSectionHeader('MILESTONE_MAP', Icons.account_tree_rounded),
            const SizedBox(height: 12),
            if (nodes.isEmpty)
              _buildEmptyState(
                icon: Icons.flag_outlined,
                title: 'No milestones yet',
                subtitle: 'Your application milestones will appear here.',
              )
            else
              for (final node in nodes) _buildMilestoneCard(node, targetsById),
            const SizedBox(height: 24),
            _buildSectionHeader(
                'ACHIEVEMENT_VAULT', Icons.workspace_premium_rounded),
            const SizedBox(height: 12),
            _AchievementVaultPanel(
                service: service, uid: uid, assets: assets, targets: targets),
            SizedBox(height: bottomDockClearance(context)),
          ],
        );
      },
    );
  }

  String _readBoard(dynamic authUser) {
    try {
      return (authUser?.board ?? '').toString();
    } catch (_) {
      return '';
    }
  }

  List<String> _readSubjects(dynamic authUser) {
    try {
      return (authUser?.subjects as List? ?? const [])
          .map((item) => item.toString())
          .where((item) => item.isNotEmpty)
          .toList();
    } catch (_) {
      return const [];
    }
  }

  Widget _buildSectionHeader(String label, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: AxonColors.accent, size: 16),
        const SizedBox(width: 8),
        Text(label,
            style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 10,
                fontWeight: FontWeight.w600,
                letterSpacing: 2)),
      ],
    );
  }

  Widget _buildEmptyState(
      {required IconData icon,
      required String title,
      required String subtitle,
      String? actionLabel,
      VoidCallback? onAction}) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.divider, width: 1),
      ),
      child: Column(
        children: [
          Icon(icon, color: AxonColors.textTertiary, size: 40),
          const SizedBox(height: 16),
          Text(title,
              style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text(subtitle,
              style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary, fontSize: 12),
              textAlign: TextAlign.center),
          if (actionLabel != null && onAction != null) ...[
            const SizedBox(height: 16),
            GestureDetector(
              onTap: onAction,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: AxonColors.accent.withValues(alpha: 0.3)),
                ),
                child: Text(actionLabel,
                    style: GoogleFonts.googleSans(
                        color: AxonColors.accent,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1)),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildTargetCard(BuildContext context, AdmissionsTarget target) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.divider, width: 1),
      ),
      child: GestureDetector(
        onTap: () => _showTargetDetailSheet(context, target),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(Icons.school_rounded,
                    color: AxonColors.accent, size: 24),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(target.universityName,
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary,
                            fontSize: 14,
                            fontWeight: FontWeight.w600)),
                    const SizedBox(height: 2),
                    Text(target.courseName,
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textSecondary, fontSize: 12)),
                    const SizedBox(height: 6),
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          _buildChip(target.country, AxonColors.accent),
                          const SizedBox(width: 8),
                          _buildChip(target.applicationSystem,
                              AxonColors.textTertiary.withValues(alpha: 0.3)),
                          if (target.fitBand.isNotEmpty) ...[
                            const SizedBox(width: 8),
                            _buildChip(target.fitBand,
                                _getFitBandColor(target.fitBand)),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  if (target.deadlineAt != null) ...[
                    Text(_formatDeadline(target.deadlineAt!),
                        style: GoogleFonts.googleSans(
                            color: _isDeadlineNear(target.deadlineAt!)
                                ? AxonColors.error
                                : AxonColors.textTertiary,
                            fontSize: 10,
                            fontWeight: FontWeight.w500)),
                    const SizedBox(height: 4),
                  ],
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color:
                          _getStatusColor(target.status).withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(target.status.toUpperCase(),
                        style: GoogleFonts.googleSans(
                            color: _getStatusColor(target.status),
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            letterSpacing: 1)),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showTargetDetailSheet(BuildContext context, AdmissionsTarget target) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: AxonColors.background,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border.all(color: AxonColors.divider, width: 0.5),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                  margin: const EdgeInsets.only(top: 12),
                  width: 36,
                  height: 4,
                  decoration: BoxDecoration(
                      color: AxonColors.divider,
                      borderRadius: BorderRadius.circular(2))),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                child: Row(children: [
                  Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                          color: AxonColors.accent.withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(12)),
                      child: Icon(Icons.school_rounded,
                          color: AxonColors.accent, size: 20)),
                  const SizedBox(width: 12),
                  Expanded(
                      child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                        Text(target.universityName,
                            style: GoogleFonts.googleSans(
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                                color: AxonColors.textPrimary)),
                        Text(target.courseName,
                            style: GoogleFonts.googleSans(
                                fontSize: 12, color: AxonColors.textSecondary)),
                      ])),
                ]),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(children: [
                        _infoChip(Icons.location_on_outlined, target.country),
                        const SizedBox(width: 12),
                        _infoChip(Icons.flag_rounded, target.fitBand),
                        if (target.deadlineAt != null) ...[
                          const SizedBox(width: 12),
                          _infoChip(Icons.calendar_today_rounded,
                              _formatDeadline(target.deadlineAt!)),
                        ],
                      ]),
                      if (target.entryRequirements.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        Text('ENTRY REQUIREMENTS',
                            style: GoogleFonts.googleSans(
                                color: AxonColors.accent,
                                fontSize: 9,
                                letterSpacing: 1.5,
                                fontWeight: FontWeight.w600)),
                        const SizedBox(height: 4),
                        Text(target.entryRequirements,
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary,
                                fontSize: 11,
                                height: 1.4)),
                      ],
                      if (target.rationale.isNotEmpty) ...[
                        const SizedBox(height: 12),
                        Text('RATIONALE',
                            style: GoogleFonts.googleSans(
                                color: AxonColors.accent,
                                fontSize: 9,
                                letterSpacing: 1.5,
                                fontWeight: FontWeight.w600)),
                        const SizedBox(height: 4),
                        Text(target.rationale,
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textSecondary, fontSize: 11)),
                      ],
                      const SizedBox(height: 20),
                    ]),
              ),
              Container(
                width: double.infinity,
                padding: EdgeInsets.fromLTRB(
                    20, 12, 20, MediaQuery.of(ctx).padding.bottom + 16),
                decoration: BoxDecoration(
                    border: Border(
                        top:
                            BorderSide(color: AxonColors.divider, width: 0.5))),
                child: GestureDetector(
                  onTap: () => Navigator.pop(ctx),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    decoration: BoxDecoration(
                        color: AxonColors.accent,
                        borderRadius: BorderRadius.circular(12)),
                    child: Center(
                        child: Text('CLOSE',
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary,
                                fontSize: 12,
                                letterSpacing: 1,
                                fontWeight: FontWeight.w600))),
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _infoChip(IconData icon, String label) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, color: AxonColors.textTertiary, size: 12),
      const SizedBox(width: 3),
      Text(label,
          style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary, fontSize: 10)),
    ]);
  }

  Widget _buildMilestoneCard(AdmissionsMilestoneMapNode node,
      Map<String, AdmissionsTarget> targetsById) {
    final target = node.milestone.targetId.isNotEmpty
        ? targetsById[node.milestone.targetId]
        : null;
    final isCompleted = node.milestone.completed;
    final isBlocked = node.milestone.isBlocked && !isCompleted;
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color: isCompleted
                ? AxonColors.success.withValues(alpha: 0.2)
                : isBlocked
                    ? AxonColors.warning.withValues(alpha: 0.2)
                    : AxonColors.divider,
            width: 1),
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isCompleted
                    ? AxonColors.success.withValues(alpha: 0.2)
                    : isBlocked
                        ? AxonColors.warning.withValues(alpha: 0.2)
                        : AxonColors.surfaceElevated),
            child: Icon(
                isCompleted
                    ? Icons.check_rounded
                    : isBlocked
                        ? Icons.lock_rounded
                        : Icons.radio_button_unchecked_rounded,
                color: isCompleted
                    ? AxonColors.success
                    : isBlocked
                        ? AxonColors.warning
                        : AxonColors.textTertiary,
                size: 18),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(node.milestone.title,
                    style: GoogleFonts.googleSans(
                        color: isCompleted
                            ? AxonColors.textSecondary
                            : AxonColors.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        decoration: _lineThrough(isCompleted))),
                const SizedBox(height: 4),
                Row(
                  children: [
                    if (target != null) ...[
                      Text(target.universityName,
                          style: GoogleFonts.googleSans(
                              color: AxonColors.accent, fontSize: 10)),
                      Text(' • ',
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary, fontSize: 10)),
                    ],
                    Text(
                        node.milestone.phase.isNotEmpty
                            ? node.milestone.phase
                            : 'No phase',
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textTertiary, fontSize: 10)),
                  ],
                ),
                if (isBlocked && node.blockerTitles.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                        color: AxonColors.warning.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(6)),
                    child: Text('Blocked: ${node.blockerTitles.join(', ')}',
                        style: GoogleFonts.googleSans(
                            color: AxonColors.warning,
                            fontSize: 9,
                            fontWeight: FontWeight.w500)),
                  ),
                ],
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(_formatDate(node.milestone.dueAt),
                  style: GoogleFonts.googleSans(
                      color: _isDeadlineNear(node.milestone.dueAt)
                          ? AxonColors.error
                          : AxonColors.textTertiary,
                      fontSize: 9,
                      fontWeight: FontWeight.w500)),
            ],
          ),
        ],
      ),
    );
  }

  Widget buildAssetCard(AdmissionsVaultAsset asset) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AxonColors.divider, width: 1),
      ),
      child: Row(
        children: [
          Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10)),
              child: Icon(_getAssetIcon(asset.assetType),
                  color: AxonColors.accent, size: 20)),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(asset.title,
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Wrap(spacing: 6, children: [
                  _buildChip(asset.assetType,
                      AxonColors.textTertiary.withValues(alpha: 0.3)),
                  if (asset.provider.isNotEmpty)
                    _buildChip(asset.provider,
                        AxonColors.accent.withValues(alpha: 0.3)),
                ]),
                if (asset.tags.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Text(asset.tags.join(' • '),
                      style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 9,
                          fontWeight: FontWeight.w500)),
                ],
              ],
            ),
          ),
          Icon(Icons.chevron_right_rounded,
              color: AxonColors.textTertiary, size: 20),
        ],
      ),
    );
  }

  Widget _buildChip(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
          color: color.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(6)),
      child: Text(label,
          style: GoogleFonts.googleSans(
              color: color == AxonColors.textTertiary.withValues(alpha: 0.3)
                  ? AxonColors.textTertiary
                  : color,
              fontSize: 9,
              fontWeight: FontWeight.w500)),
    );
  }

  Widget _buildFab(BuildContext context, WidgetRef ref, String uid,
      MetricsState metrics, dynamic authUser, AdmissionsService service) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
              color: AxonColors.accent.withValues(alpha: 0.3),
              blurRadius: 20,
              offset: const Offset(0, 4))
        ],
      ),
      child: FloatingActionButton.extended(
        onPressed: authUser == null
            ? null
            : () => _showGenerateFitSheet(
                context, ref, uid, service, metrics, authUser.subjects),
        backgroundColor: AxonColors.accent,
        icon: const Icon(Icons.auto_awesome_rounded, size: 18),
        label: Text('GENERATE_FIT',
            style: GoogleFonts.googleSans(
                fontSize: 10, letterSpacing: 1, fontWeight: FontWeight.w600)),
      ),
    );
  }

  Future<void> _showGenerateFitSheet(
      BuildContext context,
      WidgetRef ref,
      String uid,
      AdmissionsService service,
      MetricsState metrics,
      List<String> subjects) async {
    final courseCtrl = TextEditingController();
    final deadlineCtrl = TextEditingController();
    final countries = <String>{'UK'};

    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: AxonColors.background,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border.all(color: AxonColors.divider, width: 0.5),
          ),
          child: StatefulBuilder(
            builder: (ctx, setState) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                      margin: const EdgeInsets.only(top: 12),
                      width: 36,
                      height: 4,
                      decoration: BoxDecoration(
                          color: AxonColors.divider,
                          borderRadius: BorderRadius.circular(2))),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                    child: Row(
                      children: [
                        Container(
                            padding: const EdgeInsets.all(10),
                            decoration: BoxDecoration(
                                color: AxonColors.accent.withValues(alpha: 0.2),
                                borderRadius: BorderRadius.circular(12)),
                            child: Icon(Icons.auto_awesome_rounded,
                                color: AxonColors.accent, size: 20)),
                        const SizedBox(width: 12),
                        Expanded(
                            child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                              Text('Generate Admissions Fit',
                                  style: GoogleFonts.googleSans(
                                      fontSize: 18,
                                      fontWeight: FontWeight.w700,
                                      color: AxonColors.textPrimary)),
                              Text('AI matches your profile to universities',
                                  style: GoogleFonts.googleSans(
                                      fontSize: 11,
                                      color: AxonColors.textTertiary)),
                            ])),
                      ],
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 2),
                      decoration: BoxDecoration(
                          color: AxonColors.surfaceElevated,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: AxonColors.divider)),
                      child: TextField(
                        controller: courseCtrl,
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary, fontSize: 13),
                        decoration: InputDecoration(
                          labelText: 'Degree / Course',
                          labelStyle: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary, fontSize: 11),
                          hintText: 'e.g. Computer Science, Medicine',
                          hintStyle: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary
                                  .withValues(alpha: 0.5),
                              fontSize: 12),
                          border: InputBorder.none,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(children: [
                            Icon(Icons.public_rounded,
                                color: AxonColors.textTertiary, size: 13),
                            const SizedBox(width: 6),
                            Text('Countries',
                                style: GoogleFonts.googleSans(
                                    color: AxonColors.textSecondary,
                                    fontSize: 11)),
                          ]),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              'UK',
                              'Singapore',
                              'Hong Kong',
                              'Australia',
                              'Canada',
                              'USA'
                            ].map((c) {
                              final sel = countries.contains(c);
                              return GestureDetector(
                                onTap: () => setState(() {
                                  if (sel && countries.length > 1) {
                                    countries.remove(c);
                                  } else if (!sel) {
                                    countries.add(c);
                                  }
                                }),
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 12, vertical: 6),
                                  decoration: BoxDecoration(
                                    color: sel
                                        ? AxonColors.accent
                                            .withValues(alpha: 0.2)
                                        : Colors.transparent,
                                    borderRadius: BorderRadius.circular(16),
                                    border: Border.all(
                                        color: sel
                                            ? AxonColors.accent
                                            : AxonColors.divider),
                                  ),
                                  child: Text(c,
                                      style: GoogleFonts.googleSans(
                                          color: sel
                                              ? AxonColors.accent
                                              : AxonColors.textSecondary,
                                          fontSize: 10,
                                          fontWeight: FontWeight.w500)),
                                ),
                              );
                            }).toList(),
                          ),
                        ]),
                  ),
                  const SizedBox(height: 12),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 2),
                      decoration: BoxDecoration(
                          color: AxonColors.surfaceElevated,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: AxonColors.divider)),
                      child: Row(children: [
                        Icon(Icons.calendar_today_rounded,
                            color: AxonColors.textTertiary, size: 14),
                        const SizedBox(width: 8),
                        Expanded(
                            child: TextField(
                          controller: deadlineCtrl,
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textPrimary, fontSize: 13),
                          decoration: InputDecoration(
                            labelText: 'Target year',
                            labelStyle: GoogleFonts.googleSans(
                                color: AxonColors.textTertiary, fontSize: 11),
                            hintText: 'e.g. 2025, 2026',
                            hintStyle: GoogleFonts.googleSans(
                                color: AxonColors.textTertiary
                                    .withValues(alpha: 0.5),
                                fontSize: 12),
                            border: InputBorder.none,
                          ),
                          keyboardType: TextInputType.number,
                        )),
                      ]),
                    ),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: EdgeInsets.fromLTRB(
                        20, 12, 20, MediaQuery.of(ctx).padding.bottom + 16),
                    decoration: BoxDecoration(
                        border: Border(
                            top: BorderSide(
                                color: AxonColors.divider, width: 0.5))),
                    child: Row(children: [
                      Expanded(
                          child: GestureDetector(
                        onTap: () {
                          Navigator.pop(context);
                          _showManualEntrySheet(
                              context, ref, uid, service, metrics, subjects);
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                              color: AxonColors.surfaceElevated,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: AxonColors.divider)),
                          child: Center(
                              child: Text('ADD MANUALLY',
                                  style: GoogleFonts.googleSans(
                                      color: AxonColors.textPrimary,
                                      fontSize: 12,
                                      letterSpacing: 1,
                                      fontWeight: FontWeight.w600))),
                        ),
                      )),
                      const SizedBox(width: 8),
                      Expanded(
                          child: GestureDetector(
                        onTap: () {
                          if (courseCtrl.text.trim().isEmpty) return;
                          Navigator.pop(context);
                          _generateAndSave(
                              context,
                              uid,
                              service,
                              metrics,
                              subjects,
                              courseCtrl.text.trim(),
                              countries.toList());
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                              color: AxonColors.accent,
                              borderRadius: BorderRadius.circular(12)),
                          child: Center(
                              child: Text('GENERATE FIT',
                                  style: GoogleFonts.googleSans(
                                      color: AxonColors.textPrimary,
                                      fontSize: 12,
                                      letterSpacing: 1,
                                      fontWeight: FontWeight.w600))),
                        ),
                      )),
                    ]),
                  ),
                ],
              );
            },
          ),
        );
      },
    );
  }

  Future<void> _generateAndSave(
      BuildContext context,
      String uid,
      AdmissionsService service,
      MetricsState metrics,
      List<String> subjects,
      String course,
      List<String> countries) async {
    if (!context.mounted) return;
    try {
      final predictedGrades = <String, String>{};
      for (final s in subjects) {
        predictedGrades[s] = 'A';
      }
      if (predictedGrades.isEmpty) {
        predictedGrades['General'] = 'A';
      }
      final fit = await service.generateUniversityFit(
        predictedGrades: predictedGrades,
        targetCourse: course,
        countries: countries,
        readinessScore: metrics.predictedPerformance.clamp(0.0, 1.0),
      );
      await service.persistGeneratedFit(
          uid: uid,
          targetCourse: course,
          fit: fit,
          readinessScore: metrics.predictedPerformance.clamp(0.0, 1.0));
      if (context.mounted) {
        AlertService.showSuccess(
            context, 'Fit generated', 'Universities saved to your targets.');
      }
    } catch (e) {
      if (!context.mounted) return;
      try {
        final parts = course
            .split(RegExp(r'\s[-–—]\s'))
            .map((p) => p.trim())
            .where((p) => p.isNotEmpty)
            .toList();
        final uni = parts.isNotEmpty ? parts.first : course;
        final deg =
            parts.length > 1 ? parts.sublist(1).join(' - ') : 'Target course';
        await service.createManualTarget(
            uid: uid,
            universityName: uni,
            courseName: deg,
            country: countries.first);
        if (context.mounted) {
          AlertService.showWarning(context, 'Saved as manual',
              'AI generation unavailable. Target saved manually.');
        }
      } catch (_) {
        if (context.mounted) {
          AlertService.showWarning(context, 'Generation failed', e.toString());
        }
      }
    }
  }

  Future<void> _showManualEntrySheet(
      BuildContext context,
      WidgetRef ref,
      String uid,
      AdmissionsService service,
      MetricsState metrics,
      List<String> subjects) async {
    final nameCtrl = TextEditingController();
    final courseCtrl = TextEditingController();
    final countryCtrl = TextEditingController();
    String selectedBucket = 'Reach';

    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        return Container(
          margin: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: AxonColors.background,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border.all(color: AxonColors.divider, width: 0.5),
          ),
          child: StatefulBuilder(
            builder: (ctx, setState) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                      margin: const EdgeInsets.only(top: 12),
                      width: 36,
                      height: 4,
                      decoration: BoxDecoration(
                          color: AxonColors.divider,
                          borderRadius: BorderRadius.circular(2))),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 16, 20, 12),
                    child: Row(children: [
                      Container(
                          padding: const EdgeInsets.all(10),
                          decoration: BoxDecoration(
                              color: AxonColors.accent.withValues(alpha: 0.2),
                              borderRadius: BorderRadius.circular(12)),
                          child: Icon(Icons.add_business_rounded,
                              color: AxonColors.accent, size: 20)),
                      const SizedBox(width: 12),
                      Text('Add University',
                          style: GoogleFonts.googleSans(
                              fontSize: 18,
                              fontWeight: FontWeight.w700,
                              color: AxonColors.textPrimary)),
                    ]),
                  ),
                  const SizedBox(height: 12),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 14, vertical: 2),
                        decoration: BoxDecoration(
                            color: AxonColors.surfaceElevated,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: AxonColors.divider)),
                        child: TextField(
                            controller: nameCtrl,
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary, fontSize: 13),
                            decoration: InputDecoration(
                                labelText: 'University name',
                                labelStyle: GoogleFonts.googleSans(
                                    color: AxonColors.textTertiary,
                                    fontSize: 11),
                                hintText: 'e.g. Imperial College London',
                                hintStyle: GoogleFonts.googleSans(
                                    color: AxonColors.textTertiary
                                        .withValues(alpha: 0.5),
                                    fontSize: 12),
                                border: InputBorder.none))),
                  ),
                  const SizedBox(height: 12),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 14, vertical: 2),
                        decoration: BoxDecoration(
                            color: AxonColors.surfaceElevated,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: AxonColors.divider)),
                        child: TextField(
                            controller: courseCtrl,
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary, fontSize: 13),
                            decoration: InputDecoration(
                                labelText: 'Degree / Course',
                                labelStyle: GoogleFonts.googleSans(
                                    color: AxonColors.textTertiary,
                                    fontSize: 11),
                                hintText: 'e.g. Computer Science',
                                hintStyle: GoogleFonts.googleSans(
                                    color: AxonColors.textTertiary
                                        .withValues(alpha: 0.5),
                                    fontSize: 12),
                                border: InputBorder.none))),
                  ),
                  const SizedBox(height: 12),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 14, vertical: 2),
                        decoration: BoxDecoration(
                            color: AxonColors.surfaceElevated,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: AxonColors.divider)),
                        child: TextField(
                            controller: countryCtrl,
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary, fontSize: 13),
                            decoration: InputDecoration(
                                labelText: 'Country',
                                labelStyle: GoogleFonts.googleSans(
                                    color: AxonColors.textTertiary,
                                    fontSize: 11),
                                hintText: 'e.g. UK',
                                hintStyle: GoogleFonts.googleSans(
                                    color: AxonColors.textTertiary
                                        .withValues(alpha: 0.5),
                                    fontSize: 12),
                                border: InputBorder.none))),
                  ),
                  const SizedBox(height: 16),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Row(
                        children: ['Dream', 'Reach', 'Safety'].map((b) {
                      final sel = selectedBucket == b;
                      return Expanded(
                          child: GestureDetector(
                        onTap: () => setState(() => selectedBucket = b),
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 10),
                          margin: EdgeInsets.only(right: b != 'Safety' ? 8 : 0),
                          decoration: BoxDecoration(
                            color: sel
                                ? _bucketColor(b).withValues(alpha: 0.2)
                                : Colors.transparent,
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(
                                color:
                                    sel ? _bucketColor(b) : AxonColors.divider),
                          ),
                          child: Text(b,
                              textAlign: TextAlign.center,
                              style: GoogleFonts.googleSans(
                                  color: sel
                                      ? _bucketColor(b)
                                      : AxonColors.textTertiary,
                                  fontSize: 11,
                                  fontWeight: FontWeight.w700)),
                        ),
                      ));
                    }).toList()),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: EdgeInsets.fromLTRB(
                        20, 12, 20, MediaQuery.of(ctx).padding.bottom + 16),
                    decoration: BoxDecoration(
                        border: Border(
                            top: BorderSide(
                                color: AxonColors.divider, width: 0.5))),
                    child: Row(children: [
                      Expanded(
                          child: GestureDetector(
                        onTap: () => Navigator.pop(ctx),
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                              color: AxonColors.surfaceElevated,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: AxonColors.divider)),
                          child: Center(
                              child: Text('CANCEL',
                                  style: GoogleFonts.googleSans(
                                      color: AxonColors.textTertiary,
                                      fontSize: 12,
                                      letterSpacing: 1,
                                      fontWeight: FontWeight.w600))),
                        ),
                      )),
                      const SizedBox(width: 12),
                      Expanded(
                          child: GestureDetector(
                        onTap: () async {
                          if (nameCtrl.text.trim().isEmpty) return;
                          await service.createManualTarget(
                            uid: uid,
                            universityName: nameCtrl.text.trim(),
                            courseName: courseCtrl.text.trim().isEmpty
                                ? 'General Degree'
                                : courseCtrl.text.trim(),
                            country: countryCtrl.text.trim().isEmpty
                                ? 'International'
                                : countryCtrl.text.trim(),
                            classification: selectedBucket.toLowerCase(),
                            readinessScore:
                                metrics.predictedPerformance.clamp(0.0, 1.0),
                          );
                          if (ctx.mounted) {
                            Navigator.pop(ctx);
                            AlertService.showSuccess(
                                context,
                                'University added',
                                '${nameCtrl.text.trim()} was saved to $selectedBucket.');
                          }
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          decoration: BoxDecoration(
                              color: AxonColors.accent,
                              borderRadius: BorderRadius.circular(12)),
                          child: Center(
                              child: Text('SAVE',
                                  style: GoogleFonts.googleSans(
                                      color: AxonColors.textPrimary,
                                      fontSize: 12,
                                      letterSpacing: 1,
                                      fontWeight: FontWeight.w600))),
                        ),
                      )),
                    ]),
                  ),
                ],
              );
            },
          ),
        );
      },
    );
  }

  Color _bucketColor(String bucket) {
    final l = bucket.toLowerCase();
    if (l == 'dream') return AxonColors.accent;
    if (l == 'reach') return AxonColors.warning;
    if (l == 'safety') return AxonColors.success;
    return AxonColors.textTertiary;
  }

  Color _getFitBandColor(String fitBand) {
    final lower = fitBand.toLowerCase();
    if (lower.contains('reach')) {
      return AxonColors.warning;
    }
    if (lower.contains('target') || lower.contains('match')) {
      return AxonColors.success;
    }
    if (lower.contains('safety')) {
      return AxonColors.accent;
    }
    return AxonColors.textTertiary;
  }

  Color _getStatusColor(String status) {
    final lower = status.toLowerCase();
    if (lower.contains('submitted') || lower.contains('complete')) {
      return AxonColors.success;
    }
    if (lower.contains('rejected')) {
      return AxonColors.error;
    }
    if (lower.contains('waitlist')) {
      return AxonColors.warning;
    }
    return AxonColors.textSecondary;
  }

  IconData _getAssetIcon(String assetType) {
    final lower = assetType.toLowerCase();
    if (lower.contains('essay')) return Icons.article_rounded;
    if (lower.contains('project')) return Icons.folder_rounded;
    if (lower.contains('certificate')) return Icons.workspace_premium_rounded;
    if (lower.contains('video')) return Icons.videocam_rounded;
    if (lower.contains('research')) return Icons.science_rounded;
    return Icons.work_outline_rounded;
  }

  bool _isDeadlineNear(DateTime date) {
    return date.difference(DateTime.now()).inDays <= 30;
  }

  String _formatDeadline(DateTime date) {
    final days = date.difference(DateTime.now()).inDays;
    if (days < 0) return 'Passed';
    if (days == 0) return 'Today';
    if (days == 1) return 'Tomorrow';
    if (days <= 7) return '${days}d left';
    return '${date.day}/${date.month}/${date.year}';
  }

  String _formatDate(DateTime date) => '${date.day}/${date.month}/${date.year}';
}

// ─────────────────────────────────────────────────────────────────
// Generated Fit Result Model
// ─────────────────────────────────────────────────────────────────
class GeneratedFitResult {
  final String name;
  final String country;
  final String rationale;
  final String entryRequirements;
  final String bucket;
  bool isSaved = false;
  GeneratedFitResult(
      {required this.name,
      required this.country,
      required this.rationale,
      required this.entryRequirements,
      required this.bucket});
}

// ─────────────────────────────────────────────────────────────────
// University Discovery Panel
// ─────────────────────────────────────────────────────────────────
class _UniversityDiscoveryPanel extends StatefulWidget {
  final AdmissionsService service;
  final String uid;
  final String board;
  final List<String> subjects;
  final double readinessScore;

  const _UniversityDiscoveryPanel({
    required this.service,
    required this.uid,
    required this.board,
    required this.subjects,
    required this.readinessScore,
  });

  @override
  State<_UniversityDiscoveryPanel> createState() =>
      _UniversityDiscoveryPanelState();
}

class _UniversityDiscoveryPanelState extends State<_UniversityDiscoveryPanel> {
  final FocusNode _searchFocus = FocusNode();
  final TextEditingController _searchCtrl = TextEditingController();
  String _location = '';
  String _degree = '';
  bool _showResults = false;

  @override
  void initState() {
    super.initState();
    _searchFocus.addListener(() {
      setState(() => _showResults = _searchFocus.hasFocus ||
          _searchCtrl.text.isNotEmpty ||
          _location.isNotEmpty ||
          _degree.isNotEmpty);
    });
  }

  @override
  void dispose() {
    _searchFocus.dispose();
    _searchCtrl.dispose();
    super.dispose();
  }

  List<UniversityProgram> get _results {
    return widget.service.searchPrograms(
      location: _location,
      degree: _degree,
      universityQuery: _searchCtrl.text,
    );
  }

  @override
  Widget build(BuildContext context) {
    final programs = _results;

    return _GlassPanel(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(Icons.travel_explore_rounded,
              color: AxonColors.accent, size: 20),
          const SizedBox(width: 10),
          Expanded(
              child: Text('University Discovery',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w700))),
        ]),
        const SizedBox(height: 16),
        // Search field
        Container(
          decoration: BoxDecoration(
            color: AxonColors.surfaceHighlight.withValues(alpha: 0.6),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
                color: _searchFocus.hasFocus
                    ? AxonColors.accent
                    : AxonColors.divider.withValues(alpha: 0.5)),
          ),
          child: TextField(
            controller: _searchCtrl,
            focusNode: _searchFocus,
            onChanged: (_) => setState(() {}),
            style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary, fontSize: 14),
            decoration: InputDecoration(
              prefixIcon: Icon(Icons.search_rounded,
                  color: AxonColors.textTertiary, size: 20),
              suffixIcon: _searchCtrl.text.isNotEmpty
                  ? GestureDetector(
                      onTap: () {
                        _searchCtrl.clear();
                        setState(() {});
                      },
                      child: Icon(Icons.close_rounded,
                          color: AxonColors.textTertiary, size: 18))
                  : null,
              hintText: 'Search universities...',
              hintStyle: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary, fontSize: 14),
              border: InputBorder.none,
              contentPadding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
            ),
          ),
        ),
        const SizedBox(height: 12),
        // Location & Degree row
        Row(children: [
          Expanded(
              child: _buildDropdown(
                  'Location',
                  _location,
                  widget.service.availableLocations,
                  (v) => setState(() => _location = v))),
          const SizedBox(width: 10),
          Expanded(
              child: _buildDropdown(
                  'Degree',
                  _degree,
                  widget.service.availableDegrees,
                  (v) => setState(() => _degree = v))),
        ]),
        // Active filter pills
        if (_location.isNotEmpty || _degree.isNotEmpty)
          Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Row(children: [
                if (_location.isNotEmpty)
                  _filterPill(_location, () => setState(() => _location = '')),
                if (_degree.isNotEmpty) ...[
                  const SizedBox(width: 6),
                  _filterPill(_degree, () => setState(() => _degree = ''))
                ],
              ])),
        const SizedBox(height: 8),
        // Results
        if (!_showResults &&
            _searchCtrl.text.isEmpty &&
            _location.isEmpty &&
            _degree.isEmpty)
          Text('Search a university or choose a filter above to begin.',
              style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary, fontSize: 12))
        else ...[
          if (programs.isEmpty)
            Text('No programs match this filter.',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary, fontSize: 12))
          else ...[
            Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                    '${programs.length} program${programs.length == 1 ? '' : 's'} found',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 11,
                        fontWeight: FontWeight.w500))),
            for (final program in programs.take(6))
              _UniversityProgramCard(
                program: program,
                board: widget.board,
                subjects: widget.subjects,
                onSave: (bucket) async {
                  await widget.service.saveProgramTarget(
                    uid: widget.uid,
                    program: program,
                    bucket: bucket,
                    board: widget.board,
                    subjects: widget.subjects,
                    readinessScore: widget.readinessScore,
                  );
                  if (!context.mounted) return;
                  AlertService.showSuccess(context, 'Program saved',
                      '${program.universityName} was added to $bucket.');
                },
              ),
            if (programs.length > 6)
              Center(
                  child: Text('+ ${programs.length - 6} more',
                      style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary, fontSize: 11))),
          ],
        ],
      ]),
    );
  }

  Widget _buildDropdown(String label, String value, List<String> items,
      ValueChanged<String> onChanged) {
    return Container(
      height: 44,
      decoration: BoxDecoration(
        color: AxonColors.surfaceHighlight.withValues(alpha: 0.6),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AxonColors.divider.withValues(alpha: 0.5)),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value.isEmpty ? null : value,
          isExpanded: true,
          dropdownColor: AxonColors.surfaceElevated,
          borderRadius: BorderRadius.circular(12),
          padding: const EdgeInsets.symmetric(horizontal: 12),
          hint: Text(label,
              style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary, fontSize: 13)),
          items: items
              .map((item) => DropdownMenuItem(
                  value: item,
                  child: Text(item,
                      style: GoogleFonts.googleSans(
                          color: AxonColors.textPrimary, fontSize: 13))))
              .toList(),
          onChanged: (next) => onChanged(next ?? ''),
        ),
      ),
    );
  }

  Widget _filterPill(String label, VoidCallback onRemove) {
    return GestureDetector(
      onTap: onRemove,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: AxonColors.accent.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: AxonColors.accent.withValues(alpha: 0.25)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Text(label,
              style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 12,
                  fontWeight: FontWeight.w500)),
          const SizedBox(width: 4),
          Icon(Icons.close_rounded, color: AxonColors.accent, size: 14),
        ]),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// University Program Card (Redesigned)
// ─────────────────────────────────────────────────────────────────
class _UniversityProgramCard extends StatefulWidget {
  final UniversityProgram program;
  final String board;
  final List<String> subjects;
  final ValueChanged<String> onSave;

  const _UniversityProgramCard({
    required this.program,
    required this.board,
    required this.subjects,
    required this.onSave,
  });

  @override
  State<_UniversityProgramCard> createState() => _UniversityProgramCardState();
}

class _UniversityProgramCardState extends State<_UniversityProgramCard> {
  bool _showingBuckets = false;

  static const _universityGradients = <Color>[
    Color(0xFF3A86FF),
    Color(0xFF7C3AED),
    Color(0xFF059669),
    Color(0xFFEA580C),
    Color(0xFFBE123C),
    Color(0xFF0369A1),
    Color(0xFF0D9488),
    Color(0xFFB45309),
  ];

  Color get _imageColor {
    final hash = widget.program.universityName.hashCode;
    return _universityGradients[hash.abs() % _universityGradients.length];
  }

  String get _initials {
    final parts = widget.program.universityName.split(' ');
    if (parts.length >= 2) return '${parts[0][0]}${parts[1][0]}'.toUpperCase();
    return widget.program.universityName.substring(0, 2).toUpperCase();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _showDetailSheet(context),
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        decoration: BoxDecoration(
            color: AxonColors.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AxonColors.divider, width: 1)),
        clipBehavior: Clip.antiAlias,
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Stack(children: [
            Container(
              height: 100,
              width: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                    colors: [_imageColor, _imageColor.withValues(alpha: 0.6)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight),
              ),
              child: Center(
                  child: Text(_initials,
                      style: GoogleFonts.googleSans(
                          color: Colors.white.withValues(alpha: 0.9),
                          fontSize: 32,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 2))),
            ),
            Positioned(
              top: 8,
              right: 8,
              child: GestureDetector(
                onTap: () => setState(() => _showingBuckets = !_showingBuckets),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 30,
                  height: 30,
                  decoration: BoxDecoration(
                    color: _showingBuckets
                        ? AxonColors.accent
                        : AxonColors.surfaceElevated.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                      _showingBuckets ? Icons.close_rounded : Icons.add_rounded,
                      color: _showingBuckets ? AxonColors.textPrimary : AxonColors.textSecondary,
                      size: 18),
                ),
              ),
            ),
          ]),
          Padding(
            padding: const EdgeInsets.all(12),
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(widget.program.universityName,
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 14,
                      fontWeight: FontWeight.w700),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis),
              const SizedBox(height: 3),
              Text(widget.program.courseName,
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textSecondary, fontSize: 11),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis),
              const SizedBox(height: 4),
              Row(children: [
                Icon(Icons.location_on_outlined,
                    color: AxonColors.textTertiary, size: 11),
                const SizedBox(width: 3),
                Expanded(
                    child: Text(widget.program.location,
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textTertiary, fontSize: 10),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis)),
              ]),
              if (_showingBuckets) ...[
                const SizedBox(height: 10),
                Row(
                    children: ['Dream', 'Reach', 'Safety'].map((b) {
                  return Expanded(
                      child: GestureDetector(
                    onTap: () {
                      widget.onSave(b);
                      setState(() => _showingBuckets = false);
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 7),
                      margin: EdgeInsets.only(right: b != 'Safety' ? 6 : 0),
                      decoration: BoxDecoration(
                        color: _bucketColor2(b).withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                            color: _bucketColor2(b).withValues(alpha: 0.3)),
                      ),
                      child: Text(b,
                          textAlign: TextAlign.center,
                          style: GoogleFonts.googleSans(
                              color: _bucketColor2(b),
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5)),
                    ),
                  ));
                }).toList()),
              ],
            ]),
          ),
        ]),
      ),
    );
  }

  Color _bucketColor2(String b) {
    if (b == 'Dream') return AxonColors.accent;
    if (b == 'Reach') return AxonColors.warning;
    return AxonColors.success;
  }

  void _showDetailSheet(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) {
        final req = widget.program.requirementSummaryFor(
            board: widget.board, subjects: widget.subjects);
        return Container(
          margin: EdgeInsets.symmetric(
              horizontal: screenWidth > 600 ? screenWidth * 0.15 : 0),
          decoration: BoxDecoration(
            color: AxonColors.background,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          ),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Container(
              height: 130,
              width: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                    colors: [_imageColor, _imageColor.withValues(alpha: 0.4)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight),
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(24)),
              ),
              child: Stack(children: [
                Center(
                    child: Text(_initials,
                        style: GoogleFonts.googleSans(
                            color: Colors.white.withValues(alpha: 0.85),
                            fontSize: 42,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 3))),
                Positioned(
                    top: 12,
                    left: 12,
                    child: GestureDetector(
                      onTap: () => Navigator.pop(ctx),
                      child: Container(
                          width: 34,
                          height: 34,
                          decoration: BoxDecoration(
                              color: AxonColors.surfaceElevated.withValues(alpha: 0.8),
                              borderRadius: BorderRadius.circular(10)),
                          child: Icon(Icons.arrow_back_rounded,
                              color: AxonColors.textPrimary, size: 18)),
                    )),
              ]),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(widget.program.universityName,
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary,
                            fontSize: 18,
                            fontWeight: FontWeight.w700)),
                    const SizedBox(height: 4),
                    Row(children: [
                      Icon(Icons.school_rounded,
                          color: AxonColors.textSecondary, size: 13),
                      const SizedBox(width: 5),
                      Expanded(
                          child: Text(widget.program.courseName,
                              style: GoogleFonts.googleSans(
                                  color: AxonColors.textSecondary,
                                  fontSize: 12))),
                    ]),
                    const SizedBox(height: 4),
                    Row(children: [
                      Icon(Icons.location_on_outlined,
                          color: AxonColors.textTertiary, size: 13),
                      const SizedBox(width: 5),
                      Text(widget.program.location,
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary, fontSize: 12)),
                      const SizedBox(width: 10),
                      Icon(Icons.schedule_rounded,
                          color: AxonColors.textTertiary, size: 13),
                      const SizedBox(width: 3),
                      Text(widget.program.duration,
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textTertiary, fontSize: 12)),
                    ]),
                    const SizedBox(height: 12),
                    Text(req,
                        style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary,
                            fontSize: 11,
                            height: 1.4)),
                    if (widget.program.coreModules.isNotEmpty) ...[
                      const SizedBox(height: 10),
                      Text('CORE MODULES',
                          style: GoogleFonts.googleSans(
                              color: AxonColors.accent,
                              fontSize: 8,
                              letterSpacing: 1.5,
                              fontWeight: FontWeight.w600)),
                      const SizedBox(height: 6),
                      Wrap(
                          spacing: 6,
                          runSpacing: 6,
                          children: widget.program.coreModules
                              .take(5)
                              .map(
                                  (m) => smallPill2(m, AxonColors.textTertiary))
                              .toList()),
                    ],
                    const SizedBox(height: 24),
                  ]),
            ),
            Container(
              padding: EdgeInsets.fromLTRB(
                  20, 12, 20, MediaQuery.of(ctx).padding.bottom + 16),
              decoration: BoxDecoration(
                  border: Border(
                      top: BorderSide(color: AxonColors.divider, width: 0.5))),
              child: Row(
                  children: ['Dream', 'Reach', 'Safety'].map((b) {
                return Expanded(
                    child: GestureDetector(
                  onTap: () {
                    widget.onSave(b);
                    Navigator.pop(ctx);
                  },
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 11),
                    margin: EdgeInsets.only(right: b != 'Safety' ? 8 : 0),
                    decoration: BoxDecoration(
                      color: _bucketColor2(b).withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                          color: _bucketColor2(b).withValues(alpha: 0.35)),
                    ),
                    child: Text(b,
                        textAlign: TextAlign.center,
                        style: GoogleFonts.googleSans(
                            color: _bucketColor2(b),
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 0.5)),
                  ),
                ));
              }).toList()),
            ),
          ]),
        );
      },
    );
  }
}

Widget smallPill2(String label, Color color) {
  return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
          color: color.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(6)),
      child: Text(label,
          style: GoogleFonts.googleSans(
              color: color, fontSize: 9, fontWeight: FontWeight.w500)));
}

// ─────────────────────────────────────────────────────────────────
// Marks Dashboard
// ─────────────────────────────────────────────────────────────────
class _MarksDashboard extends StatefulWidget {
  final AdmissionsService service;
  final String uid;
  final List<AdmissionsTarget> targets;
  final List<AdmissionsMarkEntry> marks;

  const _MarksDashboard(
      {required this.service,
      required this.uid,
      required this.targets,
      required this.marks});

  @override
  State<_MarksDashboard> createState() => _MarksDashboardState();
}

class _MarksDashboardState extends State<_MarksDashboard> {
  final _subjectCtrl = TextEditingController();
  final _markCtrl = TextEditingController();
  final _gradeCtrl = TextEditingController();
  final _targetCtrl = TextEditingController();

  @override
  void dispose() {
    _subjectCtrl.dispose();
    _markCtrl.dispose();
    _gradeCtrl.dispose();
    _targetCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return _GlassPanel(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      for (final target in widget.targets
          .where(
              (t) => t.classification == 'dream' || t.classification == 'reach')
          .take(3))
        Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Text(
                '${target.fitBand}: ${target.universityName} requires ${target.entryRequirements}',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary,
                    fontSize: 12,
                    height: 1.35))),
      if (widget.marks.isEmpty)
        Text('Log your current marks to compare them against thresholds.',
            style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary, fontSize: 12))
      else
        for (final mark in widget.marks) _MarkRow(mark: mark),
      const SizedBox(height: 14),
      Row(children: [
        Expanded(child: _compactField(_subjectCtrl, 'Subject')),
        const SizedBox(width: 8),
        Expanded(child: _compactField(_markCtrl, 'Mark %')),
      ]),
      const SizedBox(height: 8),
      Row(children: [
        Expanded(child: _compactField(_gradeCtrl, 'Grade')),
        const SizedBox(width: 8),
        Expanded(child: _compactField(_targetCtrl, 'Target')),
        IconButton(
            onPressed: _save,
            icon: Icon(Icons.add_circle_rounded, color: AxonColors.accent)),
      ]),
    ]));
  }

  Widget _compactField(TextEditingController c, String hint) {
    return TextField(
        controller: c,
        style:
            GoogleFonts.googleSans(color: AxonColors.textPrimary, fontSize: 12),
        decoration: InputDecoration(
            hintText: hint,
            hintStyle: GoogleFonts.googleSans(color: AxonColors.textTertiary),
            isDense: true,
            filled: true,
            fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.6),
            border:
                OutlineInputBorder(borderRadius: BorderRadius.circular(12))));
  }

  Future<void> _save() async {
    if (_subjectCtrl.text.trim().isEmpty) return;
    await widget.service.saveMarkEntry(
        uid: widget.uid,
        subject: _subjectCtrl.text,
        currentMark: double.tryParse(_markCtrl.text) ?? 0,
        currentGrade: _gradeCtrl.text,
        targetGrade: _targetCtrl.text);
    _subjectCtrl.clear();
    _markCtrl.clear();
    _gradeCtrl.clear();
    _targetCtrl.clear();
  }
}

// ─────────────────────────────────────────────────────────────────
// Achievement Vault Panel
// ─────────────────────────────────────────────────────────────────
class _AchievementVaultPanel extends StatelessWidget {
  final AdmissionsService service;
  final String uid;
  final List<AdmissionsVaultAsset> assets;
  final List<AdmissionsTarget> targets;

  const _AchievementVaultPanel(
      {required this.service,
      required this.uid,
      required this.assets,
      required this.targets});

  Future<void> _pickAchievement(BuildContext context) async {
    try {
      final result = await FilePicker.platform.pickFiles(
          type: FileType.custom,
          allowedExtensions: ['pdf', 'png', 'jpg', 'jpeg', 'webp', 'mp4']);
      if (result == null || result.files.isEmpty) return;
      final file = result.files.single;
      final path = file.path;
      if (path == null) return;
      await service.uploadAchievementAsset(
        uid: uid,
        file: File(path),
        title: file.name,
        targetDegree:
            targets.isNotEmpty ? targets.first.courseName : 'Computer Science',
        extractedText: file.name,
        linkedTargetIds: targets.take(3).map((t) => t.id).toList(),
      );
      if (context.mounted) {
        AlertService.showSuccess(
            context, 'Achievement uploaded', 'Saved to your vault.');
      }
    } catch (e) {
      if (context.mounted) {
        AlertService.showWarning(
            context,
            'Upload failed',
            e.toString().length > 100
                ? e.toString().substring(0, 100)
                : e.toString());
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return _GlassPanel(
        child: Column(children: [
      GestureDetector(
        onTap: () => _pickAchievement(context),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
              color: AxonColors.accent.withValues(alpha: 0.16),
              borderRadius: BorderRadius.circular(14),
              border:
                  Border.all(color: AxonColors.accent.withValues(alpha: 0.28))),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(Icons.upload_file_rounded, color: AxonColors.accent, size: 18),
            const SizedBox(width: 8),
            Text('Upload achievement proof',
                style: GoogleFonts.googleSans(
                    color: AxonColors.accent,
                    fontWeight: FontWeight.w700,
                    fontSize: 12)),
          ]),
        ),
      ),
      const SizedBox(height: 12),
      if (assets.isEmpty)
        Text('Certificates, portfolios, and extracurricular proof appear here.',
            style: GoogleFonts.googleSans(
                color: AxonColors.textTertiary, fontSize: 12))
      else
        for (final asset in assets.take(5)) _buildAssetCard2(asset),
      if (assets.length > 5)
        Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text('+${assets.length - 5} more',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary, fontSize: 10))),
    ]));
  }

  Widget _buildAssetCard2(AdmissionsVaultAsset asset) {
    return Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
            color: AxonColors.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AxonColors.divider, width: 1)),
        child: Row(children: [
          Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10)),
              child: Icon(_getAssetIcon(asset.assetType),
                  color: AxonColors.accent, size: 20)),
          const SizedBox(width: 12),
          Expanded(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Text(asset.title,
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 13,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Wrap(spacing: 6, children: [
                  smallPill2(asset.assetType,
                      AxonColors.textTertiary.withValues(alpha: 0.3)),
                  if (asset.provider.isNotEmpty)
                    smallPill2(asset.provider,
                        AxonColors.accent.withValues(alpha: 0.3)),
                ]),
              ])),
          Icon(Icons.chevron_right_rounded,
              color: AxonColors.textTertiary, size: 20),
        ]));
  }

  IconData _getAssetIcon(String assetType) {
    final l = assetType.toLowerCase();
    if (l.contains('essay')) return Icons.article_rounded;
    if (l.contains('project')) return Icons.folder_rounded;
    if (l.contains('certificate')) return Icons.workspace_premium_rounded;
    if (l.contains('video')) return Icons.videocam_rounded;
    if (l.contains('research')) return Icons.science_rounded;
    return Icons.work_outline_rounded;
  }
}

// ─────────────────────────────────────────────────────────────────
// Shared Widgets
// ─────────────────────────────────────────────────────────────────
class _GlassPanel extends StatelessWidget {
  final Widget child;
  const _GlassPanel({required this.child});

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(22),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 24, sigmaY: 24),
        child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
                color: AxonColors.surface,
                borderRadius: BorderRadius.circular(22),
                border: Border.all(color: AxonColors.divider, width: 0.5)),
            child: child),
      ),
    );
  }
}

class FilterChipMenu extends StatelessWidget {
  final String label, value;
  final List<String> values;
  final ValueChanged<String> onChanged;
  const FilterChipMenu(
      {super.key,
      required this.label,
      required this.value,
      required this.values,
      required this.onChanged});

  @override
  Widget build(BuildContext context) {
    final items = values
        .map((v) => DropdownMenuItem(
            value: v,
            child: Text(v,
                style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary, fontSize: 12))))
        .toList();
    items.insert(
        0,
        DropdownMenuItem(
            value: '',
            child: Text('All',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary, fontSize: 12))));
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      decoration: BoxDecoration(
          color: AxonColors.surfaceHighlight.withValues(alpha: 0.6),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AxonColors.divider)),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
            value: value,
            isDense: true,
            dropdownColor: AxonColors.surface,
            hint: Text(label,
                style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary, fontSize: 11)),
            items: items,
            onChanged: (v) {
              if (v != null) onChanged(v);
            }),
      ),
    );
  }
}

class ModePill extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const ModePill(
      {super.key,
      required this.label,
      required this.selected,
      required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
              color: AxonColors.accent.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(20),
              border:
                  Border.all(color: AxonColors.accent.withValues(alpha: 0.3))),
          child: Text(label,
              style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5))),
    );
  }
}

class SmallPill extends StatelessWidget {
  final String label;
  final Color color;
  const SmallPill(this.label, this.color, {super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
        decoration: BoxDecoration(
            color: color.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(6)),
        child: Text(label,
            style: GoogleFonts.googleSans(
                color: color, fontSize: 9, fontWeight: FontWeight.w500)));
  }
}

class _TrajectoryBanner extends StatelessWidget {
  final MetricsState metrics;
  const _TrajectoryBanner({required this.metrics});

  @override
  Widget build(BuildContext context) {
    final score = metrics.predictedPerformance.clamp(0.0, 1.0);
    final color = score >= 0.7
        ? AxonColors.success
        : score >= 0.4
            ? AxonColors.warning
            : AxonColors.error;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: AxonColors.divider, width: 1)),
      child: Row(children: [
        Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
                color: color.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(16)),
            child: Icon(Icons.insights_rounded, color: color, size: 24)),
        const SizedBox(width: 14),
        Expanded(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('TRAJECTORY',
              style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary, fontSize: 9, letterSpacing: 1)),
          const SizedBox(height: 4),
          Text('${(score * 100).round()}% Readiness',
              style: GoogleFonts.googleSans(
                  color: AxonColors.textPrimary,
                  fontSize: 13,
                  fontWeight: FontWeight.w600)),
        ])),
        Icon(Icons.insights_rounded, color: color, size: 24),
      ]),
    );
  }
}

class _MarkRow extends StatelessWidget {
  final AdmissionsMarkEntry mark;
  const _MarkRow({required this.mark});

  @override
  Widget build(BuildContext context) {
    return Padding(
        padding: const EdgeInsets.only(bottom: 8),
        child: Row(children: [
          Expanded(
              child: Text(mark.subject,
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 12,
                      fontWeight: FontWeight.w600))),
          const SizedBox(width: 8),
          Text('${mark.currentMark.toStringAsFixed(0)}%',
              style: GoogleFonts.googleSans(
                  color: AxonColors.textSecondary, fontSize: 12)),
          const SizedBox(width: 8),
          Text(mark.currentGrade,
              style: GoogleFonts.googleSans(
                  color: AxonColors.textTertiary, fontSize: 12)),
          const SizedBox(width: 8),
          Text('→ ${mark.targetGrade}',
              style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 12,
                  fontWeight: FontWeight.w600)),
        ]));
  }
}

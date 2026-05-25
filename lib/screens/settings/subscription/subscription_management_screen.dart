// lib/screens/settings/subscription/subscription_management_screen.dart
// ─────────────────────────────────────────────────────────────────
// Subscription Management Screen
// Shows current plan status, usage limits, upgrade/downgrade
// ─────────────────────────────────────────────────────────────────
import 'package:google_fonts/google_fonts.dart';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../../../theme/app_theme.dart';
import '../../../services/subscription_service.dart';
import '../../../utils/nav_utils.dart';

class SubscriptionManagementScreen extends ConsumerWidget {
  const SubscriptionManagementScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final svc = SubscriptionService();
    final sub = svc.current;

    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      extendBody: true,
      appBar: AppBar(
        backgroundColor: AxonColors.oxfordBlueDark,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => popOrGo(context, '/settings/subscription'),
        ),
        title: Text(
          'My Subscription',
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
        children: [
          _buildCurrentPlanCard(sub),
          const SizedBox(height: 16),
          if (sub.hasAccess) ...[
            _buildUsageSection(sub),
            const SizedBox(height: 16),
            _buildBillingSection(sub),
            const SizedBox(height: 16),
          ],
          _buildUpgradeSection(context, sub),
          const SizedBox(height: 16),
          _buildHelpSection(context),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _buildCurrentPlanCard(SubscriptionState sub) {
    final statusColor = sub.status.isActive
        ? const Color(0xFF10B981)
        : sub.status.isPastDue
            ? Colors.orange
            : Colors.red;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            const Color(0xFF1E3A5F),
            AxonColors.oxfordBlueDark,
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border:
            Border.all(color: AxonColors.electricCyan.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: statusColor.withValues(alpha: 0.4)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: statusColor,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      sub.status.name.toUpperCase(),
                      style: TextStyle(
                        color: statusColor,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1,
                      ),
                    ),
                  ],
                ),
              ),
              const Spacer(),
              if (sub.isInTrial)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.purple.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Text(
                    'TRIAL',
                    style: TextStyle(
                      color: Colors.purple,
                      fontSize: 10,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 16),
          Text(
            sub.tier.displayName,
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 28,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            sub.tier.subtitle,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 13,
            ),
          ),
          if (sub.currentPeriodEnd != null) ...[
            const SizedBox(height: 12),
            Row(
              children: [
                Icon(
                  Icons.calendar_today,
                  color: Colors.white.withValues(alpha: 0.4),
                  size: 14,
                ),
                const SizedBox(width: 6),
                Text(
                  sub.isYearlyPlan
                      ? 'Renews yearly · ${sub.daysRemaining} days left'
                      : 'Renews monthly · ${sub.daysRemaining} days left',
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.4),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildUsageSection(SubscriptionState sub) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Usage This Period',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 16),
          if (sub.offlinePacksLimit != null) ...[
            _buildUsageRow(
              'Offline Packs',
              sub.offlinePacksUsed ?? 0,
              sub.offlinePacksLimit!,
              'packs',
            ),
            const SizedBox(height: 12),
          ],
          if (sub.aiQueriesLimit != null && sub.aiQueriesLimit! > 0) ...[
            _buildUsageRow(
              'AI Queries',
              sub.aiQueriesUsed ?? 0,
              sub.aiQueriesLimit!,
              'queries',
            ),
            const SizedBox(height: 12),
          ],
          if (sub.flashcardsLimit != null && sub.flashcardsLimit! > 0) ...[
            _buildUsageRow(
              'AI Flashcards',
              sub.flashcardsGenerated ?? 0,
              sub.flashcardsLimit!,
              'cards',
            ),
            const SizedBox(height: 12),
          ],
          _buildUsageRow(
            'Streak Shields',
            0,
            sub.streakShieldsRemaining,
            'shields',
            color: const Color(0xFF10B981),
          ),
        ],
      ),
    );
  }

  Widget _buildUsageRow(String label, int used, int limit, String unit,
      {Color? color}) {
    final effectiveLimit = limit < 0 ? 999 : limit;
    final progress =
        effectiveLimit > 0 ? (used / effectiveLimit).clamp(0.0, 1.0) : 0.0;
    final remaining =
        limit < 0 ? 'Unlimited' : '${effectiveLimit - used} $unit left';
    final barColor =
        color ?? (progress > 0.8 ? Colors.orange : AxonColors.electricCyan);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              label,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 13,
              ),
            ),
            const Spacer(),
            Text(
              '$used / $effectiveLimit',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.5),
                fontSize: 12,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              remaining,
              style: TextStyle(
                color: barColor,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: progress,
            backgroundColor: Colors.white.withValues(alpha: 0.1),
            valueColor: AlwaysStoppedAnimation(barColor),
            minHeight: 6,
          ),
        ),
      ],
    );
  }

  Widget _buildBillingSection(SubscriptionState sub) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        children: [
          _buildBillingRow(
            sub.isYearlyPlan ? 'Yearly Plan' : 'Monthly Plan',
            sub.isYearlyPlan ? sub.tier.priceYearly : sub.tier.priceMonthly,
          ),
          const SizedBox(height: 8),
          _buildBillingRow(
            'Billing',
            sub.currentPeriodEnd != null
                ? _formatDate(sub.currentPeriodEnd!)
                : '—',
          ),
          const Divider(color: Colors.white12, height: 24),
          GestureDetector(
            onTap: () {},
            child: Row(
              children: [
                Icon(Icons.open_in_new,
                    color: AxonColors.electricCyan, size: 16),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Manage Billing',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.electricCyan,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Icon(Icons.chevron_right,
                    color: Colors.white.withValues(alpha: 0.3), size: 20),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBillingRow(String label, String value) {
    return Row(
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.5),
            fontSize: 13,
          ),
        ),
        const Spacer(),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildUpgradeSection(BuildContext context, SubscriptionState sub) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Change Plan',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              if (sub.tier.index < SubscriptionTier.pro.index)
                Expanded(
                  child: ElevatedButton(
                    onPressed: () => context.push('/settings/subscription'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AxonColors.electricCyan,
                      foregroundColor: Colors.black,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: const Text(
                      'Upgrade',
                      style: TextStyle(fontWeight: FontWeight.w700),
                    ),
                  ),
                )
              else
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 12),
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: const Color(0xFF10B981).withValues(alpha: 0.15),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.check_circle,
                            color: Color(0xFF10B981), size: 16),
                        SizedBox(width: 6),
                        Text(
                          'Best Plan Active',
                          style: TextStyle(
                            color: Color(0xFF10B981),
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHelpSection(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Need Help?',
            style: GoogleFonts.googleSans(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 12),
          _buildHelpRow(Icons.email_outlined, 'Contact Support', () {}),
          const SizedBox(height: 8),
          _buildHelpRow(Icons.restore, 'Restore Purchases', () {}),
          const SizedBox(height: 8),
          _buildHelpRow(Icons.article_outlined, 'Terms of Service', () {}),
          const SizedBox(height: 8),
          _buildHelpRow(Icons.privacy_tip_outlined, 'Privacy Policy', () {}),
        ],
      ),
    );
  }

  Widget _buildHelpRow(IconData icon, String label, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Row(
        children: [
          Icon(icon, color: Colors.white.withValues(alpha: 0.5), size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.7),
                fontSize: 13,
              ),
            ),
          ),
          Icon(Icons.chevron_right,
              color: Colors.white.withValues(alpha: 0.2), size: 18),
        ],
      ),
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year}';
  }
}

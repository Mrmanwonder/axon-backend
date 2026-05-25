// lib/screens/challenge/challenge_screen.dart
// ─────────────────────────────────────────────────────────────────
// Peer Challenge & Accountability Screen
// ─────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:intl/intl.dart';
import '../../services/challenge_service.dart';
import '../../services/subscription_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/rose_loader.dart';

class ChallengeScreen extends ConsumerStatefulWidget {
  const ChallengeScreen({super.key});

  @override
  ConsumerState<ChallengeScreen> createState() => _ChallengeScreenState();
}

class _ChallengeScreenState extends ConsumerState<ChallengeScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  final ChallengeService _service = ChallengeService();

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Widget buildPaywallGate(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.emoji_events_outlined,
                size: 64, color: AxonColors.textTertiary),
            const SizedBox(height: 16),
            Text('Peer Challenges',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 20,
                    fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Text(
                'Challenge friends, join study competitions, and find accountability partners with Plus or higher.',
                textAlign: TextAlign.center,
                style: GoogleFonts.googleSans(
                    color: AxonColors.textSecondary, fontSize: 14)),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: () => context.push('/settings/subscription'),
              style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  foregroundColor: Colors.white,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 32, vertical: 14)),
              child: Text('Upgrade',
                  style: GoogleFonts.googleSans(fontWeight: FontWeight.w600)),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final svc = SubscriptionService();
    if (!svc.hasAccess(Feature.peerChallenges)) {
      return buildPaywallGate(context);
    }

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            AxonColors.oxfordBlue.withValues(alpha: 0.95),
            AxonColors.oxfordBlueDark.withValues(alpha: 0.98),
          ],
        ),
      ),
      child: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            _buildTabBar(),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _ActiveChallengesTab(service: _service),
                  _PartnersTab(service: _service),
                  _PublicChallengesTab(service: _service),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
      child: Row(
        children: [
          GestureDetector(
            onTap: () => popOrGo(context, '/home'),
            child: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AxonColors.surface.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(12),
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
                Text(
                  'Challenges',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Compete • Partner Up • Stay Accountable',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          GestureDetector(
            onTap: () => _showCreateChallengeSheet(context),
            child: Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: AxonColors.accent.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(Icons.add, color: AxonColors.accent, size: 20),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTabBar() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: TabBar(
        controller: _tabController,
        indicator: BoxDecoration(
          color: AxonColors.accent.withValues(alpha: 0.2),
          borderRadius: BorderRadius.circular(12),
        ),
        indicatorSize: TabBarIndicatorSize.tab,
        dividerColor: Colors.transparent,
        labelColor: AxonColors.accent,
        unselectedLabelColor: AxonColors.textTertiary,
        labelStyle:
            GoogleFonts.googleSans(fontSize: 11, fontWeight: FontWeight.w600),
        unselectedLabelStyle:
            GoogleFonts.googleSans(fontSize: 11, fontWeight: FontWeight.w500),
        tabs: const [
          Tab(
              icon: Icon(Icons.emoji_events_outlined, size: 18),
              text: 'Active'),
          Tab(icon: Icon(Icons.people_outline, size: 18), text: 'Partners'),
          Tab(icon: Icon(Icons.public_outlined, size: 18), text: 'Discover'),
        ],
      ),
    );
  }

  void _showCreateChallengeSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _CreateChallengeSheet(service: _service),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// ACTIVE CHALLENGES TAB
// ─────────────────────────────────────────────────────────────────

class _ActiveChallengesTab extends ConsumerWidget {
  final ChallengeService service;
  const _ActiveChallengesTab({required this.service});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return FutureBuilder<List<Challenge>>(
      future: service.getActiveChallenges(),
      builder: (context, snapshot) {
        if (!snapshot.hasData && !snapshot.hasError) {
          return const Center(child: RoseLoader(size: 24));
        }
        if (snapshot.hasError) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.error_outline, color: AxonColors.error, size: 40),
                const SizedBox(height: 12),
                Text('Failed to load challenges',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textSecondary, fontSize: 14)),
              ],
            ),
          );
        }

        final challenges = snapshot.data!;

        if (challenges.isEmpty) {
          return _ActiveEmptyState();
        }

        return ListView.builder(
          padding: const EdgeInsets.all(20),
          itemCount: challenges.length,
          itemBuilder: (context, i) => ChallengeCard(
            challenge: challenges[i],
            onUpdate: (value) =>
                service.updateProgress(challenges[i].id, value),
          ),
        );
      },
    );
  }
}

class _ActiveEmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: AxonColors.accent.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.emoji_events_outlined,
                size: 48, color: AxonColors.accent),
          ),
          const SizedBox(height: 24),
          Text(
            'No Active Challenges',
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Create one or join from the Discover tab',
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}

class ChallengeCard extends StatelessWidget {
  final Challenge challenge;
  final Function(int) onUpdate;
  const ChallengeCard(
      {super.key, required this.challenge, required this.onUpdate});

  @override
  Widget build(BuildContext context) {
    final timeLeft = challenge.timeRemaining;
    final hours = timeLeft.inHours;
    final isUrgent = hours < 4;

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isUrgent
              ? Color(0xFFE91E63).withValues(alpha: 0.3)
              : Colors.transparent,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: _getTypeColor(challenge.type).withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  challenge.subject,
                  style: GoogleFonts.googleSans(
                    color: _getTypeColor(challenge.type),
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: isUrgent
                      ? Color(0xFFE91E63).withValues(alpha: 0.2)
                      : AxonColors.accent.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.timer_outlined,
                      size: 12,
                      color: isUrgent ? Color(0xFFE91E63) : AxonColors.accent,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      hours > 24
                          ? '${(hours / 24).floor()}d'
                          : hours > 0
                              ? '${hours}h'
                              : '${timeLeft.inMinutes}m',
                      style: GoogleFonts.googleSans(
                        color: isUrgent ? Color(0xFFE91E63) : AxonColors.accent,
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            challenge.title,
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 16,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            challenge.description,
            style: GoogleFonts.googleSans(
              color: AxonColors.textSecondary,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          '${challenge.currentValue}/${challenge.targetValue}',
                          style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        Text(
                          '${(challenge.progress * 100).round()}%',
                          style: GoogleFonts.googleSans(
                            color: AxonColors.accent,
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
                        value: challenge.progress,
                        minHeight: 6,
                        backgroundColor:
                            AxonColors.surface.withValues(alpha: 0.2),
                        color: AxonColors.accent,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 12),
              GestureDetector(
                onTap: () => _showAddProgressDialog(context),
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AxonColors.accent.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(Icons.add, color: AxonColors.accent, size: 20),
                ),
              ),
            ],
          ),
          if (challenge.participants.isNotEmpty) ...[
            const SizedBox(height: 12),
            Row(
              children: [
                Icon(Icons.people_outline,
                    size: 14, color: AxonColors.textTertiary),
                const SizedBox(width: 6),
                Text(
                  '${challenge.participants.length + 1} participants',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
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

  Color _getTypeColor(ChallengeType type) {
    switch (type) {
      case ChallengeType.questionCount:
        return Color(0xFF4CAF50);
      case ChallengeType.timeSpent:
        return Color(0xFF2196F3);
      case ChallengeType.chapterComplete:
        return Color(0xFFFF9800);
      case ChallengeType.mockScore:
        return Color(0xFFE91E63);
      case ChallengeType.streakMaintain:
        return Color(0xFF9C27B0);
      case ChallengeType.custom:
        return AxonColors.accent;
    }
  }

  void _showAddProgressDialog(BuildContext context) {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: AxonColors.surface,
        title: Text('Add Progress',
            style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary, fontWeight: FontWeight.w700)),
        content: TextField(
          controller: controller,
          keyboardType: TextInputType.number,
          decoration: const InputDecoration(
            hintText: 'Enter value',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final value = int.tryParse(controller.text);
              if (value != null && value > 0) {
                onUpdate(value);
                Navigator.pop(dialogContext);
              }
            },
            child: const Text('Add'),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// PARTNERS TAB
// ─────────────────────────────────────────────────────────────────

class _PartnersTab extends ConsumerStatefulWidget {
  final ChallengeService service;
  const _PartnersTab({required this.service});

  @override
  ConsumerState<_PartnersTab> createState() => _PartnersTabState();
}

class _PartnersTabState extends ConsumerState<_PartnersTab> {
  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<AccountabilityPartner>>(
      future: widget.service.getPartners(),
      builder: (context, snapshot) {
        if (!snapshot.hasData && !snapshot.hasError) {
          return const Center(child: RoseLoader(size: 24));
        }
        if (snapshot.hasError) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.error_outline, color: AxonColors.error, size: 40),
                const SizedBox(height: 12),
                Text('Failed to load partners',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textSecondary, fontSize: 14)),
              ],
            ),
          );
        }

        final partners = snapshot.data!;

        if (partners.isEmpty) {
          return _PartnersEmptyState();
        }

        return ListView.builder(
          padding: const EdgeInsets.all(20),
          itemCount: partners.length,
          itemBuilder: (context, i) => _PartnerCard(
            partner: partners[i],
            onCheckIn: () => _showCheckInSheet(context, partners[i]),
            onRemind: () => widget.service.sendPartnerReminder(partners[i].id),
          ),
        );
      },
    );
  }

  void _showCheckInSheet(BuildContext context, AccountabilityPartner partner) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => _CheckInSheet(
        partner: partner,
        service: widget.service,
      ),
    );
  }
}

class _PartnersEmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: AxonColors.accent.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child:
                Icon(Icons.people_outline, size: 48, color: AxonColors.accent),
          ),
          const SizedBox(height: 24),
          Text(
            'No Partners Yet',
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Add accountability partners to stay motivated',
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}

class _PartnerCard extends StatelessWidget {
  final AccountabilityPartner partner;
  final VoidCallback onCheckIn;
  final VoidCallback onRemind;
  const _PartnerCard({
    required this.partner,
    required this.onCheckIn,
    required this.onRemind,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 24,
            backgroundColor: AxonColors.accent.withValues(alpha: 0.2),
            backgroundImage: partner.partnerPhoto != null
                ? NetworkImage(partner.partnerPhoto!)
                : null,
            child: partner.partnerPhoto == null
                ? Text(
                    partner.partnerName.isNotEmpty
                        ? partner.partnerName[0].toUpperCase()
                        : '?',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.accent,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  )
                : null,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  partner.partnerName,
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Icon(Icons.check_circle_outline,
                        size: 14, color: AxonColors.textTertiary),
                    const SizedBox(width: 4),
                    Text(
                      partner.lastCheckIn != null
                          ? 'Last check-in: ${_formatDate(partner.lastCheckIn!)}'
                          : 'No check-ins yet',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              GestureDetector(
                onTap: onCheckIn,
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AxonColors.accent.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.chat_bubble_outline,
                      color: AxonColors.accent, size: 18),
                ),
              ),
              const SizedBox(width: 8),
              GestureDetector(
                onTap: onRemind,
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AxonColors.surface.withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(Icons.notifications_outlined,
                      color: AxonColors.textSecondary, size: 18),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final diff = now.difference(date);
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return DateFormat('MMM d').format(date);
  }
}

// ─────────────────────────────────────────────────────────────────
// PUBLIC CHALLENGES TAB
// ─────────────────────────────────────────────────────────────────

class _PublicChallengesTab extends ConsumerWidget {
  final ChallengeService service;
  const _PublicChallengesTab({required this.service});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return FutureBuilder<List<Challenge>>(
      future: service.getPublicChallenges(),
      builder: (context, snapshot) {
        if (!snapshot.hasData && !snapshot.hasError) {
          return const Center(child: RoseLoader(size: 24));
        }
        if (snapshot.hasError) {
          return Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.error_outline, color: AxonColors.error, size: 40),
                const SizedBox(height: 12),
                Text('Failed to load challenges',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textSecondary, fontSize: 14)),
              ],
            ),
          );
        }

        final challenges = snapshot.data!;

        if (challenges.isEmpty) {
          return _PublicEmptyState();
        }

        return ListView.builder(
          padding: const EdgeInsets.all(20),
          itemCount: challenges.length,
          itemBuilder: (context, i) => _PublicChallengeCard(
            challenge: challenges[i],
            onJoin: () => service.joinChallenge(challenges[i].id),
          ),
        );
      },
    );
  }
}

class _PublicEmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: AxonColors.accent.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child:
                Icon(Icons.public_outlined, size: 48, color: AxonColors.accent),
          ),
          const SizedBox(height: 24),
          Text(
            'No Public Challenges',
            style: GoogleFonts.googleSans(
              color: AxonColors.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Be the first to create one!',
            style: GoogleFonts.googleSans(
              color: AxonColors.textTertiary,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}

class _PublicChallengeCard extends StatelessWidget {
  final Challenge challenge;
  final VoidCallback onJoin;
  const _PublicChallengeCard({required this.challenge, required this.onJoin});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          CircleAvatar(
            radius: 20,
            backgroundColor: AxonColors.accent.withValues(alpha: 0.2),
            backgroundImage: challenge.creatorPhoto != null
                ? NetworkImage(challenge.creatorPhoto!)
                : null,
            child: challenge.creatorPhoto == null
                ? Text(
                    challenge.creatorName.isNotEmpty
                        ? challenge.creatorName[0].toUpperCase()
                        : '?',
                    style: GoogleFonts.googleSans(
                      color: AxonColors.accent,
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                    ),
                  )
                : null,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  challenge.title,
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'by ${challenge.creatorName}',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          GestureDetector(
            onTap: onJoin,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: AxonColors.accent.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                'Join',
                style: GoogleFonts.googleSans(
                  color: AxonColors.accent,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────
// CREATE CHALLENGE SHEET
// ─────────────────────────────────────────────────────────────────

class _CreateChallengeSheet extends StatefulWidget {
  final ChallengeService service;
  const _CreateChallengeSheet({required this.service});

  @override
  State<_CreateChallengeSheet> createState() => _CreateChallengeSheetState();
}

class _CreateChallengeSheetState extends State<_CreateChallengeSheet> {
  final _titleController = TextEditingController();
  final _descController = TextEditingController();
  final _targetController = TextEditingController();
  String _selectedSubject = 'Mathematics';
  final ChallengeType _selectedType = ChallengeType.questionCount;
  DateTime _deadline = DateTime.now().add(const Duration(days: 1));
  bool _isPublic = true;

  final _subjects = [
    'Mathematics',
    'Physics',
    'Chemistry',
    'Biology',
    'English',
    'General'
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AxonColors.textTertiary.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Text(
              'Create Challenge',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _titleController,
              decoration: InputDecoration(
                labelText: 'Challenge Title',
                hintText: 'e.g., Finish 20 Physics questions',
                filled: true,
                fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _descController,
              maxLines: 2,
              decoration: InputDecoration(
                labelText: 'Description',
                hintText: 'What\'s the challenge about?',
                filled: true,
                fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              initialValue: _selectedSubject,
              decoration: InputDecoration(
                labelText: 'Subject',
                filled: true,
                fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
              items: _subjects
                  .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                  .toList(),
              onChanged: (v) => setState(() => _selectedSubject = v!),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _targetController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: 'Target Value',
                hintText: 'e.g., 20 questions',
                filled: true,
                fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
            const SizedBox(height: 16),
            ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text(
                'Deadline',
                style: GoogleFonts.googleSans(color: AxonColors.textPrimary),
              ),
              subtitle: Text(
                DateFormat('MMM d, y • h:mm a').format(_deadline),
                style: GoogleFonts.googleSans(color: AxonColors.accent),
              ),
              trailing: Icon(Icons.calendar_today, color: AxonColors.accent),
              onTap: () async {
                final pickerContext = context;
                final date = await showDatePicker(
                  context: pickerContext,
                  initialDate: _deadline,
                  firstDate: DateTime.now(),
                  lastDate: DateTime.now().add(const Duration(days: 30)),
                );
                if (!mounted || date == null) return;
                final time = await showTimePicker(
                  // ignore: use_build_context_synchronously
                  context: pickerContext,
                  initialTime: TimeOfDay.fromDateTime(_deadline),
                );
                if (!mounted) return;
                if (time != null) {
                  setState(() {
                    _deadline = DateTime(
                      date.year,
                      date.month,
                      date.day,
                      time.hour,
                      time.minute,
                    );
                  });
                }
              },
            ),
            const SizedBox(height: 16),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: Text(
                'Public Challenge',
                style: GoogleFonts.googleSans(color: AxonColors.textPrimary),
              ),
              subtitle: Text(
                'Anyone can join this challenge',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary, fontSize: 12),
              ),
              value: _isPublic,
              onChanged: (v) => setState(() => _isPublic = v),
              activeTrackColor: AxonColors.accent,
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _createChallenge,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: Text(
                  'Create Challenge',
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  void _createChallenge() async {
    if (_titleController.text.isEmpty || _targetController.text.isEmpty) return;

    await widget.service.createChallenge(
      title: _titleController.text,
      description: _descController.text.isEmpty
          ? 'Complete the challenge!'
          : _descController.text,
      type: _selectedType,
      targetValue: int.tryParse(_targetController.text) ?? 10,
      subject: _selectedSubject,
      deadline: _deadline,
      isPublic: _isPublic,
    );

    if (mounted) Navigator.pop(context);
  }
}

// ─────────────────────────────────────────────────────────────────
// CHECK-IN SHEET
// ─────────────────────────────────────────────────────────────────

class _CheckInSheet extends StatefulWidget {
  final AccountabilityPartner partner;
  final ChallengeService service;
  const _CheckInSheet({required this.partner, required this.service});

  @override
  State<_CheckInSheet> createState() => _CheckInSheetState();
}

class _CheckInSheetState extends State<_CheckInSheet> {
  final _messageController = TextEditingController();
  String _selectedMood = 'focused';
  final int _streak = 1;

  final _moods = [
    {'id': 'focused', 'emoji': 'FOCUS', 'label': 'Focused'},
    {'id': 'tired', 'emoji': 'LOW', 'label': 'Tired'},
    {'id': 'motivated', 'emoji': 'READY', 'label': 'Motivated'},
    {'id': 'struggling', 'emoji': 'HELP', 'label': 'Struggling'},
  ];

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AxonColors.textTertiary.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            Text(
              'Check-in with ${widget.partner.partnerName}',
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              'How are you feeling?',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: _moods.map((mood) {
                final isSelected = _selectedMood == mood['id'];
                return GestureDetector(
                  onTap: () => setState(() => _selectedMood = mood['id']!),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: isSelected
                          ? AxonColors.accent.withValues(alpha: 0.2)
                          : AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color:
                            isSelected ? AxonColors.accent : Colors.transparent,
                      ),
                    ),
                    child: Column(
                      children: [
                        Text(mood['emoji']!,
                            style: const TextStyle(fontSize: 24)),
                        const SizedBox(height: 4),
                        Text(
                          mood['label']!,
                          style: GoogleFonts.googleSans(
                            color: isSelected
                                ? AxonColors.accent
                                : AxonColors.textSecondary,
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _messageController,
              maxLines: 3,
              decoration: InputDecoration(
                hintText: 'Share an update with your partner...',
                filled: true,
                fillColor: AxonColors.surfaceHighlight.withValues(alpha: 0.5),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _sendCheckIn,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AxonColors.accent,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: Text(
                  'Send Check-in',
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  void _sendCheckIn() async {
    await widget.service.sendCheckIn(
      partnerId: widget.partner.id,
      message: _messageController.text.isEmpty
          ? 'Checking in!'
          : _messageController.text,
      streak: _streak,
      mood: _selectedMood,
    );

    if (mounted) Navigator.pop(context);
  }
}

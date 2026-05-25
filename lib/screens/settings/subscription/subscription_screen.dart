// lib/screens/settings/subscription/subscription_screen.dart
// ─────────────────────────────────────────────────────────────────
// Subscription / Paywall Screen
// Shows 4 tiers with feature comparison and Stripe checkout
// ─────────────────────────────────────────────────────────────────

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../../theme/app_theme.dart';
import '../../../services/subscription_service.dart';
import '../../../utils/nav_utils.dart';
import '../../../widgets/common/rose_loader.dart';
import '../../../widgets/axon_dialog.dart';

class SubscriptionScreen extends ConsumerStatefulWidget {
  final String? upgradeFrom;

  const SubscriptionScreen({super.key, this.upgradeFrom});

  @override
  ConsumerState<SubscriptionScreen> createState() => _SubscriptionScreenState();
}

class _SubscriptionScreenState extends ConsumerState<SubscriptionScreen> {
  bool _isYearly = true;
  SubscriptionTier? _loadingTier;
  String? _errorMessage;

  @override
  Widget build(BuildContext context) {
    final subState = SubscriptionService().current;

    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      extendBody: true,
      body: CustomScrollView(
        slivers: [
          _buildAppBar(),
          SliverToBoxAdapter(child: _buildHeader(subState)),
          SliverToBoxAdapter(child: _buildBillingToggle()),
          SliverToBoxAdapter(
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              transitionBuilder: (child, animation) => FadeTransition(
                opacity: animation,
                child: SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0, 0.05),
                    end: Offset.zero,
                  ).animate(animation),
                  child: child,
                ),
              ),
              child: _buildPricingCards(subState),
            ),
          ),
          SliverToBoxAdapter(child: const SizedBox(height: 24)),
          if (_errorMessage != null)
            SliverToBoxAdapter(child: _buildErrorBanner()),
          const SliverToBoxAdapter(child: SizedBox(height: 16)),
          SliverToBoxAdapter(child: _buildFeatureComparison(subState)),
          const SliverToBoxAdapter(child: SizedBox(height: 32)),
          SliverToBoxAdapter(child: _buildFAQ()),
          const SliverToBoxAdapter(child: SizedBox(height: 16)),
          const SliverToBoxAdapter(child: SizedBox(height: 24)),
        ],
      ),
    );
  }

  Widget _buildAppBar() {
    return SliverAppBar(
      backgroundColor: AxonColors.oxfordBlueDark,
      pinned: true,
      leading: IconButton(
        icon: const Icon(Icons.arrow_back, color: Colors.white),
        onPressed: () => popOrGo(context, '/settings'),
      ),
      title: Text(
        'Choose Your Plan',
        style: GoogleFonts.spaceGrotesk(
          color: Colors.white,
          fontSize: 18,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
      actions: [
        if (SubscriptionService().current.tier != SubscriptionTier.free)
          TextButton(
            onPressed: () => _openPortal(),
            child: Text(
              'Manage',
              style: TextStyle(color: AxonColors.electricCyan),
            ),
          ),
      ],
    );
  }

  Widget _buildHeader(SubscriptionState subState) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
      child: Column(
        children: [
          Text(
            'Unlock your full potential',
            style: GoogleFonts.spaceGrotesk(
              color: Colors.white,
              fontSize: 26,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.5,
            ),
            textAlign: TextAlign.center,
          ).animate().fadeIn(duration: 400.ms).slideY(begin: -0.1),
          const SizedBox(height: 8),
          Text(
            'Upgrade to unlock AI-powered study tools, personalized coaching, '
            'and every feature you need to excel.',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 14,
              height: 1.5,
            ),
            textAlign: TextAlign.center,
          ).animate().fadeIn(delay: 100.ms, duration: 400.ms),
          if (subState.isPaid) ...[
            const SizedBox(height: 16),
            _buildCurrentPlanBadge(subState),
          ],
        ],
      ),
    );
  }

  Widget _buildCurrentPlanBadge(SubscriptionState subState) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: AxonColors.electricCyan.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(20),
        border:
            Border.all(color: AxonColors.electricCyan.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.check_circle, color: AxonColors.electricCyan, size: 16),
          const SizedBox(width: 6),
          Text(
            'You\'re on ${subState.tier.displayName}',
            style: TextStyle(
              color: AxonColors.electricCyan,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    ).animate().fadeIn(delay: 200.ms);
  }

  Widget _buildBillingToggle() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: Container(
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Colors.white.withValues(alpha: 0.06),
              Colors.white.withValues(alpha: 0.03),
            ],
          ),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: Colors.white.withValues(alpha: 0.1),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(
              child: GestureDetector(
                onTap: () => setState(() => _isYearly = false),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 250),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  decoration: BoxDecoration(
                    gradient: !_isYearly
                        ? LinearGradient(
                            colors: [
                              Colors.white.withValues(alpha: 0.15),
                              Colors.white.withValues(alpha: 0.08),
                            ],
                          )
                        : null,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    'Monthly',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.inter(
                      color: !_isYearly ? Colors.white : Colors.white54,
                      fontWeight:
                          !_isYearly ? FontWeight.w600 : FontWeight.w400,
                      fontSize: 14,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 4),
            Expanded(
              child: GestureDetector(
                onTap: () => setState(() => _isYearly = true),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 250),
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                  decoration: BoxDecoration(
                    gradient: _isYearly
                        ? const LinearGradient(
                            colors: [
                              Color(0xFF10B981),
                              Color(0xFF059669),
                            ],
                          )
                        : null,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Yearly',
                        style: GoogleFonts.inter(
                          color: _isYearly ? Colors.white : Colors.white54,
                          fontWeight:
                              _isYearly ? FontWeight.w600 : FontWeight.w400,
                          fontSize: 14,
                        ),
                      ),
                      if (_isYearly) ...[
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: Colors.white.withValues(alpha: 0.25),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Text(
                            'Save 20%',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 9,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    ).animate().fadeIn(delay: 150.ms);
  }

  Widget _buildPricingCards(SubscriptionState subState) {
    final tiers = [
      SubscriptionTier.free,
      SubscriptionTier.plus,
      SubscriptionTier.premium,
      SubscriptionTier.pro
    ];

    return Column(
      children: List.generate(tiers.length, (i) {
        final tier = tiers[i];
        return _buildPricingCard(tier, subState, i)
            .animate()
            .fadeIn(delay: Duration(milliseconds: 200 + i * 100))
            .slideY(begin: 0.05);
      }),
    );
  }

  Widget _buildPricingCard(
      SubscriptionTier tier, SubscriptionState subState, int index) {
    final isCurrentTier = subState.tier == tier;
    final isHighlighted = tier == SubscriptionTier.premium;

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: isHighlighted
            ? Colors.white.withValues(alpha: 0.05)
            : Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: isHighlighted
              ? AxonColors.electricCyan.withValues(alpha: 0.5)
              : isCurrentTier
                  ? Colors.white.withValues(alpha: 0.15)
                  : Colors.white.withValues(alpha: 0.08),
          width: isHighlighted ? 1.5 : 1,
        ),
        boxShadow: isHighlighted
            ? [
                BoxShadow(
                  color: AxonColors.electricCyan.withValues(alpha: 0.15),
                  blurRadius: 24,
                  spreadRadius: -4,
                ),
              ]
            : null,
      ),
      child: Column(
        children: [
          if (isHighlighted)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 6),
              decoration: BoxDecoration(
                color: AxonColors.accent,
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(22)),
                boxShadow: [
                  BoxShadow(
                    color: AxonColors.accent.withValues(alpha: 0.4),
                    blurRadius: 12,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Text(
                'MOST POPULAR',
                textAlign: TextAlign.center,
                style: GoogleFonts.spaceGrotesk(
                  color: Colors.black,
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.5,
                ),
              ),
            ),
          Padding(
            padding: EdgeInsets.fromLTRB(20, isHighlighted ? 16 : 20, 20, 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                tier.displayName.toUpperCase(),
                                style: GoogleFonts.spaceGrotesk(
                                  color: isHighlighted
                                      ? AxonColors.electricCyan
                                      : Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: 1.2,
                                ),
                              ),
                              if (isCurrentTier) ...[
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withValues(alpha: 0.15),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: const Text(
                                    'Current',
                                    style: TextStyle(
                                      color: Colors.white70,
                                      fontSize: 10,
                                    ),
                                  ),
                                ),
                              ],
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            tier.subtitle,
                            style: GoogleFonts.inter(
                              color: Colors.white.withValues(alpha: 0.5),
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              _isYearly && tier.hasYearlyDiscount
                                  ? tier.priceYearly.split('/')[0]
                                  : tier.priceMonthly,
                              style: GoogleFonts.spaceGrotesk(
                                color: Colors.white,
                                fontSize: 32,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            if (tier.isPaid) ...[
                              Text(
                                _isYearly ? '/yr' : '/mo',
                                style: TextStyle(
                                  color: Colors.white.withValues(alpha: 0.5),
                                  fontSize: 13,
                                ),
                              ),
                            ],
                          ],
                        ),
                        if (tier.isPaid && _isYearly)
                          Text(
                            '\$${tier.monthlyPriceUsd.toStringAsFixed(2)}/mo',
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.4),
                              fontSize: 11,
                              decoration: TextDecoration.lineThrough,
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _buildPricingCTA(tier, subState),
                const SizedBox(height: 16),
                _buildTopFeatures(tier),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPricingCTA(SubscriptionTier tier, SubscriptionState subState) {
    if (tier == SubscriptionTier.free) {
      return SizedBox(
        width: double.infinity,
        child: OutlinedButton(
          onPressed: null,
          style: OutlinedButton.styleFrom(
            foregroundColor: Colors.white54,
            side: BorderSide(color: Colors.white.withValues(alpha: 0.15)),
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          child: const Text('Current Plan'),
        ),
      );
    }

    final isCurrentTier = subState.tier == tier;
    final isUpgrade = tier.index > subState.tier.index;

    if (isCurrentTier && subState.status == SubscriptionStatus.active) {
      return SizedBox(
        width: double.infinity,
        child: ElevatedButton(
          onPressed: () => _openPortal(),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.white.withValues(alpha: 0.1),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          child: const Text('Manage Subscription'),
        ),
      );
    }

    final isLoading = _loadingTier == tier;
    final label = isUpgrade ? 'Upgrade to ${tier.displayName}' : 'Downgrade';

    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: isLoading ? null : () => _handleCheckout(tier, isUpgrade),
        style: ElevatedButton.styleFrom(
          backgroundColor: isUpgrade
              ? AxonColors.electricCyan
              : Colors.white.withValues(alpha: 0.1),
          foregroundColor: isUpgrade ? Colors.black : Colors.white,
          disabledBackgroundColor: Colors.white.withValues(alpha: 0.05),
          disabledForegroundColor: Colors.white30,
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: isLoading
            ? const RoseLoader(size: 20, color: Colors.black54)
            : Text(
                label,
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
      ),
    );
  }

  Widget _buildTopFeatures(SubscriptionTier tier) {
    final newFeatures = TierFeatureMatrix.getNewFeaturesAtTier(tier).take(4);
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: newFeatures.map((f) {
        return ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 180),
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.06),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(f.icon, style: const TextStyle(fontSize: 11)),
                const SizedBox(width: 4),
                Flexible(
                  child: Text(
                    f.displayName,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    softWrap: false,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.7),
                      fontSize: 11,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildFeatureComparison(SubscriptionState subState) {
    const text = 'Compare all features';
    return ExpansionTile(
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: text
            .split('')
            .map((char) => Text(
                  char,
                  style: GoogleFonts.googleSans(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ))
            .toList(),
      ),
      collapsedIconColor: Colors.white54,
      iconColor: AxonColors.electricCyan,
      childrenPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      children: [
        IntrinsicWidth(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header row
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(
                  children: [
                    const SizedBox(width: 24), // icon space
                    const SizedBox(width: 10),
                    Expanded(
                        child: Text('Feature',
                            style: TextStyle(
                                color: Colors.white54,
                                fontSize: 11,
                                fontWeight: FontWeight.w600))),
                    const SizedBox(width: 8),
                    SizedBox(
                        width: 60,
                        child: Text('Free',
                            style: TextStyle(
                                color: Colors.white54,
                                fontSize: 10,
                                fontWeight: FontWeight.w600),
                            textAlign: TextAlign.center)),
                    SizedBox(
                        width: 60,
                        child: Text('Plus',
                            style: TextStyle(
                                color: Colors.white54,
                                fontSize: 10,
                                fontWeight: FontWeight.w600),
                            textAlign: TextAlign.center)),
                    SizedBox(
                        width: 60,
                        child: Text('Premium',
                            style: TextStyle(
                                color: Colors.white54,
                                fontSize: 10,
                                fontWeight: FontWeight.w600),
                            textAlign: TextAlign.center)),
                    SizedBox(
                        width: 60,
                        child: Text('Pro',
                            style: TextStyle(
                                color: Colors.white54,
                                fontSize: 10,
                                fontWeight: FontWeight.w600),
                            textAlign: TextAlign.center)),
                  ],
                ),
              ),
              ...Feature.values.map((feature) {
                final tiersWithFeature = SubscriptionTier.values
                    .where((t) => TierFeatureMatrix.hasFeature(t, feature))
                    .toList();
                return _buildFeatureRow(feature, tiersWithFeature, subState);
              }),
            ],
          ),
        ),
      ],
    ).animate().fadeIn(delay: 600.ms);
  }

  Widget _buildFeatureRow(Feature feature,
      List<SubscriptionTier> tiersWithAccess, SubscriptionState subState) {
    final hasAccess = tiersWithAccess.contains(subState.tier);
    return SizedBox(
      width: double.infinity,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          children: [
            Text(feature.icon, style: const TextStyle(fontSize: 14)),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    feature.displayName,
                    style: TextStyle(
                      color: hasAccess ? Colors.white : Colors.white54,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  Text(
                    feature.description,
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.35),
                      fontSize: 10,
                    ),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 60,
              child: tiersWithAccess.contains(SubscriptionTier.free)
                  ? Icon(Icons.check, color: const Color(0xFF10B981), size: 18)
                  : tiersWithAccess.contains(subState.tier)
                      ? Icon(Icons.check,
                          color: AxonColors.electricCyan, size: 18)
                      : Icon(Icons.lock_outline,
                          color: Colors.white.withValues(alpha: 0.2), size: 16),
            ),
            for (final tier in [
              SubscriptionTier.plus,
              SubscriptionTier.premium,
              SubscriptionTier.pro
            ])
              SizedBox(
                width: 60,
                child: Icon(
                  tiersWithAccess.contains(tier) ? Icons.check : Icons.close,
                  color: tiersWithAccess.contains(tier)
                      ? const Color(0xFF10B981)
                      : Colors.white.withValues(alpha: 0.15),
                  size: 16,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildFAQ() {
    return ExpansionTile(
      title: Text(
        'Frequently Asked Questions',
        style: GoogleFonts.googleSans(
          color: Colors.white,
          fontSize: 16,
          fontWeight: FontWeight.w600,
        ),
      ),
      collapsedIconColor: Colors.white54,
      iconColor: Colors.white54,
      children: [
        _buildFAQItem(
          'Can I cancel anytime?',
          'Yes. Cancel at any time from your account settings. '
              'You\'ll keep access until the end of your billing period. '
              'No hidden fees or cancellation charges.',
        ),
        _buildFAQItem(
          'What payment methods do you accept?',
          'We accept all major credit and debit cards (Visa, Mastercard, AMEX) '
              'through our secure payment provider, Stripe.',
        ),
        _buildFAQItem(
          'Is there a free trial?',
          'Plus subscribers get a 7-day free trial. No credit card required '
              'to start. Cancel before the trial ends and you won\'t be charged.',
        ),
        _buildFAQItem(
          'What happens to my data if I downgrade?',
          'Your data is always safe. Downgrading restricts new feature access '
              'but your existing study history, analytics, and progress are preserved.',
        ),
        _buildFAQItem(
          'Can I switch between monthly and yearly?',
          'Yes. Switch anytime from your subscription management portal. '
              'Yearly subscribers get a pro-rated credit for unused time.',
        ),
      ],
    ).animate().fadeIn(delay: 700.ms);
  }

  Widget _buildFAQItem(String question, String answer) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            question,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            answer,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 13,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildErrorBanner() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.red, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _errorMessage ?? 'Something went wrong. Please try again.',
              style: const TextStyle(color: Colors.red, fontSize: 13),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.close, color: Colors.red, size: 16),
            onPressed: () => setState(() => _errorMessage = null),
          ),
        ],
      ),
    );
  }

  Future<void> _handleCheckout(SubscriptionTier tier, bool isUpgrade) async {
    setState(() {
      _loadingTier = tier;
      _errorMessage = null;
    });

    try {
      final svc = SubscriptionService();
      final checkoutUrl = await svc.startCheckout(
        tier: tier,
        isYearly: _isYearly,
        successUrl: 'axon://subscription/success',
        cancelUrl: 'axon://settings/subscription',
      );

      if (checkoutUrl != null && checkoutUrl.isNotEmpty) {
        if (context.mounted) {
          _showCheckoutDialog(checkoutUrl);
        }
      } else {
        setState(() => _errorMessage =
            'Checkout unavailable. Please configure Stripe Price IDs.');
      }
    } catch (e) {
      setState(
          () => _errorMessage = 'Failed to start checkout. Please try again.');
    } finally {
      if (mounted) {
        setState(() {
          _loadingTier = null;
        });
      }
    }
  }

  void _showCheckoutDialog(String checkoutUrl) {
    AxonDialog.showCustom(
      context: context,
      title: 'Complete Your Purchase',
      backgroundColor: const Color(0xFF1a1a2e),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'To complete your subscription, open the Stripe checkout page:\n',
            style: TextStyle(color: Colors.white.withValues(alpha: 0.7)),
          ),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
            ),
            child: SelectableText(
              checkoutUrl,
              style: const TextStyle(
                color: Color(0xFF3A86FF),
                fontSize: 11,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            '1. Copy the URL above\n'
            '2. Open it in your browser\n'
            '3. Complete payment on Stripe\n'
            '4. Return here — access unlocks automatically',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 12,
              height: 1.5,
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Done'),
        ),
      ],
    );
  }

  Future<void> _openPortal() async {
    final svc = SubscriptionService();
    final portalUrl = await svc.openCustomerPortal();
    if (portalUrl != null && portalUrl.isNotEmpty && mounted) {
      _showPortalDialog(portalUrl);
    }
  }

  void _showPortalDialog(String portalUrl) {
    AxonDialog.showCustom(
      context: context,
      title: 'Manage Subscription',
      backgroundColor: const Color(0xFF1a1a2e),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Open the Stripe portal to cancel, upgrade, or change your billing.',
            style: TextStyle(color: Colors.white.withValues(alpha: 0.7)),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
            ),
            child: SelectableText(
              portalUrl,
              style: const TextStyle(color: Color(0xFF3A86FF), fontSize: 11),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Close'),
        ),
      ],
    );
  }
}

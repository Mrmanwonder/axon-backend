// lib/screens/leaderboard/leaderboard_screen.dart
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:go_router/go_router.dart';
import '../../services/leaderboard_service.dart';
import '../../services/app_state.dart';
import '../../services/subscription_service.dart';
import '../../theme/app_theme.dart';
import '../../widgets/common/rose_refresh_indicator.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/axon_widgets.dart';
import '../../widgets/common/rose_loader.dart';

String _leaderboardInitials(String name) {
  final parts = name.trim().split(RegExp(r'\s+'));
  if (parts.length >= 2 && parts[0].isNotEmpty && parts[1].isNotEmpty) {
    return '${parts[0][0]}${parts[1][0]}'.toUpperCase();
  }
  return name.isNotEmpty ? name[0].toUpperCase() : '?';
}

class LeaderboardScreen extends ConsumerStatefulWidget {
  const LeaderboardScreen({super.key});

  @override
  ConsumerState<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends ConsumerState<LeaderboardScreen> {
  final LeaderboardService _service = LeaderboardService();
  List<LeaderboardEntry> _leaderboard = [];
  LeaderboardEntry? _currentUser;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadLeaderboard();
  }

  Future<void> _loadLeaderboard() async {
    try {
      final user = ref.read(authStateProvider).user;
      final entries = await _service.getLeaderboard(limit: 50);

      LeaderboardEntry? userEntry;
      if (user != null) {
        userEntry = await _service.getUserRank(user.uid);
      }

      if (mounted) {
        setState(() {
          _leaderboard = entries;
          _currentUser = userEntry;
          _loading = false;
          _error = null;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = e.toString();
        });
      }
    }
  }

  Widget _buildPaywallGate(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.leaderboard_outlined,
                size: 64, color: AxonColors.textTertiary),
            const SizedBox(height: 16),
            Text('Leaderboard',
                style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 20,
                    fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            Text(
                'Compete with peers, track rankings, and earn rewards with Plus or higher.',
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
    if (!svc.hasAccess(Feature.leaderboard)) {
      return _buildPaywallGate(context);
    }

    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      body: Container(
        color: AxonColors.oxfordBlueDark,
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.all(20),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: () => popOrGo(context, '/home'),
                      child: Row(
                        children: [
                          Icon(Icons.arrow_back_rounded,
                              color: AxonColors.textSecondary, size: 20),
                          const SizedBox(width: 6),
                          Text('Back',
                              style: GoogleFonts.googleSans(
                                  color: AxonColors.textSecondary,
                                  fontSize: 14)),
                        ],
                      ),
                    ),
                    const Spacer(),
                    Text('Leaderboard',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textPrimary,
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                        )),
                    const Spacer(),
                    const SizedBox(width: 60),
                  ],
                ),
              ),

              // Your Rank Card
              if (_currentUser != null)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  child: AxonCard(
                    glowColor: AxonColors.accent,
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        _LeaderboardAvatar(
                          size: 48,
                          photoUrl: _currentUser!.photoUrl,
                          displayName: _currentUser!.displayName,
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Your Rank',
                                style: GoogleFonts.googleSans(
                                  color: AxonColors.textTertiary,
                                  fontSize: 12,
                                ),
                              ),
                              Text(
                                _currentUser!.displayName,
                                style: GoogleFonts.googleSans(
                                  color: AxonColors.textPrimary,
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              '${_currentUser!.totalSessions}',
                              style: GoogleFonts.googleSans(
                                color: AxonColors.accent,
                                fontSize: 20,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            Text(
                              'sessions',
                              style: GoogleFonts.googleSans(
                                color: AxonColors.textTertiary,
                                fontSize: 11,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),

              const SizedBox(height: 20),

              // Column Headers
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Row(
                  children: [
                    const SizedBox(width: 36),
                    Text('Student',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        )),
                    const Spacer(),
                    Text('Sessions',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        )),
                    const SizedBox(width: 50),
                    Text('Streak',
                        style: GoogleFonts.googleSans(
                          color: AxonColors.textTertiary,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        )),
                    const SizedBox(width: 20),
                  ],
                ),
              ),

              const SizedBox(height: 12),

              // Leaderboard List
              Expanded(
                child: _loading
                    ? Center(
                        child: RoseLoader(size: 24, color: AxonColors.accent),
                      )
                    : _error != null
                        ? Center(
                            child: Padding(
                              padding: const EdgeInsets.all(32),
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Icon(Icons.error_outline,
                                      size: 48, color: AxonColors.error),
                                  const SizedBox(height: 16),
                                  Text('Failed to load leaderboard',
                                      style: GoogleFonts.googleSans(
                                          color: AxonColors.textPrimary,
                                          fontSize: 16,
                                          fontWeight: FontWeight.w600)),
                                  const SizedBox(height: 8),
                                  Text(_error!,
                                      textAlign: TextAlign.center,
                                      style: GoogleFonts.googleSans(
                                          color: AxonColors.textSecondary,
                                          fontSize: 13)),
                                  const SizedBox(height: 20),
                                  ElevatedButton.icon(
                                    onPressed: _loadLeaderboard,
                                    icon: const Icon(Icons.refresh, size: 18),
                                    label: Text('Retry',
                                        style: GoogleFonts.googleSans()),
                                    style: ElevatedButton.styleFrom(
                                      backgroundColor: AxonColors.accent,
                                      foregroundColor: Colors.white,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          )
                        : _leaderboard.isEmpty
                            ? Center(
                                child: Column(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(Icons.leaderboard_outlined,
                                        color: AxonColors.textTertiary,
                                        size: 48),
                                    const SizedBox(height: 12),
                                    Text(
                                      'No students yet',
                                      style: GoogleFonts.googleSans(
                                        color: AxonColors.textTertiary,
                                        fontSize: 14,
                                      ),
                                    ),
                                  ],
                                ),
                              )
                            : RoseRefreshIndicator(
                                onRefresh: _loadLeaderboard,
                                color: AxonColors.accent,
                                child: ListView.builder(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 20),
                                  itemCount: _leaderboard.length,
                                  itemBuilder: (context, index) {
                                    final entry = _leaderboard[index];
                                    final user =
                                        ref.read(authStateProvider).user;
                                    final isCurrentUser =
                                        user?.uid == entry.uid;
                                    return _LeaderboardRow(
                                      rank: index + 1,
                                      entry: entry,
                                      isCurrentUser: isCurrentUser,
                                    );
                                  },
                                ),
                              ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LeaderboardRow extends StatelessWidget {
  final int rank;
  final LeaderboardEntry entry;
  final bool isCurrentUser;

  const _LeaderboardRow({
    required this.rank,
    required this.entry,
    required this.isCurrentUser,
  });

  @override
  Widget build(BuildContext context) {
    final isTopThree = rank <= 3;
    final rankColor = rank == 1
        ? const Color(0xFFFFD700)
        : rank == 2
            ? const Color(0xFFC0C0C0)
            : rank == 3
                ? const Color(0xFFCD7F32)
                : AxonColors.textTertiary;

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: isCurrentUser
            ? AxonColors.accent.withValues(alpha: 0.1)
            : AxonColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: isCurrentUser
            ? Border.all(color: AxonColors.accent.withValues(alpha: 0.3))
            : null,
      ),
      child: Row(
        children: [
          // Rank
          SizedBox(
            width: 36,
            child: isTopThree
                ? Icon(Icons.emoji_events_rounded, color: rankColor, size: 22)
                : Text(
                    '#$rank',
                    style: GoogleFonts.googleSans(
                      color: rankColor,
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
          ),

          // Avatar
          _LeaderboardAvatar(
            size: 36,
            photoUrl: entry.photoUrl,
            displayName: entry.displayName,
          ),

          const SizedBox(width: 10),

          // Name
          Expanded(
            child: Text(
              entry.displayName,
              style: GoogleFonts.googleSans(
                color:
                    isCurrentUser ? AxonColors.accent : AxonColors.textPrimary,
                fontSize: 14,
                fontWeight: isCurrentUser ? FontWeight.w700 : FontWeight.w500,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),

          // Sessions
          SizedBox(
            width: 50,
            child: Text(
              '${entry.totalSessions}',
              textAlign: TextAlign.center,
              style: GoogleFonts.googleSans(
                color: AxonColors.textPrimary,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),

          // Streak
          SizedBox(
            width: 50,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.local_fire_department_rounded,
                    color: entry.currentStreak > 0
                        ? AxonColors.warning
                        : AxonColors.textTertiary,
                    size: 14),
                const SizedBox(width: 2),
                Text(
                  '${entry.currentStreak}',
                  style: GoogleFonts.googleSans(
                    color: entry.currentStreak > 0
                        ? AxonColors.warning
                        : AxonColors.textTertiary,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _LeaderboardAvatar extends StatelessWidget {
  final double size;
  final String? photoUrl;
  final String displayName;

  const _LeaderboardAvatar({
    required this.size,
    required this.photoUrl,
    required this.displayName,
  });

  @override
  Widget build(BuildContext context) {
    final trimmedPhotoUrl = photoUrl?.trim();
    final hasPhoto = trimmedPhotoUrl != null && trimmedPhotoUrl.isNotEmpty;
    final isLocalFile = hasPhoto &&
        !trimmedPhotoUrl.startsWith('http://') &&
        !trimmedPhotoUrl.startsWith('https://');
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: AxonColors.accent.withValues(alpha: 0.15),
        shape: BoxShape.circle,
      ),
      clipBehavior: Clip.antiAlias,
      child: hasPhoto
          ? isLocalFile
              ? Image.file(
                  File(trimmedPhotoUrl),
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => _LeaderboardInitials(
                    displayName: displayName,
                    size: size,
                  ),
                )
              : CachedNetworkImage(
                  imageUrl: trimmedPhotoUrl,
                  fit: BoxFit.cover,
                  errorWidget: (_, __, ___) => _LeaderboardInitials(
                    displayName: displayName,
                    size: size,
                  ),
                )
          : _LeaderboardInitials(
              displayName: displayName,
              size: size,
            ),
    );
  }
}

class _LeaderboardInitials extends StatelessWidget {
  final String displayName;
  final double size;

  const _LeaderboardInitials({
    required this.displayName,
    required this.size,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text(
        _leaderboardInitials(displayName),
        style: GoogleFonts.googleSans(
          color: AxonColors.accent,
          fontWeight: FontWeight.w700,
          fontSize: size * 0.34,
        ),
      ),
    );
  }
}

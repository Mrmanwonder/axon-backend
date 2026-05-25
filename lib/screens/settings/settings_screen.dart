import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme/app_theme.dart';
import '../../services/app_state.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final auth = ref.watch(authStateProvider);
    final user = auth.user;

    return Scaffold(
      backgroundColor: AxonColors.background,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('SETTINGS', style: GoogleFonts.googleSans(
                    color: AxonColors.accent, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 2)),
                  const SizedBox(height: 4),
                  Text('Preferences & Configuration', style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary, fontSize: 20, fontWeight: FontWeight.w700)),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                children: [
                  _buildSection('ACCOUNT'),
                  _SettingsTile(
                    icon: Icons.person_rounded,
                    title: 'Account',
                    subtitle: user?.displayName ?? 'Edit profile',
                    trailing: user?.email ?? '',
                    onTap: () => context.push('/settings/account'),
                  ),
                  _SettingsTile(
                    icon: Icons.sync_rounded,
                    title: 'Sync',
                    subtitle: 'Google Drive, calendar, and data sync',
                    onTap: () => context.push('/settings/sync'),
                  ),
                  const SizedBox(height: 24),
                  _buildSection('STUDY'),
                  _SettingsTile(
                    icon: Icons.psychology_rounded,
                    title: 'Coaching Persona',
                    subtitle: 'Choose how AXON communicates with you',
                    onTap: () => context.push('/settings/coaching-persona'),
                  ),
                  _SettingsTile(
                    icon: Icons.lock_rounded,
                    title: 'Study Lock',
                    subtitle: 'Block distracting apps during study sessions',
                    onTap: () => context.push('/settings/study-lock'),
                  ),
                  const SizedBox(height: 24),
                  _buildSection('APP'),
                  _SettingsTile(
                    icon: Icons.accessibility_new_rounded,
                    title: 'Accessibility',
                    subtitle: 'Visual and interaction preferences',
                    onTap: () => context.push('/settings/accessibility'),
                  ),
                  _SettingsTile(
                    icon: Icons.subscriptions_rounded,
                    title: 'Subscription',
                    subtitle: 'Manage your plan and billing',
                    onTap: () => context.push('/settings/subscription'),
                  ),
                  const SizedBox(height: 24),
                  _buildSection('SESSION'),
                  _SettingsTile(
                    icon: Icons.logout_rounded,
                    title: 'Sign Out',
                    subtitle: 'End your current session',
                    iconColor: AxonColors.error,
                    onTap: () async {
                      final confirm = await showDialog<bool>(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          title: const Text('Sign Out'),
                          content: const Text('Are you sure you want to sign out?'),
                          actions: [
                            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                            TextButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Sign Out')),
                          ],
                        ),
                      );
                      if (confirm == true && context.mounted) {
                        ref.read(authStateProvider.notifier).signOut();
                      }
                    },
                  ),
                  _SettingsTile(
                    icon: Icons.delete_forever_rounded,
                    title: 'Delete Account',
                    subtitle: 'Permanently remove your account and data',
                    iconColor: AxonColors.error,
                    onTap: () async {
                      final confirm = await showDialog<bool>(
                        context: context,
                        builder: (ctx) => AlertDialog(
                          title: const Text('Delete Account'),
                          content: const Text('This action cannot be undone. All your data will be permanently deleted.'),
                          actions: [
                            TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
                            TextButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete', style: TextStyle(color: AxonColors.error))),
                          ],
                        ),
                      );
                      if (confirm == true && context.mounted) {
                        final user = ref.read(authStateProvider.notifier);
                        await user.signOut();
                        if (context.mounted) context.go('/auth/login');
                      }
                    },
                  ),
                  const SizedBox(height: 24),
                  _buildSection('INFO'),
                  const SizedBox(height: 80),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSection(String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(label, style: GoogleFonts.googleSans(
        color: AxonColors.accent, fontSize: 10, fontWeight: FontWeight.w600, letterSpacing: 1.5)),
    );
  }
}

class _SettingsTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final String? trailing;
  final Color? iconColor;
  final VoidCallback onTap;

  const _SettingsTile({
    required this.icon, required this.title, required this.subtitle,
    this.trailing, this.iconColor, required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.only(bottom: 2),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AxonColors.divider, width: 0.5),
        ),
        child: Row(
          children: [
            Container(
              width: 36, height: 36,
              decoration: BoxDecoration(
                color: (iconColor ?? AxonColors.accent).withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(icon, color: iconColor ?? AxonColors.accent, size: 18),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: GoogleFonts.googleSans(color: AxonColors.textPrimary, fontSize: 14, fontWeight: FontWeight.w600)),
                  Text(subtitle, style: GoogleFonts.googleSans(color: AxonColors.textTertiary, fontSize: 11)),
                ],
              ),
            ),
            if (trailing != null)
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: Text(trailing!, style: GoogleFonts.googleSans(color: AxonColors.textTertiary, fontSize: 11)),
              ),
            Icon(Icons.chevron_right_rounded, color: AxonColors.textTertiary, size: 18),
          ],
        ),
      ),
    );
  }
}

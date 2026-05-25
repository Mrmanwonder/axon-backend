// lib/screens/settings/sync_settings_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../theme/app_theme.dart';
import '../../services/notion_service.dart';
import '../../services/google_calendar_service.dart';
import '../../services/google_drive_downloader.dart';
import '../../services/obsidian_service.dart';
import '../../services/sync_manager.dart';
import '../../services/alert_service.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/rose_loader.dart';

enum SyncService { notion, googleCalendar, googleDrive, obsidian }

class SyncSettingsScreen extends ConsumerStatefulWidget {
  const SyncSettingsScreen({super.key});

  @override
  ConsumerState<SyncSettingsScreen> createState() => _SyncSettingsScreenState();
}

class _SyncSettingsScreenState extends ConsumerState<SyncSettingsScreen> {
  Map<SyncService, SyncStatus> _statuses = {};
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadStatuses();
  }

  Future<void> _loadStatuses() async {
    final prefs = await SharedPreferences.getInstance();
    final notionService = NotionService();
    final googleService = GoogleCalendarService();
    final obsidianService = ObsidianService();
    final offlineEnabled = prefs.getBool('offline_mode_enabled') ?? false;

    await notionService.initialize();
    await googleService.initialize();
    await obsidianService.initialize();

    if (mounted) {
      setState(() {
        _statuses = {
          SyncService.notion: SyncStatus(
            isConnected: notionService.isConnected,
            lastSync: prefs.getInt('notion_last_sync') != null
                ? DateTime.fromMillisecondsSinceEpoch(
                    prefs.getInt('notion_last_sync')!)
                : null,
          ),
          SyncService.googleCalendar: SyncStatus(
            isConnected: googleService.isConnected,
            lastSync: prefs.getInt('google_calendar_last_sync') != null
                ? DateTime.fromMillisecondsSinceEpoch(
                    prefs.getInt('google_calendar_last_sync')!)
                : null,
          ),
          SyncService.googleDrive: SyncStatus(
            isConnected: offlineEnabled,
            lastSync: prefs.getInt('google_drive_last_sync') != null
                ? DateTime.fromMillisecondsSinceEpoch(
                    prefs.getInt('google_drive_last_sync')!)
                : null,
          ),
          SyncService.obsidian: SyncStatus(
            isConnected: obsidianService.isConnected,
            lastSync: prefs.getInt('obsidian_last_sync') != null
                ? DateTime.fromMillisecondsSinceEpoch(
                    prefs.getInt('obsidian_last_sync')!)
                : null,
          ),
        };
        _isLoading = false;
      });
    }
  }

  Future<void> _connect(SyncService service) async {
    HapticFeedback.mediumImpact();

    switch (service) {
      case SyncService.notion:
        await _connectNotion();
        break;
      case SyncService.googleCalendar:
        await _connectGoogleCalendar();
        break;
      case SyncService.googleDrive:
        await _enableOfflineMode();
        break;
      case SyncService.obsidian:
        await _connectObsidian();
        break;
    }

    await _loadStatuses();
  }

  Future<void> _connectNotion() async {
    final token = await _showTextInputDialog(
      title: 'Connect Notion',
      label: 'Integration Token',
      hint: 'Enter your Notion integration token',
    );
    if (token == null || token.isEmpty) return;

    final databaseId = await _showTextInputDialog(
      title: 'Notion Database',
      label: 'Database ID',
      hint: 'Enter your study plan database ID',
    );
    if (databaseId == null || databaseId.isEmpty) return;

    final service = NotionService();
    await service.connect(token, databaseId);
  }

  Future<void> _connectGoogleCalendar() async {
    final service = GoogleCalendarService();
    try {
      await service.signIn();
    } catch (e) {
      if (mounted) {
        AlertService.showError(
          context,
          'Connection Failed',
          'Google Sign-In failed: $e',
        );
      }
    }
  }

  Future<void> _connectObsidian() async {
    final path = await _showTextInputDialog(
      title: 'Connect Obsidian',
      label: 'Vault Path',
      hint: '/storage/emulated/0/Documents/Obsidian/Vault',
    );
    if (path == null || path.isEmpty) return;

    final service = ObsidianService();
    try {
      await service.connect(path);
    } catch (e) {
      if (mounted) {
        AlertService.showError(
          context,
          'Connection Failed',
          'Failed to connect: $e',
        );
      }
    }
  }

  Future<void> _enableOfflineMode() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('offline_mode_enabled', true);
    await GoogleDriveDownloader.instance.initialize();
  }

  Future<void> _disableOfflineMode() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('offline_mode_enabled', false);
  }

  Future<void> _disconnect(SyncService service) async {
    HapticFeedback.lightImpact();
    final prefs = await SharedPreferences.getInstance();

    switch (service) {
      case SyncService.notion:
        await NotionService().disconnect();
        await prefs.remove('notion_integration_token');
        await prefs.remove('notion_database_id');
        break;
      case SyncService.googleCalendar:
        await GoogleCalendarService().signOut();
        break;
      case SyncService.googleDrive:
        await _disableOfflineMode();
        break;
      case SyncService.obsidian:
        await ObsidianService().disconnect();
        await prefs.remove('obsidian_vault_path');
        break;
    }

    await _loadStatuses();
  }

  Future<void> _syncNow(SyncService service) async {
    HapticFeedback.selectionClick();

    try {
      switch (service) {
        case SyncService.notion:
          await _syncNotion();
          break;
        case SyncService.googleCalendar:
          await _syncGoogleCalendar();
          break;
        case SyncService.googleDrive:
          await GoogleDriveDownloader.instance.initialize();
          break;
        case SyncService.obsidian:
          await _syncObsidian();
          break;
      }

      await _loadStatuses();

      if (mounted) {
        AlertService.showSuccess(
          context,
          'Sync Complete',
          '${_serviceName(service)} synced',
        );
      }
    } catch (e) {
      if (mounted) {
        AlertService.showError(
          context,
          'Sync Failed',
          'Sync failed: $e',
        );
      }
    }
  }

  Future<void> _syncNotion() async {
    try {
      await SyncManager().syncNotion();
    } catch (e) {
      rethrow;
    }
  }

  Future<void> _syncGoogleCalendar() async {
    try {
      await SyncManager().syncGoogleCalendar();
    } catch (e) {
      rethrow;
    }
  }

  Future<void> _syncObsidian() async {
    try {
      await SyncManager().syncObsidian();
    } catch (e) {
      rethrow;
    }
  }

  Future<String?> _showTextInputDialog({
    required String title,
    required String label,
    required String hint,
  }) async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      barrierDismissible: true,
      builder: (context) => Dismissible(
        key: UniqueKey(),
        direction: DismissDirection.vertical,
        onDismissed: (_) => Navigator.pop(context),
        child: AlertDialog(
          backgroundColor: AxonColors.surfaceElevated,
          title:
              Text(title, style: GoogleFonts.googleSans(color: Colors.white)),
          content: TextField(
            controller: controller,
            decoration: InputDecoration(
              labelText: label,
              hintText: hint,
              hintStyle: const TextStyle(color: Color(0xFF666666)),
              labelStyle: const TextStyle(color: Color(0xFF8B949E)),
              enabledBorder: const UnderlineInputBorder(
                borderSide: BorderSide(color: Color(0xFF3A86FF)),
              ),
              focusedBorder: const UnderlineInputBorder(
                borderSide: BorderSide(color: Color(0xFF3A86FF), width: 2),
              ),
            ),
            style: const TextStyle(color: Colors.white),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel',
                  style: TextStyle(color: Color(0xFF8B949E))),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(context, controller.text.trim()),
              style: FilledButton.styleFrom(
                  backgroundColor: const Color(0xFF3A86FF)),
              child: const Text('Connect'),
            ),
          ],
        ),
      ),
    );
  }

  String _serviceName(SyncService service) {
    switch (service) {
      case SyncService.notion:
        return 'Notion';
      case SyncService.googleCalendar:
        return 'Google Calendar';
      case SyncService.googleDrive:
        return 'Google Drive (Past Papers)';
      case SyncService.obsidian:
        return 'Obsidian';
    }
  }

  Color _serviceColor(SyncService service) {
    switch (service) {
      case SyncService.notion:
        return const Color(0xFF000000);
      case SyncService.googleCalendar:
        return const Color(0xFF4285F4);
      case SyncService.googleDrive:
        return const Color(0xFF0F9D58);
      case SyncService.obsidian:
        return const Color(0xFF7C3AED);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        backgroundColor: AxonColors.surface,
        body:
            const Center(child: RoseLoader(size: 24, color: Color(0xFF3A86FF))),
      );
    }

    return Scaffold(
      backgroundColor: AxonColors.surface,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: Text(
          'Sync Services',
          style: GoogleFonts.googleSans(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => popOrGo(context, '/settings'),
        ),
      ),
      body: ListView.separated(
        padding: const EdgeInsets.all(20),
        itemCount: SyncService.values.length,
        separatorBuilder: (_, __) => const SizedBox(height: 16),
        itemBuilder: (context, index) {
          final service = SyncService.values[index];
          final status = _statuses[service]!;
          return _buildServiceCard(service, status);
        },
      ),
    );
  }

  Widget _buildServiceCard(SyncService service, SyncStatus status) {
    final isConnected = status.isConnected;
    final lastSync = status.lastSync != null
        ? 'Last sync: ${_formatDate(status.lastSync!)}'
        : 'Never synced';

    return Container(
      decoration: BoxDecoration(
        color: AxonColors.cardSurface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color:
              isConnected ? const Color(0xFF3A86FF) : AxonColors.divider,
          width: isConnected ? 1.5 : 1.0,
        ),
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: _serviceColor(service).withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: SvgPicture.asset(
                    _serviceIconPath(service),
                    width: 28,
                    height: 28,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _serviceName(service),
                        style: GoogleFonts.googleSans(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        lastSync,
                        style: GoogleFonts.googleSans(
                          color: const Color(0xFF8B949E),
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Row(
              children: [
                Expanded(
                  child: isConnected
                      ? _ActionButton(
                          label: 'Disconnect',
                          icon: Icons.link_off,
                          onPressed: () => _disconnect(service),
                          isPrimary: false,
                        )
                      : _ActionButton(
                          label: 'Connect',
                          icon: Icons.link,
                          onPressed: () => _connect(service),
                          isPrimary: true,
                        ),
                ),
                if (isConnected) ...[
                  const SizedBox(width: 12),
                  Expanded(
                    child: _ActionButton(
                      label: 'Sync Now',
                      icon: Icons.sync,
                      onPressed: () => _syncNow(service),
                      isPrimary: true,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _serviceIconPath(SyncService service) {
    switch (service) {
      case SyncService.notion:
        return 'assets/icons/notion.svg';
      case SyncService.googleCalendar:
        return 'assets/icons/google_calendar.svg';
      case SyncService.googleDrive:
        return 'assets/icons/google_drive.svg';
      case SyncService.obsidian:
        return 'assets/icons/obsidian.svg';
    }
  }

  String _formatDate(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year} ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
  }
}

class SyncStatus {
  final bool isConnected;
  final DateTime? lastSync;

  SyncStatus({required this.isConnected, this.lastSync});
}

class _ActionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback onPressed;
  final bool isPrimary;

  const _ActionButton({
    required this.label,
    required this.icon,
    required this.onPressed,
    this.isPrimary = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 36,
      child: FilledButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 16),
        label: Text(label, style: const TextStyle(fontSize: 12)),
        style: FilledButton.styleFrom(
          backgroundColor:
              isPrimary ? const Color(0xFF3A86FF) : AxonColors.surfaceElevated,
          foregroundColor: isPrimary ? Colors.white : const Color(0xFF8B949E),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
            side: BorderSide(
              color: isPrimary ? Colors.transparent : const Color(0xFF3A86FF),
              width: 1.0,
            ),
          ),
        ),
      ),
    );
  }
}

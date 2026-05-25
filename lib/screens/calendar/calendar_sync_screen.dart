import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/google_calendar_service.dart';
import '../../theme/app_theme.dart';

/// Real-time Google Calendar sync screen.
/// Wires [GoogleCalendarService] with bidirectional sync and event stream.
class CalendarSyncScreen extends ConsumerStatefulWidget {
  const CalendarSyncScreen({super.key});

  @override
  ConsumerState<CalendarSyncScreen> createState() => _CalendarSyncScreenState();
}

class _CalendarSyncScreenState extends ConsumerState<CalendarSyncScreen> {
  final _gcal = GoogleCalendarService();
  StreamSubscription<List<CalendarEvent>>? _eventSub;
  List<CalendarEvent> _events = [];
  bool _isConnected = false;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    final connected = await _gcal.initialize();
    if (!mounted) return;

    if (connected) {
      _gcal.startRealtimeSync(
        intervalSeconds: 60,
        conflictResolution: ConflictResolution.latestWins,
      );
      _eventSub = _gcal.eventStream.listen((events) {
        if (mounted) setState(() => _events = events);
      });
    }

    setState(() {
      _isConnected = connected;
      _isLoading = false;
      _events = _gcal.events;
    });
  }

  Future<void> _addStudyEvent() async {
    final now = DateTime.now();
    await _gcal.upsertEvent(CalendarEvent(
      localId: '${DateTime.now().millisecondsSinceEpoch}',
      title: 'Study Block',
      description: 'Focus session',
      startTime: now,
      endTime: now.add(const Duration(hours: 1)),
      source: 'app',
      updatedAt: now,
    ));
  }

  Future<void> _removeEvent(CalendarEvent event) async {
    await _gcal.removeEvent(event.localId);
  }

  @override
  void dispose() {
    _eventSub?.cancel();
    _gcal.stopRealtimeSync();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          'Calendar Sync',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontWeight: FontWeight.w700,
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: _StatusIndicator(isConnected: _isConnected),
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _isConnected
              ? _buildEventList()
              : _buildConnectPrompt(),
    );
  }

  Widget _buildConnectPrompt() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.calendar_today, size: 64, color: AxonColors.textTertiary),
          const SizedBox(height: 20),
          Text(
            'Not connected to Google Calendar',
            style: GoogleFonts.googleSans(color: AxonColors.textSecondary),
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: () async {
              try {
                await _gcal.signIn();
                if (mounted) {
                  _gcal.startRealtimeSync(conflictResolution: ConflictResolution.latestWins);
                  _eventSub = _gcal.eventStream.listen((events) {
                    if (mounted) setState(() => _events = events);
                  });
                  setState(() {
                    _isConnected = true;
                    _events = _gcal.events;
                  });
                }
              } catch (_) {}
            },
            icon: const Icon(Icons.login),
            label: const Text('Sign in with Google'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF3A86FF),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(24),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEventList() {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          child: Row(
            children: [
              Text(
                '${_events.length} event${_events.length == 1 ? '' : 's'}',
                style: GoogleFonts.googleSans(
                  color: AxonColors.textSecondary,
                  fontSize: 13,
                ),
              ),
              const Spacer(),
              TextButton.icon(
                onPressed: _addStudyEvent,
                icon: const Icon(Icons.add, size: 18),
                label: const Text('Add Study Block'),
              ),
            ],
          ),
        ),
        Expanded(
          child: _events.isEmpty
              ? Center(
                  child: Text(
                    'No upcoming events. Tap "Add Study Block" to create one.',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary,
                      fontSize: 14,
                    ),
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  itemCount: _events.length,
                  itemBuilder: (context, i) {
                    final event = _events[i];
                    return _EventCard(
                      event: event,
                      onDelete: () => _removeEvent(event),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

class _StatusIndicator extends StatelessWidget {
  final bool isConnected;
  const _StatusIndicator({required this.isConnected});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8, height: 8,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isConnected ? const Color(0xFF4CAF50) : AxonColors.warning,
          ),
        ),
        const SizedBox(width: 6),
        Text(
          isConnected ? 'Live' : 'Offline',
          style: GoogleFonts.robotoMono(
            color: AxonColors.textSecondary,
            fontSize: 11,
          ),
        ),
      ],
    );
  }
}

class _EventCard extends StatelessWidget {
  final CalendarEvent event;
  final VoidCallback onDelete;

  const _EventCard({required this.event, required this.onDelete});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AxonColors.surfaceElevated,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  event.title,
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '${_formatTime(event.startTime)} — ${_formatTime(event.endTime)}',
                  style: GoogleFonts.jetBrainsMono(
                    color: AxonColors.textSecondary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: onDelete,
            icon: Icon(Icons.delete_outline, color: AxonColors.warning, size: 20),
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final h = dt.hour.toString().padLeft(2, '0');
    final m = dt.minute.toString().padLeft(2, '0');
    return '$h:$m';
  }
}

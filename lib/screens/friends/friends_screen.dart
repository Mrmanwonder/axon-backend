// lib/screens/friends/friends_screen.dart
// ignore_for_file: unused_element, unused_element_parameter

import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:go_router/go_router.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../services/haptics_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/axon_widgets.dart';

class StudyGroup {
  final String id;
  final String name;
  final String subject;
  final List<String> members;
  final String creatorId;
  final DateTime createdAt;
  final bool isPublic;

  StudyGroup({
    required this.id,
    required this.name,
    required this.subject,
    required this.members,
    required this.creatorId,
    required this.createdAt,
    this.isPublic = true,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'subject': subject,
        'members': members,
        'creatorId': creatorId,
        'createdAt': createdAt.toIso8601String(),
        'isPublic': isPublic,
      };

  factory StudyGroup.fromJson(Map<String, dynamic> json) => StudyGroup(
        id: json['id'] ?? '',
        name: json['name'] ?? '',
        subject: json['subject'] ?? '',
        members: List<String>.from(json['members'] ?? []),
        creatorId: json['creatorId'] ?? '',
        createdAt: DateTime.tryParse(json['createdAt'] ?? '') ?? DateTime.now(),
        isPublic: json['isPublic'] ?? true,
      );
}

class FriendsScreen extends StatefulWidget {
  const FriendsScreen({super.key});

  @override
  State<FriendsScreen> createState() => _FriendsScreenState();
}

class _FriendsScreenState extends State<FriendsScreen> {
  static final Guid _axonServiceUuid =
      Guid('8e400001-b5a3-f393-e0a9-e50e24dcca9e');
  static const String _friendsKey = 'axon_friends';
  static const String _studyGroupsKey = 'axon_study_groups';

  final List<_FriendPresence> _friends = [];
  final List<_NearbyPresence> _nearby = [];
  final List<StudyGroup> _studyGroups = [];
  StreamSubscription<List<ScanResult>>? _scanSub;
  bool _scanning = false;
  bool _permissionsGranted = false;
  int _tabIndex = 0;

  @override
  void initState() {
    super.initState();
    _loadFriends();
    _loadStudyGroups();
  }

  @override
  void dispose() {
    _scanSub?.cancel();
    FlutterBluePlus.stopScan();
    super.dispose();
  }

  Future<void> _loadFriends() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_friendsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      final restored = list
          .whereType<Map>()
          .map((e) => _FriendPresence.fromJson(Map<String, dynamic>.from(e)))
          .toList();
      if (!mounted) return;
      setState(() => _friends.addAll(restored));
    } catch (_) {}
  }

  Future<void> _saveFriends() async {
    final prefs = await SharedPreferences.getInstance();
    final data = jsonEncode(_friends.map((f) => f.toJson()).toList());
    await prefs.setString(_friendsKey, data);
  }

  Future<void> _loadStudyGroups() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_studyGroupsKey);
    if (raw == null || raw.isEmpty) return;
    try {
      final list = jsonDecode(raw) as List<dynamic>;
      final restored = list
          .whereType<Map>()
          .map((e) => StudyGroup.fromJson(Map<String, dynamic>.from(e)))
          .toList();
      if (!mounted) return;
      setState(() => _studyGroups.addAll(restored));
    } catch (_) {}
  }

  Future<void> _saveStudyGroups() async {
    final prefs = await SharedPreferences.getInstance();
    final data = jsonEncode(_studyGroups.map((g) => g.toJson()).toList());
    await prefs.setString(_studyGroupsKey, data);
  }

  Future<void> _createStudyGroup(String name, String subject) async {
    final group = StudyGroup(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: name,
      subject: subject,
      members: ['You'],
      creatorId: 'user',
      createdAt: DateTime.now(),
    );
    setState(() => _studyGroups.add(group));
    await _saveStudyGroups();
  }

  Future<void> _ensurePermissions() async {
    final results = await [
      Permission.bluetoothScan,
      Permission.bluetoothConnect,
      Permission.bluetoothAdvertise,
      Permission.locationWhenInUse,
    ].request();
    final ok = results.values.every((r) => r.isGranted);
    if (!mounted) return;
    setState(() => _permissionsGranted = ok);
  }

  Future<void> _startScan() async {
    await _ensurePermissions();
    if (!_permissionsGranted) return;
    await FlutterBluePlus.startScan(
      timeout: const Duration(seconds: 8),
      withServices: [_axonServiceUuid],
    );
    _scanSub?.cancel();
    _scanSub = FlutterBluePlus.scanResults.listen((results) {
      final next = <_NearbyPresence>[];
      for (final r in results) {
        final adv = r.advertisementData;
        if (!adv.serviceUuids.contains(_axonServiceUuid)) continue;
        if (r.rssi < -80) continue;
        final status = _decodeStatus(adv.serviceData[_axonServiceUuid]);
        next.add(_NearbyPresence(
          id: r.device.remoteId.str,
          name: adv.advName.isNotEmpty ? adv.advName : 'AXON_DEVICE',
          status: status,
          rssi: r.rssi,
        ));
      }
      if (!mounted) return;
      setState(() {
        _nearby
          ..clear()
          ..addAll(next);
      });
    });
    if (!mounted) return;
    setState(() => _scanning = true);
    await FlutterBluePlus.isScanning.where((s) => !s).first;
    if (!mounted) return;
    setState(() => _scanning = false);
  }

  String _decodeStatus(List<int>? payload) {
    if (payload == null || payload.isEmpty) return 'STATUS: UNKNOWN';
    try {
      final text = utf8.decode(payload).trim();
      if (text.isEmpty) return 'STATUS: UNKNOWN';
      return text.toUpperCase();
    } catch (_) {
      return 'STATUS: UNKNOWN';
    }
  }

  void _addFriendFromNearby(_NearbyPresence nearby) {
    final exists = _friends.any((f) => f.id == nearby.id);
    if (exists) return;
    final index = _friends.length + 1;
    final label = 'FRIEND_${index.toString().padLeft(2, '0')}';
    final friend = _FriendPresence(
      id: nearby.id,
      label: label,
      status: nearby.status,
      isStudying: nearby.status.contains('STUDY'),
      isBehindTarget:
          nearby.status.contains('AMBER') || nearby.status.contains('LAG'),
    );
    setState(() => _friends.add(friend));
    _saveFriends();
    AxonHaptics.selectionClick();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Added $label from local discovery')),
    );
  }

  void _sendSpark(_FriendPresence friend) {
    AxonHaptics.mediumImpact();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Spark sent to ${friend.label}')),
    );
  }

  void _joinSession(_FriendPresence friend) {
    context.push('/study');
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Race started with ${friend.label}')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          Container(
              decoration:
                  BoxDecoration(gradient: AxonGradients.backgroundGradient)),
          SafeArea(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
                  child: Row(
                    children: [
                      GestureDetector(
                        onTap: () => popOrGo(context, '/settings'),
                        child: Icon(Icons.arrow_back_rounded,
                            color: AxonColors.textSecondary, size: 22),
                      ),
                      const SizedBox(width: 12),
                      Text('Focus Lobby',
                          style: GoogleFonts.googleSans(
                              color: AxonColors.textPrimary,
                              fontSize: 20,
                              fontWeight: FontWeight.w700)),
                      const Spacer(),
                      GestureDetector(
                        onTap: () => context.push('/leaderboard'),
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: AxonColors.accent.withValues(alpha: 0.08),
                            borderRadius: BorderRadius.circular(10),
                            border: Border.all(
                                color:
                                    AxonColors.accent.withValues(alpha: 0.25)),
                          ),
                          child: Icon(Icons.leaderboard_rounded,
                              color: AxonColors.accent, size: 20),
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 2, 20, 12),
                  child: Text(
                    'No feed. Just focus signals and proximity discovery.',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary, fontSize: 12),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
                  child: _SegmentBar(
                    leftLabel: 'Lobby',
                    rightLabel: 'Study Groups',
                    index: _tabIndex,
                    onChanged: (i) => setState(() => _tabIndex = i),
                  ),
                ),
                Expanded(
                  child: _tabIndex == 0 ? _buildLobby() : _buildStudyGroups(),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLobby() {
    if (_friends.isEmpty) {
      return Center(
        child: AxonCard(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.group_outlined,
                  color: AxonColors.electricCyan, size: 28),
              const SizedBox(height: 10),
              Text('No friends yet',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w600)),
              const SizedBox(height: 6),
              Text('Add friends via Local Discovery to see their focus status.',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary, fontSize: 12)),
            ],
          ),
        ),
      );
    }
    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
      itemCount: _friends.length,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        final friend = _friends[index];
        return _FriendLogRow(
          friend: friend,
          onSpark: () => _sendSpark(friend),
          onJoin: () => _joinSession(friend),
        );
      },
    );
  }

  Widget _buildDiscovery() {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 4, 20, 12),
          child: Row(
            children: [
              Icon(Icons.bluetooth_rounded,
                  color: AxonColors.textTertiary, size: 14),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  _permissionsGranted
                      ? 'BLE discovery is local-only. Only nearby Axon devices appear.'
                      : 'Bluetooth permissions required to discover nearby peers.',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary, fontSize: 12),
                ),
              ),
              const SizedBox(width: 12),
              CyberButton(
                label: _scanning ? 'Scanningâ€' : 'Scan',
                onTap: _scanning ? null : _startScan,
                icon: Icons.radar_rounded,
                
                fullWidth: false,
                showShadow: false,
              ),
            ],
          ),
        ),
        Expanded(
          child: _nearby.isEmpty
              ? Center(
                  child: AxonCard(
                    padding: const EdgeInsets.all(18),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.sensors_rounded,
                            color: AxonColors.electricCyan, size: 26),
                        const SizedBox(height: 10),
                        Text('No nearby Axon devices',
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textPrimary,
                                fontSize: 15,
                                fontWeight: FontWeight.w600)),
                        const SizedBox(height: 6),
                        Text(
                            'Ask classmates to open Axon and enable discovery.',
                            textAlign: TextAlign.center,
                            style: GoogleFonts.googleSans(
                                color: AxonColors.textTertiary, fontSize: 12)),
                      ],
                    ),
                  ),
                )
              : ListView.separated(
                  padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
                  itemCount: _nearby.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, index) {
                    final device = _nearby[index];
                    return _NearbyRow(
                      nearby: device,
                      onAdd: () => _addFriendFromNearby(device),
                    );
                  },
                ),
        ),
      ],
    );
  }

  Widget _buildStudyGroups() {
    if (_studyGroups.isEmpty) {
      return Center(
        child: AxonCard(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.groups_rounded,
                  color: AxonColors.electricCyan, size: 40),
              const SizedBox(height: 12),
              Text('No Study Groups Yet',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 16,
                      fontWeight: FontWeight.w600)),
              const SizedBox(height: 8),
              Text('Create a group to study with friends',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary, fontSize: 12)),
              const SizedBox(height: 16),
              CyberButton(
                label: 'Create Group',
                onTap: () => _showCreateGroupDialog(),
                icon: Icons.add_rounded,
                fullWidth: false,
              ),
            ],
          ),
        ),
      );
    }

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 20),
      itemCount: _studyGroups.length + 1,
      separatorBuilder: (_, __) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        if (index == _studyGroups.length) {
          return Center(
            child: CyberButton(
              label: 'Create New Group',
              onTap: () => _showCreateGroupDialog(),
              icon: Icons.add_rounded,
              fullWidth: false,
            ),
          );
        }
        final group = _studyGroups[index];
        return _StudyGroupCard(group: group);
      },
    );
  }

  void _showCreateGroupDialog() {
    final nameController = TextEditingController();
    String selectedSubject = 'Physics';
    final subjects = [
      'Physics',
      'Chemistry',
      'Mathematics',
      'Biology',
      'Computer Science',
      'English'
    ];

    showModalBottomSheet(
      context: context,
      backgroundColor: AxonColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => StatefulBuilder(
        builder: (context, setModalState) => Padding(
          padding: EdgeInsets.fromLTRB(
              20, 20, 20, MediaQuery.of(context).viewInsets.bottom + 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Create Study Group',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w600)),
              const SizedBox(height: 20),
              TextField(
                controller: nameController,
                style: GoogleFonts.googleSans(color: AxonColors.textPrimary),
                decoration: InputDecoration(
                  labelText: 'Group Name',
                  labelStyle:
                      GoogleFonts.googleSans(color: AxonColors.textTertiary),
                  enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: AxonColors.divider)),
                  focusedBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: AxonColors.electricCyan)),
                ),
              ),
              const SizedBox(height: 16),
              Text('Subject',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary, fontSize: 12)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: subjects
                    .map((s) => ChoiceChip(
                          label: Text(s),
                          selected: selectedSubject == s,
                          selectedColor:
                              AxonColors.electricCyan.withValues(alpha: 0.2),
                          onSelected: (_) =>
                              setModalState(() => selectedSubject = s),
                        ))
                    .toList(),
              ),
              const SizedBox(height: 20),
              SizedBox(
                width: double.infinity,
                child: CyberButton(
                  label: 'Create',
                  onTap: () {
                    if (nameController.text.trim().isNotEmpty) {
                      _createStudyGroup(
                          nameController.text.trim(), selectedSubject);
                      Navigator.pop(ctx);
                    }
                  },
                  fullWidth: true,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _StudyGroupCard extends StatelessWidget {
  final StudyGroup group;
  const _StudyGroupCard({required this.group});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AxonColors.divider),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AxonColors.electricCyan.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(Icons.groups_rounded, color: AxonColors.electricCyan),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(group.name,
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text('${group.subject} • ${group.members.length} members',
                    style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary, fontSize: 12)),
              ],
            ),
          ),
          Icon(Icons.chevron_right_rounded, color: AxonColors.textTertiary),
        ],
      ),
    );
  }
}

class _SegmentBar extends StatelessWidget {
  final String leftLabel;
  final String rightLabel;
  final String? middleLabel;
  final int index;
  final ValueChanged<int> onChanged;
  const _SegmentBar({
    required this.leftLabel,
    required this.rightLabel,
    this.middleLabel,
    required this.index,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final hasMiddle = middleLabel != null;
    return Container(
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: AxonColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AxonColors.divider),
      ),
      child: Row(
        children: [
          _SegmentOption(
            label: leftLabel,
            selected: index == 0,
            onTap: () => onChanged(0),
          ),
          if (hasMiddle) ...[
            const SizedBox(width: 6),
            _SegmentOption(
              label: middleLabel!,
              selected: index == 1,
              onTap: () => onChanged(1),
            ),
          ],
          const SizedBox(width: 6),
          _SegmentOption(
            label: rightLabel,
            selected: hasMiddle ? index == 2 : index == 1,
            onTap: () => onChanged(hasMiddle ? 2 : 1),
          ),
        ],
      ),
    );
  }
}

class _SegmentOption extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _SegmentOption(
      {required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected
                ? AxonColors.electricCyan.withValues(alpha: 0.12)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: selected ? AxonColors.electricCyan : Colors.transparent,
              width: 1,
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: GoogleFonts.googleSans(
                color: selected
                    ? AxonColors.electricCyan
                    : AxonColors.textSecondary,
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.6,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _FriendLogRow extends StatelessWidget {
  final _FriendPresence friend;
  final VoidCallback onSpark;
  final VoidCallback onJoin;
  const _FriendLogRow({
    required this.friend,
    required this.onSpark,
    required this.onJoin,
  });

  @override
  Widget build(BuildContext context) {
    final statusColor = friend.isStudying
        ? AxonColors.electricCyan
        : (friend.isBehindTarget ? AxonColors.poor : AxonColors.textTertiary);
    return AxonCard(
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${friend.label}:',
            style: GoogleFonts.googleSans(
              color: friend.isStudying
                  ? AxonColors.electricCyan
                  : AxonColors.textSecondary,
              fontSize: 13,
              fontWeight: FontWeight.w700,
              letterSpacing: 1.2,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            friend.status,
            style: GoogleFonts.googleSans(
              color: statusColor,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              OutlinedButton.icon(
                onPressed: onSpark,
                icon: const Icon(Icons.flash_on_rounded, size: 16),
                label: const Text('Spark'),
              ),
              const SizedBox(width: 10),
              OutlinedButton.icon(
                onPressed: onJoin,
                icon: const Icon(Icons.timer_rounded, size: 16),
                label: const Text('Join Session'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _NearbyRow extends StatelessWidget {
  final _NearbyPresence nearby;
  final VoidCallback onAdd;
  const _NearbyRow({required this.nearby, required this.onAdd});

  @override
  Widget build(BuildContext context) {
    return AxonCard(
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  nearby.name,
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  nearby.status,
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textTertiary,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'RSSI ${nearby.rssi} dBm',
                  style: GoogleFonts.googleSans(
                      color: AxonColors.textTertiary, fontSize: 10),
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: onAdd,
            icon: const Icon(Icons.person_add_alt_1_rounded, size: 20),
            color: AxonColors.electricCyan,
          ),
        ],
      ),
    );
  }
}

class _FriendPresence {
  final String id;
  final String label;
  final String status;
  final bool isStudying;
  final bool isBehindTarget;
  const _FriendPresence({
    required this.id,
    required this.label,
    required this.status,
    required this.isStudying,
    required this.isBehindTarget,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'label': label,
        'status': status,
        'isStudying': isStudying,
        'isBehindTarget': isBehindTarget,
      };

  factory _FriendPresence.fromJson(Map<String, dynamic> json) {
    return _FriendPresence(
      id: json['id']?.toString() ?? '',
      label: json['label']?.toString() ?? 'FRIEND',
      status: json['status']?.toString() ?? 'STATUS: UNKNOWN',
      isStudying: json['isStudying'] == true,
      isBehindTarget: json['isBehindTarget'] == true,
    );
  }
}

class _NearbyPresence {
  final String id;
  final String name;
  final String status;
  final int rssi;
  const _NearbyPresence({
    required this.id,
    required this.name,
    required this.status,
    required this.rssi,
  });
}

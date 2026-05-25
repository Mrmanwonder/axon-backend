import 'dart:async';
import 'dart:io';
import 'package:cached_network_image/cached_network_image.dart';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import '../../services/app_state.dart';
import '../../theme/app_theme.dart';
import '../../utils/nav_utils.dart';
import '../../widgets/common/axon_widgets.dart';
import '../../widgets/common/rose_loader.dart';

class EditAccountScreen extends ConsumerStatefulWidget {
  const EditAccountScreen({super.key});

  @override
  ConsumerState<EditAccountScreen> createState() => _EditAccountScreenState();
}

class _EditAccountScreenState extends ConsumerState<EditAccountScreen> {
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();

  String? _error;
  bool _saving = false;
  bool _uploadingPhoto = false;
  String? _photoUrl;
  bool _seededProfile = false;
  String _lastSavedName = '';
  Timer? _autoSaveDebounce;

  @override
  void initState() {
    super.initState();
    _nameCtrl.addListener(_scheduleAutoSave);
  }

  @override
  void dispose() {
    _autoSaveDebounce?.cancel();
    _nameCtrl.removeListener(_scheduleAutoSave);
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    super.dispose();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final user = ref.read(authStateProvider).user;
    if (user != null && !_seededProfile) {
      _nameCtrl.text = user.displayName;
      _emailCtrl.text = user.email;
      _photoUrl = user.photoUrl;
      _lastSavedName = user.displayName;
      _seededProfile = true;
    }
  }

  void _scheduleAutoSave() {
    if (!_seededProfile) return;
    _autoSaveDebounce?.cancel();
    _autoSaveDebounce = Timer(const Duration(milliseconds: 500), () {
      unawaited(_saveSilently());
    });
  }

  Future<void> _changePhoto(ImageSource source) async {
    Navigator.of(context).pop();
    setState(() {
      _uploadingPhoto = true;
      _error = null;
    });
    try {
      final path = await ref
          .read(authStateProvider.notifier)
          .pickAndUploadProfilePhoto(source);
      if (!mounted) return;
      if (path != null && path.isNotEmpty) {
        setState(() {
          _photoUrl = path;
        });
      } else {
        setState(() => _error = 'Unable to update profile photo.');
      }
    } catch (_) {
      if (!mounted) return;
      setState(() => _error = 'Unable to update profile photo.');
    } finally {
      if (mounted) {
        setState(() => _uploadingPhoto = false);
      }
    }
  }

  Future<void> _removePhoto() async {
    Navigator.of(context).pop();
    try {
      await ref.read(authStateProvider.notifier).removeProfilePhoto();
      if (mounted) {
        setState(() => _photoUrl = null);
      }
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Unable to remove profile photo.');
      }
    }
  }

  Future<void> _showPhotoOptions() async {
    await showModalBottomSheet<void>(
      context: context,
      backgroundColor: AxonColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        return SafeArea(
          top: false,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Profile Photo',
                  style: GoogleFonts.googleSans(
                    color: AxonColors.textPrimary,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 16),
                _PhotoOptionTile(
                  icon: Icons.photo_library_outlined,
                  label: 'Choose from gallery',
                  onTap: () => _changePhoto(ImageSource.gallery),
                ),
                const SizedBox(height: 8),
                _PhotoOptionTile(
                  icon: Icons.photo_camera_outlined,
                  label: 'Take a photo',
                  onTap: () => _changePhoto(ImageSource.camera),
                ),
                const SizedBox(height: 8),
                if (_photoUrl != null && _photoUrl!.isNotEmpty)
                  _PhotoOptionTile(
                    icon: Icons.delete_outline,
                    label: 'Remove photo',
                    isDestructive: true,
                    onTap: () => _removePhoto(),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _saveSilently() async {
    final name = _nameCtrl.text.trim();
    if (name.isEmpty || name == _lastSavedName || _saving) {
      return;
    }

    setState(() {
      _saving = true;
      _error = null;
    });

    await ref.read(authStateProvider.notifier).updateProfile(
          displayName: name,
          photoUrl: _photoUrl,
        );

    if (!mounted) return;
    setState(() {
      _saving = false;
      _lastSavedName = name;
    });
  }

  @override
  Widget build(BuildContext context) {
    final user = ref.watch(authStateProvider).user;
    final bottomInset = MediaQuery.of(context).padding.bottom;

    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Container(
        color: AxonColors.surface,
        child: SafeArea(
          child: SingleChildScrollView(
            padding: EdgeInsets.fromLTRB(20, 16, 20, 128 + bottomInset),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    GestureDetector(
                      onTap: () => popOrGo(context, '/settings'),
                      child: Icon(
                        Icons.arrow_back_rounded,
                        color: AxonColors.textSecondary,
                        size: 22,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'Edit Account',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textPrimary,
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: AppSpacing.m),
                if (user == null)
                  AxonCard(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      'You must be signed in to edit your account.',
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 12,
                      ),
                    ),
                  )
                else
                  AxonCard(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Profile Details',
                          style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: AppSpacing.s),
                        Center(
                          child: GestureDetector(
                            onTap: _uploadingPhoto ? null : _showPhotoOptions,
                            child: Container(
                              width: 80,
                              height: 80,
                              clipBehavior: Clip.hardEdge,
                              decoration: BoxDecoration(
                                color:
                                    AxonColors.accent.withValues(alpha: 0.12),
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color:
                                      AxonColors.accent.withValues(alpha: 0.2),
                                ),
                              ),
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  if (_photoUrl != null &&
                                      _photoUrl!.isNotEmpty)
                                    _ProfilePhotoPreview(photoUrl: _photoUrl!)
                                  else
                                    Center(
                                      child: Icon(
                                        Icons.camera_alt_rounded,
                                        color: AxonColors.accent,
                                        size: 28,
                                      ),
                                    ),
                                  if (_uploadingPhoto)
                                    Container(
                                      color:
                                          Colors.black.withValues(alpha: 0.35),
                                      child: const Center(
                                        child: RoseLoader(size: 22),
                                      ),
                                    ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: AppSpacing.s),
                        TextField(
                          controller: _nameCtrl,
                          style: GoogleFonts.googleSans(
                            color: AxonColors.textPrimary,
                          ),
                          decoration: InputDecoration(
                            labelText: 'Full Name',
                            prefixIcon: Icon(
                              Icons.person_outline_rounded,
                              color: AxonColors.textTertiary,
                              size: 18,
                            ),
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          _saving
                              ? 'Saving changes...'
                              : 'Changes save automatically',
                          style: GoogleFonts.googleSans(
                            color: AxonColors.textTertiary,
                            fontSize: 11,
                          ),
                        ),
                        const SizedBox(height: AppSpacing.s),
                        TextField(
                          controller: _emailCtrl,
                          readOnly: true,
                          enabled: false,
                          style: GoogleFonts.googleSans(
                            color: AxonColors.textSecondary,
                          ),
                          decoration: InputDecoration(
                            labelText: 'Email',
                            prefixIcon: Icon(
                              Icons.email_outlined,
                              color: AxonColors.textTertiary,
                              size: 18,
                            ),
                          ),
                        ),
                        if (_error != null) ...[
                          const SizedBox(height: AppSpacing.s),
                          Text(
                            _error!,
                            style: GoogleFonts.googleSans(
                              color: AxonColors.error,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _PhotoOptionTile extends StatelessWidget {
  const _PhotoOptionTile({
    required this.icon,
    required this.label,
    required this.onTap,
    this.isDestructive = false,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool isDestructive;

  @override
  Widget build(BuildContext context) {
    final color = isDestructive ? Colors.red : AxonColors.textPrimary;
    return InkWell(
      borderRadius: BorderRadius.circular(16),
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        decoration: BoxDecoration(
          color: isDestructive
              ? Colors.red.withValues(alpha: 0.1)
              : Colors.white.withValues(alpha: 0.03),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isDestructive
                ? Colors.red.withValues(alpha: 0.3)
                : Colors.white.withValues(alpha: 0.08),
          ),
        ),
        child: Row(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 12),
            Text(
              label,
              style: GoogleFonts.googleSans(
                color: color,
                fontSize: 14,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ProfilePhotoPreview extends StatelessWidget {
  const _ProfilePhotoPreview({required this.photoUrl});

  final String photoUrl;

  @override
  Widget build(BuildContext context) {
    if (photoUrl.startsWith('http://') || photoUrl.startsWith('https://')) {
      return CachedNetworkImage(
        imageUrl: photoUrl,
        fit: BoxFit.cover,
        errorWidget: (_, __, ___) => _fallback(),
      );
    }

    final file = File(photoUrl);
    return Image.file(
      file,
      fit: BoxFit.cover,
      errorBuilder: (_, __, ___) => _fallback(),
    );
  }

  Widget _fallback() {
    return Center(
      child: Icon(
        Icons.camera_alt_rounded,
        color: AxonColors.accent,
        size: 28,
      ),
    );
  }
}

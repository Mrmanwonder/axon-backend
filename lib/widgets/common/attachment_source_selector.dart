import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme/app_theme.dart';

class AttachmentSourceTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const AttachmentSourceTile({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AxonColors.background,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AxonColors.accent.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: AxonColors.accent, size: 24),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      title,
                      style: GoogleFonts.googleSans(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: AxonColors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: GoogleFonts.googleSans(
                        fontSize: 13,
                        color: AxonColors.textTertiary,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.chevron_right_rounded, color: AxonColors.textTertiary),
            ],
          ),
        ),
      ),
    );
  }
}

Future<String?> showAttachmentSourceSelector(BuildContext context) {
  return showModalBottomSheet<String>(
    context: context,
    backgroundColor: Colors.transparent,
    isScrollControlled: true,
    builder: (context) => Padding(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
      ),
      child: Container(
        margin: const EdgeInsets.all(16),
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 20),
        decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: SafeArea(
          top: false,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: AxonColors.textTertiary,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              Text(
                'Add Content',
                style: GoogleFonts.googleSans(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: AxonColors.textPrimary,
                ),
              ),
              const SizedBox(height: 20),
              AttachmentSourceTile(
                icon: Icons.upload_file_rounded,
                title: 'Local Files',
                subtitle: 'Pick PDFs, documents',
                onTap: () => Navigator.pop(context, 'local'),
              ),
              const SizedBox(height: 12),
              AttachmentSourceTile(
                icon: Icons.cloud_rounded,
                title: 'Google Drive',
                subtitle: 'Search & import from Drive',
                onTap: () => Navigator.pop(context, 'drive'),
              ),
              const SizedBox(height: 12),
              AttachmentSourceTile(
                icon: Icons.search_rounded,
                title: 'Search Online',
                subtitle: 'Find past papers via Serper',
                onTap: () => Navigator.pop(context, 'search'),
              ),
              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    ),
  );
}

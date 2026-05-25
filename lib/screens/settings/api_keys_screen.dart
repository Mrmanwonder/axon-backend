// lib/screens/settings/api_keys_screen.dart
//
// API Keys Management Screen
// Securely store and manage all API keys locally
// Uses flutter_secure_storage for encrypted credential storage

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../services/secure_credentials_service.dart';
import '../../theme/app_theme.dart';

class ApiKeysScreen extends ConsumerStatefulWidget {
  const ApiKeysScreen({super.key});

  @override
  ConsumerState<ApiKeysScreen> createState() => _ApiKeysScreenState();
}

class _ApiKeysScreenState extends ConsumerState<ApiKeysScreen> {
  final _formKey = GlobalKey<FormState>();

  final _supabaseUrlController = TextEditingController();
  final _supabaseAnonKeyController = TextEditingController();
  final _geminiApiKeyController = TextEditingController();
  final _serperApiKeyController = TextEditingController();
  final _cloudinaryCloudController = TextEditingController();
  final _cloudinaryPresetController = TextEditingController();
  final _grokApiKeyController = TextEditingController();
  final _vercelApiKeyController = TextEditingController();
  final _deepseekApiKeyController = TextEditingController();
  final _googleDriveKeyController = TextEditingController();

  bool _isLoading = true;
  bool _isSaving = false;

  @override
  void initState() {
    super.initState();
    _loadCredentials();
  }

  @override
  void dispose() {
    _supabaseUrlController.dispose();
    _supabaseAnonKeyController.dispose();
    _geminiApiKeyController.dispose();
    _serperApiKeyController.dispose();
    _cloudinaryCloudController.dispose();
    _cloudinaryPresetController.dispose();
    _grokApiKeyController.dispose();
    _vercelApiKeyController.dispose();
    _deepseekApiKeyController.dispose();
    _googleDriveKeyController.dispose();
    super.dispose();
  }

  Future<void> _loadCredentials() async {
    final credentials = SecureCredentialsService();
    await credentials.initialize();
    final creds = await credentials.getAllCredentials();

    if (mounted) {
      setState(() {
        if (creds.effectiveSupabaseUrl != null) {
          _supabaseUrlController.text = creds.effectiveSupabaseUrl!;
        }
        if (creds.effectiveSupabaseAnonKey != null) {
          _supabaseAnonKeyController.text = creds.effectiveSupabaseAnonKey!;
        }
        if (creds.effectiveGeminiKey != null) {
          _geminiApiKeyController.text = creds.effectiveGeminiKey!;
        }
        if (creds.effectiveSerperKey != null) {
          _serperApiKeyController.text = creds.effectiveSerperKey!;
        }
        if (creds.effectiveCloudinaryCloud != null) {
          _cloudinaryCloudController.text = creds.effectiveCloudinaryCloud!;
        }
        if (creds.effectiveCloudinaryPreset != null) {
          _cloudinaryPresetController.text = creds.effectiveCloudinaryPreset!;
        }
        if (creds.effectiveGrokKey != null) {
          _grokApiKeyController.text = creds.effectiveGrokKey!;
        }
        if (creds.effectiveVercelKey != null) {
          _vercelApiKeyController.text = creds.effectiveVercelKey!;
        }
        if (creds.effectiveDeepseekKey != null) {
          _deepseekApiKeyController.text = creds.effectiveDeepseekKey!;
        }
        if (creds.effectiveGoogleDriveKey != null) {
          _googleDriveKeyController.text = creds.effectiveGoogleDriveKey!;
        }
        _isLoading = false;
      });
    }
  }

  Future<void> _saveCredentials() async {
    if (_isSaving) return;

    setState(() => _isSaving = true);

    try {
      final credentials = SecureCredentialsService();
      await credentials.initialize();

      final creds = ApiCredentials(
        supabaseUrl: _supabaseUrlController.text.trim(),
        supabaseAnonKey: _supabaseAnonKeyController.text.trim(),
        geminiApiKey: _geminiApiKeyController.text.trim(),
        serperApiKey: _serperApiKeyController.text.trim(),
        cloudinaryCloud: _cloudinaryCloudController.text.trim(),
        cloudinaryPreset: _cloudinaryPresetController.text.trim(),
        grokApiKey: _grokApiKeyController.text.trim(),
        vercelApiKey: _vercelApiKeyController.text.trim(),
        deepseekApiKey: _deepseekApiKeyController.text.trim(),
        googleDrivePrivateKey: _googleDriveKeyController.text.trim(),
      );

      await credentials.storeAllCredentials(creds);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('API keys saved securely'),
            backgroundColor: Colors.green.shade700,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to save: $e'),
            backgroundColor: Colors.red.shade700,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  Future<void> _clearAllCredentials() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear All API Keys?'),
        content: const Text(
          'This will remove all stored API keys. You will need to re-enter them to use AI features.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Clear All'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      final credentials = SecureCredentialsService();
      await credentials.clearAllCredentials();

      _supabaseUrlController.clear();
      _supabaseAnonKeyController.clear();
      _geminiApiKeyController.clear();
      _serperApiKeyController.clear();
      _cloudinaryCloudController.clear();
      _cloudinaryPresetController.clear();
      _grokApiKeyController.clear();
      _vercelApiKeyController.clear();
      _deepseekApiKeyController.clear();
      _googleDriveKeyController.clear();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('All API keys cleared'),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    }
  }

  Widget _buildKeyField({
    required TextEditingController controller,
    required String label,
    required String hint,
    String? helperText,
    bool isPassword = false,
    bool showVisibilityToggle = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: GoogleFonts.googleSans(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AxonColors.textSecondary,
            ),
          ),
          const SizedBox(height: 6),
          TextFormField(
            controller: controller,
            obscureText: isPassword && !showVisibilityToggle,
            style: GoogleFonts.googleSans(
              fontSize: 14,
              color: AxonColors.textPrimary,
            ),
            decoration: InputDecoration(
              hintText: hint,
              hintStyle: TextStyle(
                color: AxonColors.textSecondary.withValues(alpha: 0.5),
                fontSize: 13,
              ),
              filled: true,
              fillColor: AxonColors.surfaceSecondary,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: BorderSide.none,
              ),
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 12,
              ),
              suffixIcon: isPassword
                  ? IconButton(
                      icon: Icon(
                        showVisibilityToggle
                            ? Icons.visibility_off
                            : Icons.visibility,
                        size: 20,
                        color: AxonColors.textSecondary,
                      ),
                      onPressed: () {
                        setState(() => showVisibilityToggle = !showVisibilityToggle);
                      },
                    )
                  : null,
            ),
          ),
          if (helperText != null)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                helperText,
                style: TextStyle(
                  fontSize: 11,
                  color: AxonColors.textSecondary.withValues(alpha: 0.7),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(top: 24, bottom: 12),
      child: Text(
        title,
        style: GoogleFonts.googleSans(
          fontSize: 16,
          fontWeight: FontWeight.w700,
          color: AxonColors.textPrimary,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.backgroundPrimary,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          'API Keys',
          style: GoogleFonts.googleSans(
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        actions: [
          if (_isSaving)
            const Padding(
              padding: EdgeInsets.only(right: 16),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            TextButton(
              onPressed: _saveCredentials,
              child: Text(
                'Save',
                style: GoogleFonts.googleSans(
                  color: AxonColors.accentPrimary,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Info banner
                    Container(
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: AxonColors.accentPrimary.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: AxonColors.accentPrimary.withValues(alpha: 0.3),
                        ),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            Icons.security,
                            color: AxonColors.accentPrimary,
                            size: 22,
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              'Keys are stored locally with encryption. Never sent to any server.',
                              style: GoogleFonts.googleSans(
                                fontSize: 12,
                                color: AxonColors.textSecondary,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),

                    // AI Services Section
                    _buildSectionHeader('AI Services'),

                    _buildKeyField(
                      controller: _geminiApiKeyController,
                      label: 'Gemini API Key',
                      hint: 'Enter your Google Gemini API key',
                      helperText: 'Get from: https://makersuite.google.com/app/apikey',
                      isPassword: true,
                    ),

                    _buildKeyField(
                      controller: _grokApiKeyController,
                      label: 'Grok API Key',
                      hint: 'Enter your xAI Grok API key',
                      helperText: 'Get from: https://console.x.ai/',
                      isPassword: true,
                    ),

                    _buildKeyField(
                      controller: _deepseekApiKeyController,
                      label: 'DeepSeek API Key',
                      hint: 'Enter your DeepSeek API key',
                      isPassword: true,
                    ),

                    _buildKeyField(
                      controller: _vercelApiKeyController,
                      label: 'Vercel AI API Key',
                      hint: 'Fallback when Grok is exhausted',
                      isPassword: true,
                    ),

                    _buildKeyField(
                      controller: _serperApiKeyController,
                      label: 'Serper API Key (Search)',
                      hint: 'For web search functionality',
                      helperText: 'Get from: https://serper.dev/',
                      isPassword: true,
                    ),

                    // Database & Storage Section
                    _buildSectionHeader('Database & Storage'),

                    _buildKeyField(
                      controller: _supabaseUrlController,
                      label: 'Supabase URL',
                      hint: 'https://your-project.supabase.co',
                    ),

                    _buildKeyField(
                      controller: _supabaseAnonKeyController,
                      label: 'Supabase Anonymous Key',
                      hint: 'Enter your Supabase anon key',
                      isPassword: true,
                    ),

                    _buildKeyField(
                      controller: _cloudinaryCloudController,
                      label: 'Cloudinary Cloud Name',
                      hint: 'e.g., mycloud',
                      helperText: 'Get from: Cloudinary Dashboard > Settings',
                    ),

                    _buildKeyField(
                      controller: _cloudinaryPresetController,
                      label: 'Cloudinary Upload Preset',
                      hint: 'e.g., axon_presets',
                    ),

                    // Google Drive Section
                    _buildSectionHeader('Google Drive'),

                    _buildKeyField(
                      controller: _googleDriveKeyController,
                      label: 'Service Account Private Key',
                      hint: 'Paste the entire private key including BEGIN/END markers',
                      helperText:
                          'For downloading past papers and Gemma model. Get from Google Cloud Console > IAM > Service Accounts',
                      isPassword: true,
                    ),

                    const SizedBox(height: 32),

                    // Clear All Button
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: _clearAllCredentials,
                        icon: const Icon(Icons.delete_outline, size: 20),
                        label: Text(
                          'Clear All API Keys',
                          style: GoogleFonts.googleSans(
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.red.shade400,
                          side: BorderSide(color: Colors.red.shade300),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                      ),
                    ),

                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
    );
  }
}
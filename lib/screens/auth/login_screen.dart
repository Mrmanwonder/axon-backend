// lib/screens/auth/login_screen.dart
import 'dart:ui';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../services/app_state.dart';
import '../../services/input_validation_service.dart';
import '../../theme/app_theme.dart';
import 'register_screen.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({super.key});
  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  String? _error;
  bool _isEmailFocused = false;
  bool _isPasswordFocused = false;

  bool get _supportsGoogleSignIn =>
      !kIsWeb && defaultTargetPlatform != TargetPlatform.windows;

  @override
  void initState() {
    super.initState();
    _emailCtrl.addListener(_onEmailChanged);
    _passCtrl.addListener(_onPasswordChanged);
  }

  @override
  void dispose() {
    _emailCtrl.removeListener(_onEmailChanged);
    _passCtrl.removeListener(_onPasswordChanged);
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  void _onEmailChanged() {}

  void _onPasswordChanged() {}

  Future<void> _showPasswordResetDialog(BuildContext context) async {
    final emailCtrl = TextEditingController();
    final authNotifier = ref.read(authStateProvider.notifier);

    final result = await showDialog<bool>(
      context: context,
      barrierColor: Colors.black54,
      builder: (ctx) => AlertDialog(
        backgroundColor: AxonColors.surface,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: BorderSide(color: AxonColors.divider, width: 1),
        ),
        title: Text(
          'RECOVERY_PROTOCOL',
          style: GoogleFonts.googleSans(
            color: AxonColors.textPrimary,
            fontSize: 16,
            fontWeight: FontWeight.w700,
            letterSpacing: 1,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Enter your email to receive a password reset link.',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontSize: 13,
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: AxonColors.surfaceElevated,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AxonColors.divider),
              ),
              child: TextField(
                controller: emailCtrl,
                style: GoogleFonts.googleSans(color: AxonColors.textPrimary),
                decoration: InputDecoration(
                  hintText: 'your@email.com',
                  hintStyle:
                      GoogleFonts.googleSans(color: AxonColors.textTertiary),
                  border: InputBorder.none,
                  isDense: true,
                  contentPadding: EdgeInsets.zero,
                ),
                keyboardType: TextInputType.emailAddress,
                textInputAction: TextInputAction.done,
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(
              'CANCEL',
              style: GoogleFonts.googleSans(
                color: AxonColors.textSecondary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const SizedBox(width: 8),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(
              'SEND',
              style: GoogleFonts.googleSans(
                color: AxonColors.accent,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );

    if (result == true && emailCtrl.text.trim().isNotEmpty) {
      try {
        final success = await authNotifier.resetPassword(emailCtrl.text.trim());
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                success
                    ? 'Password reset link sent to ${emailCtrl.text.trim()}'
                    : 'Failed to send reset email',
                style: GoogleFonts.googleSans(color: Colors.white),
              ),
              backgroundColor: (success ? AxonColors.success : AxonColors.error)
                  .withValues(alpha: 0.9),
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          );
        }
      } catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Failed to send reset email',
                style: GoogleFonts.googleSans(color: Colors.white),
              ),
              backgroundColor: AxonColors.error.withValues(alpha: 0.9),
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          );
        }
      }
    }
    emailCtrl.dispose();
  }

  Future<void> _login() async {
    final validation = InputValidationService();
    final emailResult = validation.validateEmail(_emailCtrl.text);
    if (!emailResult.isValid) {
      setState(() => _error = emailResult.error);
      return;
    }
    final passResult = validation.validatePassword(_passCtrl.text);
    if (!passResult.isValid) {
      setState(() => _error = passResult.error);
      return;
    }
    setState(() => _error = null);
    final ok = await ref
        .read(authStateProvider.notifier)
        .signInWithEmail(_emailCtrl.text.trim(), _passCtrl.text);
    if (!ok && mounted) {
      setState(() {
        _error =
            ref.read(authStateProvider).error ?? 'Invalid email or password.';
      });
    } else if (ok && mounted) {
      context.go('/home');
    }
  }

  Future<void> _loginWithGoogle() async {
    setState(() => _error = null);

    final ok = await ref.read(authStateProvider.notifier).signInWithGoogle();
    if (!ok && mounted) {
      setState(() {
        _error = ref.read(authStateProvider).error ??
            'Google sign-in failed. Check network connection.';
      });
    } else if (ok && mounted) {
      context.go('/home');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF050505),
      body: Stack(
        children: [
          Positioned(
            top: -100,
            left: -100,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF3A86FF).withValues(alpha: 0.08),
              ),
            ),
          ),
          Positioned(
            bottom: -150,
            right: -100,
            child: Container(
              width: 400,
              height: 400,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF3A86FF).withValues(alpha: 0.05),
              ),
            ),
          ),
          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    "AXON_ACCESS_PROTOCOL",
                    style: GoogleFonts.robotoMono(
                      color: Colors.white24,
                      fontSize: 10,
                      letterSpacing: 2.0,
                    ),
                  ).animate().fadeIn(duration: 600.ms),
                  const SizedBox(height: 32),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(32),
                    child: BackdropFilter(
                      filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
                      child: Container(
                        padding: const EdgeInsets.all(32),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.03),
                          borderRadius: BorderRadius.circular(32),
                          border: Border.all(
                            color: Colors.white.withValues(alpha: 0.08),
                            width: 0.8,
                          ),
                        ),
                        child: Column(
                          children: [
                            if (_error != null) ...[
                              Container(
                                width: double.infinity,
                                padding: const EdgeInsets.all(12),
                                margin: const EdgeInsets.only(bottom: 20),
                                decoration: BoxDecoration(
                                  color: const Color(0xFFEF4444)
                                      .withValues(alpha: 0.1),
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(
                                    color: const Color(0xFFEF4444)
                                        .withValues(alpha: 0.2),
                                  ),
                                ),
                                child: Text(
                                  _error!,
                                  style: GoogleFonts.robotoMono(
                                    color: const Color(0xFFEF4444),
                                    fontSize: 11,
                                  ),
                                ),
                              ),
                            ],
                            _AuthInputField(
                              controller: _emailCtrl,
                              label: "IDENTIFIER",
                              hint: "email@axon.ai",
                              icon: Icons.alternate_email_rounded,
                              isFocused: _isEmailFocused,
                              onFocusChange: (focused) =>
                                  setState(() => _isEmailFocused = focused),
                            ),
                            const SizedBox(height: 20),
                            _AuthInputField(
                              controller: _passCtrl,
                              label: "SECURITY_KEY",
                              hint: "••••••••",
                              isPassword: true,
                              icon: Icons.lock_outline_rounded,
                              isFocused: _isPasswordFocused,
                              onFocusChange: (focused) =>
                                  setState(() => _isPasswordFocused = focused),
                            ),
                            const SizedBox(height: 32),
                            _PrimaryAuthButton(
                              label: "ESTABLISH_LINK",
                              onTap: _login,
                            ),
                            if (_supportsGoogleSignIn) ...[
                              const SizedBox(height: 16),
                              _SecondaryAuthButton(
                                label: "CONTINUE_WITH_GOOGLE",
                                onTap: _loginWithGoogle,
                                icon: Icons.g_mobiledata_rounded,
                              ),
                            ] else ...[
                              const SizedBox(height: 16),
                              Text(
                                "WINDOWS BUILD: USE EMAIL/PASSWORD AUTH",
                                textAlign: TextAlign.center,
                                style: GoogleFonts.robotoMono(
                                  color: Colors.white24,
                                  fontSize: 10,
                                  letterSpacing: 1.0,
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 32),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        "No credentials? ",
                        style: GoogleFonts.robotoMono(
                          color: Colors.white38,
                          fontSize: 10,
                          letterSpacing: 1.0,
                        ),
                      ),
                      GestureDetector(
                        onTap: () => Navigator.of(context).push(
                          MaterialPageRoute(
                              builder: (_) => const RegisterScreen()),
                        ),
                        child: Text(
                          "INITIALIZE_NEW_USER",
                          style: GoogleFonts.robotoMono(
                            color: const Color(0xFF3A86FF),
                            fontSize: 10,
                            letterSpacing: 1.0,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  GestureDetector(
                    onTap: () => _showPasswordResetDialog(context),
                    child: Text(
                      "RECOVERY_PROTOCOL",
                      style: GoogleFonts.googleSans(
                        color: AxonColors.textTertiary,
                        fontSize: 10,
                        fontWeight: FontWeight.w500,
                        letterSpacing: 1.0,
                        decoration: TextDecoration.underline,
                        decorationColor: AxonColors.divider,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _AuthInputField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final String hint;
  final IconData icon;
  final bool isPassword;
  final bool isFocused;
  final ValueChanged<bool>? onFocusChange;

  const _AuthInputField({
    required this.controller,
    required this.label,
    required this.hint,
    required this.icon,
    this.isPassword = false,
    this.isFocused = false,
    this.onFocusChange,
  });

  @override
  Widget build(BuildContext context) {
    return Focus(
      onFocusChange: onFocusChange,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: GoogleFonts.robotoMono(
                color: Colors.white38, fontSize: 8, letterSpacing: 1.5),
          ),
          const SizedBox(height: 8),
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              boxShadow: isFocused
                  ? [
                      BoxShadow(
                        color: const Color(0xFF3A86FF).withValues(alpha: 0.2),
                        blurRadius: 12,
                        spreadRadius: 0,
                      ),
                    ]
                  : [],
            ),
            child: TextField(
              controller: controller,
              obscureText: isPassword,
              style: GoogleFonts.robotoMono(color: Colors.white, fontSize: 14),
              cursorColor: const Color(0xFF3A86FF),
              decoration: InputDecoration(
                hintText: hint,
                hintStyle:
                    GoogleFonts.robotoMono(color: Colors.white10, fontSize: 14),
                prefixIcon: Icon(icon, color: Colors.white24, size: 18),
                filled: true,
                fillColor: Colors.white.withValues(alpha: 0.02),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide:
                      BorderSide(color: Colors.white.withValues(alpha: 0.05)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(
                      color: const Color(0xFF3A86FF).withValues(alpha: 0.5)),
                ),
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _PrimaryAuthButton extends StatefulWidget {
  final String label;
  final VoidCallback onTap;

  const _PrimaryAuthButton({required this.label, required this.onTap});

  @override
  State<_PrimaryAuthButton> createState() => _PrimaryAuthButtonState();
}

class _PrimaryAuthButtonState extends State<_PrimaryAuthButton> {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) => setState(() => _isPressed = false),
      onTapCancel: () => setState(() => _isPressed = false),
      onTap: () {
        HapticFeedback.mediumImpact();
        widget.onTap();
      },
      child: AnimatedScale(
        scale: _isPressed ? 0.96 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: Container(
          width: double.infinity,
          height: 56,
          decoration: BoxDecoration(
            color: const Color(0xFF3A86FF),
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.3),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          alignment: Alignment.center,
          child: Text(
            widget.label,
            style: GoogleFonts.robotoMono(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 12,
              letterSpacing: 1.2,
            ),
          ),
        ),
      ),
    );
  }
}

class _SecondaryAuthButton extends StatefulWidget {
  final String label;
  final VoidCallback onTap;
  final IconData icon;

  const _SecondaryAuthButton({
    required this.label,
    required this.onTap,
    required this.icon,
  });

  @override
  State<_SecondaryAuthButton> createState() => _SecondaryAuthButtonState();
}

class _SecondaryAuthButtonState extends State<_SecondaryAuthButton> {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) => setState(() => _isPressed = false),
      onTapCancel: () => setState(() => _isPressed = false),
      onTap: () {
        HapticFeedback.lightImpact();
        widget.onTap();
      },
      child: AnimatedScale(
        scale: _isPressed ? 0.96 : 1.0,
        duration: const Duration(milliseconds: 100),
        child: Container(
          width: double.infinity,
          height: 56,
          decoration: BoxDecoration(
            color: Colors.transparent,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: Colors.white.withValues(alpha: 0.1),
            ),
          ),
          alignment: Alignment.center,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(widget.icon, color: Colors.white54, size: 20),
              const SizedBox(width: 8),
              Text(
                widget.label,
                style: GoogleFonts.robotoMono(
                  color: Colors.white54,
                  fontSize: 11,
                  letterSpacing: 1.0,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

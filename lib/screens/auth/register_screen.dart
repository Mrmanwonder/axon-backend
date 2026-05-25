// lib/screens/auth/register_screen.dart
import 'dart:ui';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:go_router/go_router.dart';
import '../../services/app_state.dart';
import '../../services/input_validation_service.dart';

class RegisterScreen extends ConsumerStatefulWidget {
  const RegisterScreen({super.key});

  @override
  ConsumerState<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends ConsumerState<RegisterScreen> {
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  bool _obscure = true;
  String? _error;
  bool _isNameFocused = false;
  bool _isEmailFocused = false;
  bool _isPasswordFocused = false;

  bool get _supportsGoogleSignIn =>
      !kIsWeb && defaultTargetPlatform != TargetPlatform.windows;

  @override
  void initState() {
    super.initState();
    final draft = ref.read(authDraftProvider);
    _nameCtrl.text = draft.name;
    _emailCtrl.text = draft.email;
    _passCtrl.text = draft.password;
    _nameCtrl.addListener(_onNameChanged);
    _emailCtrl.addListener(_onEmailChanged);
    _passCtrl.addListener(_onPasswordChanged);
  }

  @override
  void dispose() {
    _nameCtrl.removeListener(_onNameChanged);
    _emailCtrl.removeListener(_onEmailChanged);
    _passCtrl.removeListener(_onPasswordChanged);
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  void _onNameChanged() {
    ref.read(authDraftProvider.notifier).setName(_nameCtrl.text);
  }

  void _onEmailChanged() {
    ref.read(authDraftProvider.notifier).setEmail(_emailCtrl.text);
  }

  void _onPasswordChanged() {
    ref.read(authDraftProvider.notifier).setPassword(_passCtrl.text);
  }

  Future<void> _register() async {
    final validation = InputValidationService();
    final nameResult = validation.validateDisplayName(_nameCtrl.text);
    if (!nameResult.isValid) {
      setState(() => _error = nameResult.error);
      return;
    }
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
    final ok = await ref.read(authStateProvider.notifier).register(
          _nameCtrl.text.trim(),
          _emailCtrl.text.trim(),
          _passCtrl.text,
        );
    if (!ok && mounted) {
      setState(() {
        _error = ref.read(authStateProvider).error ?? 'Registration failed.';
      });
    } else if (ok && mounted) {
      context.go('/home');
    }
  }

  Future<void> _registerWithGoogle() async {
    setState(() => _error = null);
    final ok = await ref.read(authStateProvider.notifier).signInWithGoogle();
    if (!ok && mounted) {
      setState(() {
        _error = ref.read(authStateProvider).error ?? 'Google sign-up failed.';
      });
    } else if (ok && mounted) {
      context.go('/home');
    }
  }

  @override
  Widget build(BuildContext context) {
    final authState = ref.watch(authStateProvider);
    return Scaffold(
      backgroundColor: const Color(0xFF050505),
      body: Stack(
        children: [
          Positioned(
            top: -150,
            right: -100,
            child: Container(
              width: 350,
              height: 350,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF3A86FF).withValues(alpha: 0.06),
              ),
            ),
          ),
          Positioned(
            bottom: -100,
            left: -100,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFF3A86FF).withValues(alpha: 0.04),
              ),
            ),
          ),
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        IconButton(
                          onPressed: () => context.go('/auth/login'),
                          icon: const Icon(Icons.arrow_back, color: Colors.white24),
                        ),
                      ],
                    ),
                    Text(
                      "IDENTITY_SEQUENCE_START",
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
                                    color: const Color(0xFFEF4444).withValues(alpha: 0.1),
                                    borderRadius: BorderRadius.circular(12),
                                    border: Border.all(
                                      color: const Color(0xFFEF4444).withValues(alpha: 0.2),
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
                                controller: _nameCtrl,
                                label: "USER_DESIGNATION",
                                hint: "Full Name",
                                icon: Icons.person_outline_rounded,
                                isFocused: _isNameFocused,
                                onFocusChange: (bool focused) => setState(() => _isNameFocused = focused),
                              ),
                              const SizedBox(height: 20),
                              _AuthInputField(
                                controller: _emailCtrl,
                                label: "PRIMARY_COMMS",
                                hint: "Email address",
                                icon: Icons.alternate_email_rounded,
                                isFocused: _isEmailFocused,
                                onFocusChange: (bool focused) => setState(() => _isEmailFocused = focused),
                              ),
                              const SizedBox(height: 20),
                              _AuthInputField(
                                controller: _passCtrl,
                                label: "ENCRYPTION_KEY",
                                hint: "Create Password",
                                isPassword: true,
                                isObscured: _obscure,
                                icon: Icons.fingerprint_rounded,
                                isFocused: _isPasswordFocused,
                                onFocusChange: (bool focused) => setState(() => _isPasswordFocused = focused),
                                onToggleVisibility: () => setState(() => _obscure = !_obscure),
                              ),
                              const SizedBox(height: 32),
                              _PrimaryAuthButton(
                                label: "CONFIRM_IDENTITY",
                                isLoading: authState.isLoading,
                                onTap: _register,
                              ),
                              if (_supportsGoogleSignIn) ...[
                                const SizedBox(height: 16),
                                _SecondaryAuthButton(
                                  label: "CONTINUE_WITH_GOOGLE",
                                  onTap: _registerWithGoogle,
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
                    const SizedBox(height: 24),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          "Already registered? ",
                          style: GoogleFonts.robotoMono(
                            color: Colors.white38,
                            fontSize: 10,
                            letterSpacing: 1.0,
                          ),
                        ),
                        GestureDetector(
                          onTap: () => context.go('/auth/login'),
                          child: Text(
                            "RETURN_TO_PROTOCOL",
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
                  ],
                ),
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
  final bool isObscured;
  final bool isFocused;
  final void Function(bool)? onFocusChange;
  final VoidCallback? onToggleVisibility;

  const _AuthInputField({
    required this.controller,
    required this.label,
    required this.hint,
    required this.icon,
    this.isPassword = false,
    this.isObscured = true,
    this.isFocused = false,
    this.onFocusChange,
    this.onToggleVisibility,
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
            style: GoogleFonts.robotoMono(color: Colors.white38, fontSize: 8, letterSpacing: 1.5),
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
              obscureText: isPassword && isObscured,
              style: GoogleFonts.robotoMono(color: Colors.white, fontSize: 14),
              cursorColor: const Color(0xFF3A86FF),
              decoration: InputDecoration(
                hintText: hint,
                hintStyle: GoogleFonts.robotoMono(color: Colors.white10, fontSize: 14),
                prefixIcon: Icon(icon, color: Colors.white24, size: 18),
                suffixIcon: isPassword
                    ? IconButton(
                        icon: Icon(
                          isObscured ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                          color: Colors.white24,
                          size: 18,
                        ),
                        onPressed: onToggleVisibility,
                      )
                    : null,
                filled: true,
                fillColor: Colors.white.withValues(alpha: 0.02),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: const Color(0xFF3A86FF).withValues(alpha: 0.5)),
                ),
                contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
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
  final bool isLoading;

  const _PrimaryAuthButton({
    required this.label,
    required this.onTap,
    this.isLoading = false,
  });

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
      onTap: widget.isLoading ? null : () {
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
            color: widget.isLoading 
                ? const Color(0xFF3A86FF).withValues(alpha: 0.5)
                : const Color(0xFF3A86FF),
            borderRadius: BorderRadius.circular(16),
            boxShadow: widget.isLoading ? [] : [
              BoxShadow(
                color: const Color(0xFF3A86FF).withValues(alpha: 0.3),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          alignment: Alignment.center,
          child: widget.isLoading
              ? const SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    valueColor: AlwaysStoppedAnimation(Colors.white),
                  ),
                )
              : Text(
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

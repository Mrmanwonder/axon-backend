// lib/services/abuse_protection_service.dart
//
// Comprehensive Abuse Protection Service
// Implements rate limiting, bot detection, and abuse prevention
// across all application endpoints

import 'dart:async';
import 'dart:math';
import 'package:shared_preferences/shared_preferences.dart';

enum AbuseType {
  // Authentication abuse
  loginAttempt,
  loginFailed,
  accountCreation,
  passwordReset,
  emailVerification,

  // API abuse
  apiRequest,
  aiGeneration,
  pdfUpload,
  dataScrape,

  // Content abuse
  spam,
  maliciousContent,
}

enum RateLimitScope {
  global,
  perUser,
  perIP,
}

class RateLimitConfig {
  final int maxAttempts;
  final Duration window;
  final Duration lockoutDuration;
  final bool progressivePenalty;

  const RateLimitConfig({
    required this.maxAttempts,
    required this.window,
    required this.lockoutDuration,
    this.progressivePenalty = false,
  });
}

class RateLimitResult {
  final bool allowed;
  final int remainingAttempts;
  final Duration? retryAfter;
  final String? message;

  const RateLimitResult({
    required this.allowed,
    required this.remainingAttempts,
    this.retryAfter,
    this.message,
  });
}

class AbuseProtectionService {
  static final AbuseProtectionService _instance =
      AbuseProtectionService._internal();
  factory AbuseProtectionService() => _instance;
  AbuseProtectionService._internal();

  // Rate limit configurations
  static const Map<AbuseType, RateLimitConfig> _configs = {
    AbuseType.loginAttempt: RateLimitConfig(
      maxAttempts: 5,
      window: Duration(minutes: 1),
      lockoutDuration: Duration(minutes: 15),
      progressivePenalty: true,
    ),
    AbuseType.loginFailed: RateLimitConfig(
      maxAttempts: 3,
      window: Duration(minutes: 5),
      lockoutDuration: Duration(minutes: 30),
      progressivePenalty: true,
    ),
    AbuseType.accountCreation: RateLimitConfig(
      maxAttempts: 3,
      window: Duration(hours: 24),
      lockoutDuration: Duration(hours: 24),
    ),
    AbuseType.passwordReset: RateLimitConfig(
      maxAttempts: 3,
      window: Duration(hours: 1),
      lockoutDuration: Duration(hours: 1),
    ),
    AbuseType.aiGeneration: RateLimitConfig(
      maxAttempts: 20,
      window: Duration(minutes: 10),
      lockoutDuration: Duration(minutes: 30),
    ),
    AbuseType.pdfUpload: RateLimitConfig(
      maxAttempts: 10,
      window: Duration(hours: 1),
      lockoutDuration: Duration(hours: 2),
    ),
    AbuseType.apiRequest: RateLimitConfig(
      maxAttempts: 100,
      window: Duration(minutes: 1),
      lockoutDuration: Duration(minutes: 5),
    ),
  };

  // Keys for SharedPreferences
  String _key(AbuseType type, String identifier) =>
      'abuse_${type.name}_$identifier';

  // Check rate limit
  Future<RateLimitResult> checkLimit(
    AbuseType type,
    String identifier, {
    RateLimitScope scope = RateLimitScope.perUser,
  }) async {
    final config = _configs[type];
    if (config == null) {
      return const RateLimitResult(
        allowed: true,
        remainingAttempts: 999,
      );
    }

    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now().millisecondsSinceEpoch;
    final windowMs = config.window.inMilliseconds;
    final lockoutMs = config.lockoutDuration.inMilliseconds;

    // Get attempt count and first attempt time
    final countKey = _key(type, identifier);
    final timeKey = '${countKey}_time';

    final attemptCount = prefs.getInt(countKey) ?? 0;
    final firstAttemptTime = prefs.getInt(timeKey) ?? now;

    // Check if within lockout period
    if (attemptCount >= config.maxAttempts) {
      final lockoutEnd = firstAttemptTime + windowMs + lockoutMs;
      if (now < lockoutEnd) {
        return RateLimitResult(
          allowed: false,
          remainingAttempts: 0,
          retryAfter: Duration(milliseconds: lockoutEnd - now),
          message: 'Too many attempts. Please try again later.',
        );
      }
      // Lockout expired, reset
      await prefs.setInt(countKey, 0);
      await prefs.setInt(timeKey, now);
    }

    // Check if window has expired (reset count)
    if (now - firstAttemptTime > windowMs) {
      await prefs.setInt(countKey, 1);
      await prefs.setInt(timeKey, now);
      return RateLimitResult(
        allowed: true,
        remainingAttempts: config.maxAttempts - 1,
      );
    }

    // Increment attempt count
    await prefs.setInt(countKey, attemptCount + 1);

    final remaining = config.maxAttempts - attemptCount - 1;
    return RateLimitResult(
      allowed: true,
      remainingAttempts: remaining < 0 ? 0 : remaining,
      message: remaining <= 1 ? 'Warning: One attempt remaining.' : null,
    );
  }

  // Record a failed attempt
  Future<void> recordFailure(AbuseType type, String identifier) async {
    final prefs = await SharedPreferences.getInstance();
    final countKey = _key(type, '${identifier}_failed');
    final timeKey = '${countKey}_time';

    final count = prefs.getInt(countKey) ?? 0;
    final now = DateTime.now().millisecondsSinceEpoch;

    await prefs.setInt(countKey, count + 1);
    await prefs.setInt(timeKey, now);
  }

  // Reset rate limit (after successful action)
  Future<void> resetLimit(AbuseType type, String identifier) async {
    final prefs = await SharedPreferences.getInstance();
    final countKey = _key(type, identifier);
    final failedKey = _key(type, '${identifier}_failed');

    await prefs.remove(countKey);
    await prefs.remove('${countKey}_time');
    await prefs.remove(failedKey);
    await prefs.remove('${failedKey}_time');
  }

  // Get remaining attempts
  Future<int> getRemainingAttempts(AbuseType type, String identifier) async {
    final result = await checkLimit(type, identifier);
    return result.remainingAttempts;
  }

  // Check if locked out
  Future<bool> isLockedOut(AbuseType type, String identifier) async {
    final result = await checkLimit(type, identifier);
    return !result.allowed;
  }

  // Clean up old rate limit data
  Future<void> cleanup() async {
    final prefs = await SharedPreferences.getInstance();
    final now = DateTime.now().millisecondsSinceEpoch;
    final keys = prefs.getKeys().where((k) => k.startsWith('abuse_'));

    for (final key in keys) {
      final timeKey = '${key}_time';
      final time = prefs.getInt(timeKey);
      if (time != null && now - time > 7 * 24 * 60 * 60 * 1000) {
        // Remove entries older than 7 days
        await prefs.remove(key);
        await prefs.remove(timeKey);
      }
    }
  }
}

// Bot Detection Service
class BotDetectionService {
  static final BotDetectionService _instance = BotDetectionService._internal();
  factory BotDetectionService() => _instance;
  BotDetectionService._internal();

  // Suspicious patterns
  static final List<RegExp> _suspiciousPatterns = [
    // Automated user agents
    RegExp(r'(bot|crawler|spider|scraper|automation)', caseSensitive: false),
    // Common scraping tools
    RegExp(r'(curl|wget|python|requests|httpclient)', caseSensitive: false),
    // Headless browsers
    RegExp(r'(headless|phantom|selenium|playwright)', caseSensitive: false),
  ];

  // Suspicious request patterns
  static final List<RegExp> _suspiciousPaths = [
    RegExp(r'/admin|/wp-admin|/phpinfo'),
    RegExp(r'\.env|\.git|\.aws'),
    RegExp(r'select.*from|union.*select', caseSensitive: false),
  ];

  int _suspiciousScore = 0;
  final List<DateTime> _requestTimestamps = [];
  static const int _maxRequestsPerMinute = 60;

  // Check if request is from a bot
  Future<bool> isBot({
    String? userAgent,
    String? ipAddress,
    String? requestPath,
  }) async {
    int score = 0;

    // Check user agent
    if (userAgent != null) {
      for (final pattern in _suspiciousPatterns) {
        if (pattern.hasMatch(userAgent)) {
          score += 50;
        }
      }
    }

    // Check request path for scanning attempts
    if (requestPath != null) {
      for (final pattern in _suspiciousPaths) {
        if (pattern.hasMatch(requestPath)) {
          score += 100;
        }
      }
    }

    // Check request rate
    final now = DateTime.now();
    _requestTimestamps.removeWhere(
      (t) => now.difference(t).inMinutes > 1,
    );
    _requestTimestamps.add(now);

    if (_requestTimestamps.length > _maxRequestsPerMinute) {
      score += 30;
    }

    _suspiciousScore = score;
    return score >= 50;
  }

  // Check for rapid requests (potential automated attack)
  bool isRapidRequest() {
    final now = DateTime.now();
    _requestTimestamps.removeWhere(
      (t) => now.difference(t).inSeconds > 10,
    );

    // More than 10 requests in 10 seconds is suspicious
    return _requestTimestamps.length > 10;
  }

  // Get current suspicion score
  int get suspiciousScore => _suspiciousScore;

  // Reset suspicion score
  void resetScore() {
    _suspiciousScore = 0;
    _requestTimestamps.clear();
  }
}

// Challenge state for math-based CAPTCHA
class ChallengeState {
  final String question;
  final String correctAnswer;
  final DateTime startedAt;

  const ChallengeState({
    required this.question,
    required this.correctAnswer,
    required this.startedAt,
  });
}

// CAPTCHA Service Interface
class CaptchaService {
  static final CaptchaService _instance = CaptchaService._internal();
  factory CaptchaService() => _instance;
  CaptchaService._internal();

  // In production, integrate with reCAPTCHA v3 or hCaptcha
  // For now, implement a simple challenge system

  bool _captchaRequired = false;
  static const int _thresholdForCaptcha = 3;
  ChallengeState? _state;

  bool get isCaptchaRequired => _captchaRequired;

  // Start a math challenge
  void _startChallenge() {
    final a = Random().nextInt(20);
    final b = Random().nextInt(20);
    _state = ChallengeState(question: 'What is $a + $b?', correctAnswer: (a + b).toString(), startedAt: DateTime.now());
  }

  // Determine if CAPTCHA is needed based on behavior
  bool shouldRequireCaptcha({
    required int failedLoginAttempts,
    required int failedAttempts,
    required bool isSuspiciousIP,
  }) {
    _captchaRequired = failedLoginAttempts >= _thresholdForCaptcha ||
        failedAttempts >= _thresholdForCaptcha ||
        isSuspiciousIP;

    return _captchaRequired;
  }

  // Verify CAPTCHA response (simplified)
  Future<bool> verifyCaptcha(String response) async {
    await Future.delayed(const Duration(milliseconds: 500));

    final valid = _state != null && response.trim() == _state!.correctAnswer;

    if (valid) {
      _captchaRequired = false;
      _state = null;
    }

    return valid;
  }

  // Get current challenge question (starts one if needed)
  String getChallengeQuestion() {
    if (_state == null) _startChallenge();
    return _state!.question;
  }

  // Reset CAPTCHA state
  void reset() {
    _captchaRequired = false;
  }
}

// Combined Abuse Protection
class AbuseProtection {
  static final AbuseProtectionService rateLimit = AbuseProtectionService();
  static final BotDetectionService bot = BotDetectionService();
  static final CaptchaService captcha = CaptchaService();

  // Check login attempt
  static Future<RateLimitResult> checkLoginAttempt(String email) async {
    // Check rate limit
    final result = await rateLimit.checkLimit(AbuseType.loginAttempt, email);

    if (!result.allowed) {
      // Log failed attempt
      await rateLimit.recordFailure(AbuseType.loginFailed, email);
    }

    return result;
  }

  // Check account creation
  static Future<RateLimitResult> checkAccountCreation(String email) async {
    return await rateLimit.checkLimit(AbuseType.accountCreation, email);
  }

  // Check AI generation request
  static Future<RateLimitResult> checkAIGeneration(String userId) async {
    return await rateLimit.checkLimit(AbuseType.aiGeneration, userId);
  }

  // Check PDF upload
  static Future<RateLimitResult> checkPdfUpload(String userId) async {
    return await rateLimit.checkLimit(AbuseType.pdfUpload, userId);
  }

  // Check general API request
  static Future<RateLimitResult> checkApiRequest(String userId) async {
    return await rateLimit.checkLimit(AbuseType.apiRequest, userId);
  }

  // Reset after successful action
  static Future<void> resetAfterSuccess(
      AbuseType type, String identifier) async {
    await rateLimit.resetLimit(type, identifier);
  }

  // Detect bots
  static Future<bool> detectBot({
    String? userAgent,
    String? ipAddress,
    String? requestPath,
  }) async {
    return await bot.isBot(
      userAgent: userAgent,
      ipAddress: ipAddress,
      requestPath: requestPath,
    );
  }

  // Check if CAPTCHA required
  static bool shouldRequireCaptcha({
    required int failedAttempts,
    bool isSuspiciousIP = false,
  }) {
    return captcha.shouldRequireCaptcha(
      failedLoginAttempts: failedAttempts,
      failedAttempts: failedAttempts,
      isSuspiciousIP: isSuspiciousIP,
    );
  }
}

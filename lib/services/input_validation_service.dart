// lib/services/input_validation_service.dart
//
// Comprehensive Input Validation Service
// Provides strict validation and sanitization for all user inputs
// Prevents SQL injection, command injection, script injection, and unsafe file uploads

import 'dart:convert';
import 'package:path/path.dart' as path;

class InputValidationResult {
  final bool isValid;
  final String? error;
  final String? sanitized;

  const InputValidationResult({
    required this.isValid,
    this.error,
    this.sanitized,
  });

  factory InputValidationResult.valid([String? sanitized]) =>
      InputValidationResult(isValid: true, sanitized: sanitized);

  factory InputValidationResult.invalid(String error) =>
      InputValidationResult(isValid: false, error: error);
}

class InputValidationService {
  static final InputValidationService _instance =
      InputValidationService._internal();
  factory InputValidationService() => _instance;
  InputValidationService._internal();

  static const int _maxInputLength = 10000;
  static const int _maxFilenameLength = 255;
  static const int _maxEmailLength = 254;
  static const int _maxNameLength = 100;
  static const int _maxSearchQueryLength = 500;

  // Dangerous patterns for injection attacks
  static final RegExp _sqlInjectionPattern = RegExp(
    r'(select|insert|update|delete|drop|union|alter|create|truncate)',
    caseSensitive: false,
  );

  static final RegExp _commandInjectionPattern = RegExp(r'[;&|`$]');

  static final RegExp _scriptInjectionPattern = RegExp(
    r'<script|javascript:|onclick|onerror',
    caseSensitive: false,
    multiLine: true,
  );

  static final RegExp _htmlTagsPattern = RegExp(
    r"<[^>]*>",
    multiLine: true,
  );

  static final RegExp _pathTraversalPattern = RegExp(
    r"(\.\.[\/\\])|(\.\.[\/\\]$)|(^\\[/\\]:)|(^\/etc\/)|(^\/proc\/)",
    caseSensitive: false,
  );

  static final RegExp _nullBytePattern = RegExp(r"\x00");

  // Email validation - strict RFC 5322 simplified
  static final RegExp _emailPattern = RegExp(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$",
  );

  // Allowed file extensions for uploads
  static const Set<String> _allowedExtensions = {
    '.pdf',
  };

  // Allowed MIME types for uploads
  static const Set<String> _allowedMimeTypes = {
    'application/pdf',
    'application/x-pdf',
  };

  // Blocked content patterns in file names - more permissive for valid filenames
  static final RegExp _dangerousFilenamePattern =
      RegExp(r'[<>:"/\\|?*\x00-\x1f]');

  // Validate and sanitize email
  InputValidationResult validateEmail(String email) {
    if (email.isEmpty) {
      return InputValidationResult.invalid('Email is required');
    }

    if (email.length > _maxEmailLength) {
      return InputValidationResult.invalid(
        'Email exceeds maximum length of $_maxEmailLength characters',
      );
    }

    final trimmed = email.trim().toLowerCase();

    // Check for null bytes
    if (_nullBytePattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Email contains invalid characters');
    }

    // Strict email format validation
    if (!_emailPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Invalid email format');
    }

    // Check for injection patterns
    if (_sqlInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Email contains invalid characters');
    }

    if (_scriptInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Email contains invalid characters');
    }

    return InputValidationResult.valid(trimmed);
  }

  // Validate and sanitize display name
  InputValidationResult validateDisplayName(String name) {
    if (name.isEmpty) {
      return InputValidationResult.invalid('Name is required');
    }

    if (name.length > _maxNameLength) {
      return InputValidationResult.invalid(
        'Name exceeds maximum length of $_maxNameLength characters',
      );
    }

    final trimmed = name.trim();

    // Check for null bytes
    if (_nullBytePattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Name contains invalid characters');
    }

    // Check for injection patterns
    if (_sqlInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Name contains invalid characters');
    }

    if (_scriptInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Name contains invalid characters');
    }

    if (_commandInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Name contains invalid characters');
    }

    // Sanitize - remove any HTML tags
    final sanitized = _htmlTagsPattern.allMatches(trimmed).fold<String>(
          trimmed,
          (result, match) => result.replaceRange(match.start, match.end, ''),
        );

    // Validate characters (letters, numbers, spaces, common punctuation)
    if (!RegExp(r"^[\w\s\.\\'-]+$").hasMatch(sanitized)) {
      return InputValidationResult.invalid(
        'Name contains invalid characters. Only letters, numbers, spaces, dots, hyphens, and apostrophes are allowed.',
      );
    }

    return InputValidationResult.valid(sanitized);
  }

  // Validate and sanitize password
  InputValidationResult validatePassword(String password) {
    if (password.isEmpty) {
      return InputValidationResult.invalid('Password is required');
    }

    if (password.length < 8) {
      return InputValidationResult.invalid(
        'Password must be at least 8 characters',
      );
    }

    if (password.length > 128) {
      return InputValidationResult.invalid(
        'Password exceeds maximum length',
      );
    }

    // Check for null bytes
    if (_nullBytePattern.hasMatch(password)) {
      return InputValidationResult.invalid(
          'Password contains invalid characters');
    }

    return InputValidationResult.valid(password);
  }

  // Validate and sanitize search query
  InputValidationResult validateSearchQuery(String query) {
    if (query.isEmpty) {
      return InputValidationResult.invalid('Search query is required');
    }

    if (query.length > _maxSearchQueryLength) {
      return InputValidationResult.invalid(
        'Search query exceeds maximum length of $_maxSearchQueryLength characters',
      );
    }

    final trimmed = query.trim();

    // Check for null bytes
    if (_nullBytePattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid('Query contains invalid characters');
    }

    // Check for SQL injection
    if (_sqlInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid(
        'Query contains potentially dangerous patterns',
      );
    }

    // Check for command injection
    if (_commandInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid(
        'Query contains potentially dangerous characters',
      );
    }

    // Check for script injection
    if (_scriptInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid(
        'Query contains potentially dangerous content',
      );
    }

    // Sanitize - remove HTML tags
    final sanitized = _stripHtmlTags(trimmed);

    // Remove excessive whitespace
    final normalized = sanitized.replaceAll(RegExp(r'\s+'), ' ').trim();

    return InputValidationResult.valid(normalized);
  }

  // Validate file path for security
  InputValidationResult validateFilePath(String filePath) {
    if (filePath.isEmpty) {
      return InputValidationResult.invalid('File path is required');
    }

    // Check for null bytes
    if (_nullBytePattern.hasMatch(filePath)) {
      return InputValidationResult.invalid('Path contains invalid characters');
    }

    // Check for path traversal attempts
    if (_pathTraversalPattern.hasMatch(filePath)) {
      return InputValidationResult.invalid(
        'Path contains invalid traversal patterns',
      );
    }

    return InputValidationResult.valid(filePath);
  }

  // Validate uploaded file
  Future<InputValidationResult> validateUploadedFile({
    required String fileName,
    required int fileSize,
    String? mimeType,
  }) async {
    // Check filename length
    if (fileName.isEmpty) {
      return InputValidationResult.invalid('File name is required');
    }

    if (fileName.length > _maxFilenameLength) {
      return InputValidationResult.invalid(
        'File name exceeds maximum length of $_maxFilenameLength characters',
      );
    }

    // Check for null bytes in filename
    if (_nullBytePattern.hasMatch(fileName)) {
      return InputValidationResult.invalid(
        'File name contains invalid characters',
      );
    }

    // Check for path traversal in filename
    if (_pathTraversalPattern.hasMatch(fileName)) {
      return InputValidationResult.invalid(
        'File name contains invalid path patterns',
      );
    }

    // Check for dangerous characters in filename
    if (_dangerousFilenamePattern.hasMatch(fileName)) {
      return InputValidationResult.invalid(
        'File name contains invalid characters',
      );
    }

    // Validate file extension
    final extension = path.extension(fileName).toLowerCase();
    if (!_allowedExtensions.contains(extension)) {
      return InputValidationResult.invalid(
        'Invalid file type. Only PDF files are allowed.',
      );
    }

    // Validate MIME type if provided
    if (mimeType != null &&
        !_allowedMimeTypes.contains(mimeType.toLowerCase())) {
      return InputValidationResult.invalid(
        'Invalid file type. Only PDF files are allowed.',
      );
    }

    // Check file size (max 500MB for large PDFs like textbooks)
    const maxFileSize = 500 * 1024 * 1024; // 500MB
    if (fileSize > maxFileSize) {
      return InputValidationResult.invalid(
        'File size exceeds maximum allowed size of 500MB',
      );
    }

    // Check for empty file
    if (fileSize == 0) {
      return InputValidationResult.invalid('File is empty');
    }

    // Additional security: Check magic bytes for PDF
    // This would require reading the file, which we do in the actual validation

    return InputValidationResult.valid(fileName);
  }

  // Validate generic text input
  InputValidationResult validateTextInput(
    String input, {
    int? maxLength,
    bool allowHtml = false,
    bool allowSpecialChars = false,
  }) {
    final effectiveMaxLength = maxLength ?? _maxInputLength;

    if (input.isEmpty) {
      return InputValidationResult.invalid('Input is required');
    }

    if (input.length > effectiveMaxLength) {
      return InputValidationResult.invalid(
        'Input exceeds maximum length of $effectiveMaxLength characters',
      );
    }

    // Check for null bytes
    if (_nullBytePattern.hasMatch(input)) {
      return InputValidationResult.invalid('Input contains invalid characters');
    }

    // Check for SQL injection
    if (_sqlInjectionPattern.hasMatch(input)) {
      return InputValidationResult.invalid(
        'Input contains potentially dangerous patterns',
      );
    }

    // Check for command injection
    if (!allowSpecialChars && _commandInjectionPattern.hasMatch(input)) {
      return InputValidationResult.invalid(
        'Input contains potentially dangerous characters',
      );
    }

    // Check for script injection
    if (!allowHtml && _scriptInjectionPattern.hasMatch(input)) {
      return InputValidationResult.invalid(
        'Input contains potentially dangerous content',
      );
    }

    // Sanitize
    String sanitized = input;
    if (!allowHtml) {
      sanitized = _stripHtmlTags(sanitized);
    }

    // Normalize whitespace
    sanitized = sanitized.replaceAll(RegExp(r'\s+'), ' ').trim();

    return InputValidationResult.valid(sanitized);
  }

  // Validate board/exam input
  InputValidationResult validateBoardInput(String board) {
    if (board.isEmpty) {
      return InputValidationResult.invalid('Board is required');
    }

    final normalized = board
        .trim()
        .toLowerCase()
        .replaceAll(' ', '')
        .replaceAll('(', '')
        .replaceAll(')', '');

    // Map variations to standard keys
    final boardMapping = {
      'ib': 'ib',
      'ibdiploma': 'ib',
      'igcse': 'cambridge',
      'igcsecaie': 'cambridge',
      'alevel': 'cambridge',
      'alevelcaie': 'cambridge',
      'alevel_caie': 'cambridge',
      'aslevel': 'cambridge',
      'caie': 'cambridge',
      'cambridge': 'cambridge',
      'cie': 'cambridge',
    };

    final sanitized = boardMapping[normalized];
    if (sanitized == null) {
      return InputValidationResult.invalid(
        'Invalid board. Must be one of: IGCSE, A Level',
      );
    }

    return InputValidationResult.valid(sanitized);
  }

  // Validate subject input
  InputValidationResult validateSubject(String subject) {
    if (subject.isEmpty) {
      return InputValidationResult.invalid('Subject is required');
    }

    if (subject.length > _maxNameLength) {
      return InputValidationResult.invalid(
        'Subject name exceeds maximum length',
      );
    }

    final trimmed = subject.trim();

    // Check for null bytes
    if (_nullBytePattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid(
          'Subject contains invalid characters');
    }

    // Check for injection patterns
    if (_sqlInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid(
          'Subject contains invalid characters');
    }

    if (_scriptInjectionPattern.hasMatch(trimmed)) {
      return InputValidationResult.invalid(
          'Subject contains invalid characters');
    }

    // Allow letters, numbers, spaces, and common punctuation
    if (!RegExp(r'^[\w\s\.\\-]+$').hasMatch(trimmed)) {
      return InputValidationResult.invalid(
        'Subject contains invalid characters',
      );
    }

    return InputValidationResult.valid(trimmed);
  }

  // Helper to strip HTML tags safely
  String _stripHtmlTags(String input) {
    return input.replaceAll(_htmlTagsPattern, '');
  }

  // Sanitize for display (prevent XSS)
  String sanitizeForDisplay(String input) {
    if (input.isEmpty) return '';

    // Remove null bytes
    var sanitized = input.replaceAll(_nullBytePattern, '');

    // Remove HTML tags
    sanitized = sanitized.replaceAll(_htmlTagsPattern, '');

    // Encode HTML entities
    sanitized = const HtmlEscape(HtmlEscapeMode.unknown).convert(sanitized);

    return sanitized;
  }

  // Validate JSON string
  InputValidationResult validateJson(String jsonString) {
    if (jsonString.isEmpty) {
      return InputValidationResult.invalid('JSON is required');
    }

    try {
      json.decode(jsonString);
      return InputValidationResult.valid(jsonString);
    } catch (e) {
      return InputValidationResult.invalid('Invalid JSON format');
    }
  }

  // Validate numeric input
  InputValidationResult validateNumeric(String input) {
    if (input.isEmpty) {
      return InputValidationResult.invalid('Number is required');
    }

    if (double.tryParse(input) == null) {
      return InputValidationResult.invalid('Invalid number format');
    }

    return InputValidationResult.valid(input);
  }

  // Validate integer range
  InputValidationResult validateIntegerRange(
    String input, {
    int? min,
    int? max,
  }) {
    final numericResult = validateNumeric(input);
    if (!numericResult.isValid) {
      return numericResult;
    }

    final value = int.tryParse(input);
    if (value == null) {
      return InputValidationResult.invalid('Please enter a valid number');
    }

    if (min != null && value < min) {
      return InputValidationResult.invalid('Value must be at least $min');
    }

    if (max != null && value > max) {
      return InputValidationResult.invalid('Value must be at most $max');
    }

    return InputValidationResult.valid(input);
  }
}

// HTML escape utility
class HtmlEscape {
  final HtmlEscapeMode mode;
  const HtmlEscape(this.mode);

  String convert(String text) {
    return text
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
  }
}

enum HtmlEscapeMode { unknown, attribute, element }

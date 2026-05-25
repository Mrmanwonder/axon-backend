import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';
import 'pdf_service.dart';

class BookDoc {
  final String title;
  final String subject;
  final String filePath;

  const BookDoc({
    required this.title,
    required this.subject,
    required this.filePath,
  });

  factory BookDoc.fromJson(Map<String, dynamic> json) => BookDoc(
        title: (json['title'] ?? 'Untitled').toString(),
        subject: (json['subject'] ?? 'General').toString(),
        filePath: (json['filePath'] ?? '').toString(),
      );
}

class BookContextBundle {
  final List<BookDoc> books;
  final String context;

  const BookContextBundle({
    required this.books,
    required this.context,
  });
}

class BookDataService {
  BookDataService({PdfService? pdfService})
      : _pdfService = pdfService ?? PdfService();

  static const _docsKey = 'pdf_docs';
  final PdfService _pdfService;

  Future<List<BookDoc>> loadBooks({String? subject}) async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getString(_docsKey);
    if (raw == null || raw.isEmpty) return const [];
    try {
      final decoded = jsonDecode(raw);
      if (decoded is! List) return const [];
      final books = decoded
          .whereType<Map>()
          .map((e) => BookDoc.fromJson(Map<String, dynamic>.from(e)))
          .where((book) => book.filePath.trim().isNotEmpty)
          .toList();
      if (subject == null || subject.trim().isEmpty) {
        return books;
      }
      final normalized = subject.trim().toLowerCase();
      return books
          .where((book) => book.subject.trim().toLowerCase() == normalized)
          .toList();
    } catch (_) {
      return const [];
    }
  }

  Future<String> loadBookText(String filePath) async {
    return _pdfService.loadOrExtractText(filePath);
  }

  Future<List<String>> importBookChapters(
      String subject, String filePath) async {
    final chapters = await _pdfService.extractChapters(filePath);
    if (chapters.isEmpty) return const [];
    return chapters;
  }

  Future<BookContextBundle> buildContext({
    required String subject,
    String? chapter,
    int maxBooks = 3,
    int maxChars = 12000,
  }) async {
    final books = await loadBooks(subject: subject);
    if (books.isEmpty) {
      return const BookContextBundle(books: [], context: '');
    }

    final selectedBooks = books.take(maxBooks).toList();
    final scoredSections = <_ScoredSection>[];
    final queryTerms = _queryTerms(subject: subject, chapter: chapter);

    for (final book in selectedBooks) {
      final text = await loadBookText(book.filePath);
      if (text.trim().isEmpty) continue;
      final sections = _splitSections(text);
      for (final section in sections) {
        final compact = section.replaceAll(RegExp(r'\s+'), ' ').trim();
        if (compact.length < 80) continue;
        final score = _scoreSection(compact, queryTerms);
        if (score <= 0 && chapter != null && chapter.trim().isNotEmpty) {
          continue;
        }
        scoredSections.add(_ScoredSection(
          book: book,
          text: compact,
          score: score,
        ));
      }
    }

    scoredSections.sort((a, b) => b.score.compareTo(a.score));
    final buffer = StringBuffer();
    final usedBooks = <BookDoc>[];
    var currentLength = 0;

    for (final section in scoredSections) {
      final sectionText = 'Book: ${section.book.title}\n${section.text}\n';
      if (currentLength + sectionText.length > maxChars) break;
      if (buffer.isNotEmpty) {
        buffer.writeln();
      }
      buffer.write(sectionText);
      currentLength += sectionText.length;
      if (!usedBooks.any((book) => book.filePath == section.book.filePath)) {
        usedBooks.add(section.book);
      }
      if (usedBooks.length >= maxBooks && currentLength > (maxChars * 0.65)) {
        break;
      }
    }

    if (buffer.isEmpty) {
      for (final book in selectedBooks) {
        final text = await loadBookText(book.filePath);
        if (text.trim().isEmpty) continue;
        final compact = text.replaceAll(RegExp(r'\s+'), ' ').trim();
        if (compact.isEmpty) continue;
        final slice =
            compact.substring(0, compact.length > 3500 ? 3500 : compact.length);
        final sectionText = 'Book: ${book.title}\n$slice\n';
        if (currentLength + sectionText.length > maxChars) break;
        if (buffer.isNotEmpty) {
          buffer.writeln();
        }
        buffer.write(sectionText);
        currentLength += sectionText.length;
        usedBooks.add(book);
      }
    }

    return BookContextBundle(
      books: usedBooks.isEmpty ? selectedBooks : usedBooks,
      context: buffer.toString().trim(),
    );
  }

  List<String> _queryTerms({required String subject, String? chapter}) {
    final source = [subject, chapter ?? ''].join(' ');
    return RegExp(r'[A-Za-z0-9]{3,}')
        .allMatches(source.toLowerCase())
        .map((m) => m.group(0)!)
        .toSet()
        .toList();
  }

  List<String> _splitSections(String text) {
    return text
        .split(RegExp(r'\n{2,}|(?<=[\.\?\!])\s{2,}'))
        .map((part) => part.trim())
        .where((part) => part.isNotEmpty)
        .toList();
  }

  int _scoreSection(String text, List<String> queryTerms) {
    final lower = text.toLowerCase();
    var score = 0;
    for (final term in queryTerms) {
      if (lower.contains(term)) {
        score += 3;
      }
    }
    if (RegExp(r'formula|definition|example|worked example|summary|key points',
            caseSensitive: false)
        .hasMatch(text)) {
      score += 1;
    }
    return score;
  }
}

class _ScoredSection {
  final BookDoc book;
  final String text;
  final int score;

  const _ScoredSection({
    required this.book,
    required this.text,
    required this.score,
  });
}

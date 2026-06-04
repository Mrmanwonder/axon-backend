import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:file_picker/file_picker.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'pdf_viewer_screen.dart';
import 'pdf_results_screen.dart';
import '../../services/app_state.dart';
import '../../services/google_drive_service.dart';
import '../../services/pdf_service_stub.dart';
import '../../services/serper_search_service.dart';
import '../../theme/app_theme.dart';
import '../../utils/layout_utils.dart';
import '../../widgets/common/attachment_source_selector.dart';
import '../../widgets/common/axon_widgets.dart';

class PdfLibraryScreen extends ConsumerStatefulWidget {
  const PdfLibraryScreen({super.key});

  @override
  ConsumerState<PdfLibraryScreen> createState() => _PdfLibraryScreenState();
}

class _PdfLibraryScreenState extends ConsumerState<PdfLibraryScreen> {
  final PdfService _pdfService = PdfService();
  final GoogleDriveService _driveService = GoogleDriveService();

  List<Map<String, dynamic>> _documents = [];
  List<Map<String, dynamic>> _filteredDocuments = [];
  List<Map<String, dynamic>> _selectedDocs = [];
  bool _selectionMode = false;
  bool _loading = true;
  String? _error;
  bool _uploading = false;
  String _searchQuery = '';

  List<Map<String, dynamic>> get _displayDocuments =>
      _searchQuery.isEmpty ? _documents : _filteredDocuments;

  void _onSearchChanged(String query) {
    setState(() {
      _searchQuery = query.trim().toLowerCase();
      if (_searchQuery.isEmpty) {
        _filteredDocuments = [];
      } else {
        _filteredDocuments = _documents.where((doc) {
          final title = (doc['title'] ?? '').toString().toLowerCase();
          final subject = (doc['subject'] ?? '').toString().toLowerCase();
          final tags = (doc['tags'] as List<dynamic>?)?.cast<String>() ?? [];
          return title.contains(_searchQuery) ||
              subject.contains(_searchQuery) ||
              tags.any((t) => t.toLowerCase().contains(_searchQuery));
        }).toList();
      }
    });
  }

  // ==================== Selection Mode ====================

  void _toggleSelectionMode() {
    setState(() {
      _selectionMode = !_selectionMode;
      if (!_selectionMode) {
        _selectedDocs.clear();
      }
    });
  }

  void _toggleDocumentSelection(Map<String, dynamic> doc) {
    setState(() {
      final id = doc['id'];
      final index = _selectedDocs.indexWhere((d) => d['id'] == id);
      if (index >= 0) {
        _selectedDocs.removeAt(index);
      } else {
        _selectedDocs.add(doc);
      }
    });
  }

  bool _isDocumentSelected(Map<String, dynamic> doc) {
    return _selectedDocs.any((d) => d['id'] == doc['id']);
  }

  void _selectAll() {
    setState(() {
      _selectedDocs = List.from(_documents);
    });
  }

  void _deselectAll() {
    setState(() {
      _selectedDocs.clear();
    });
  }

  Future<void> _deleteSelected() async {
    if (_selectedDocs.isEmpty) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AxonColors.surface,
        title: Text('Delete ${_selectedDocs.length} PDFs?',
            style: GoogleFonts.googleSans(color: Colors.white)),
        content: Text('This action cannot be undone.',
            style: GoogleFonts.googleSans(color: Colors.white54)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text('Cancel', style: GoogleFonts.googleSans()),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
            child: Text('Delete', style: GoogleFonts.googleSans()),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;

    final idsToDelete = _selectedDocs.map((d) => d['id']).toSet();
    setState(() {
      _documents.removeWhere((d) => idsToDelete.contains(d['id']));
      _selectedDocs.clear();
      _selectionMode = false;
    });
    await _saveDocs();
  }

  // ==================== Attachment Source Selection Helpers ====================

  Future<String?> _showDriveSearchDialog() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Search Google Drive'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'Enter filename or subject...',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text),
            child: const Text('Search'),
          ),
        ],
      ),
    );
  }

  Future<String?> _showSearchDialog() async {
    final controller = TextEditingController();
    return showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Search Past Papers'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: 'e.g., Biology 9700 June 2023',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text),
            child: const Text('Search'),
          ),
        ],
      ),
    );
  }

  Future<void> _pickLocalFiles() async {
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );
    if (result == null || result.files.isEmpty || !mounted) return;

    for (final file in result.files) {
      final path = file.path;
      if (path == null || path.trim().isEmpty) continue;
      final fileObj = File(path);
      final name = file.name;
      final subject = _detectSubjectFromFilename(name);
      await _addLocalPdf(fileObj, name, subject);
    }
  }

  Future<void> _pickFromGoogleDrive() async {
    final drive = _driveService;
    final hasAccess = await drive.hasAccess();
    if (!hasAccess || !mounted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Failed to connect to Google Drive')),
        );
      }
      return;
    }

    final query = await _showDriveSearchDialog();
    if (query == null || query.isEmpty || !mounted) return;

    final items = await drive.searchFiles(query);
    final pdfItems = items.where((item) => item.isPdf).toList();
    if (!mounted) return;
    if (pdfItems.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('No PDFs found for "$query"')),
      );
      return;
    }

    final selected = await showModalBottomSheet<GoogleDriveItem>(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.6,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Select PDF',
              style: GoogleFonts.googleSans(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: AxonColors.textPrimary,
              ),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.builder(
                itemCount: pdfItems.length,
                itemBuilder: (context, index) {
                  final item = pdfItems[index];
                  return ListTile(
                    leading: Icon(
                      item.isPdf
                          ? Icons.picture_as_pdf_rounded
                          : Icons.insert_drive_file_rounded,
                      color: item.isPdf ? Colors.red : AxonColors.textSecondary,
                    ),
                    title: Text(item.name,
                        maxLines: 2, overflow: TextOverflow.ellipsis),
                    onTap: () => Navigator.pop(context, item),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );

    if (selected == null || !mounted) return;

    setState(() => _uploading = true);
    try {
      final tempPath = await drive.downloadAndSave(selected.id, selected.name);
      if (tempPath != null && mounted) {
        final file = File(tempPath);
        final name = selected.name;
        final subject = _detectSubjectFromFilename(name);
        await _addLocalPdf(file, name, subject);
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Download failed')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  Future<void> _pickFromSerper() async {
    final query = await _showSearchDialog();
    if (query == null || query.isEmpty || !mounted) return;

    final results = await SerperSearchService.instance.searchPastPapers(query);
    if (!mounted) return;
    if (results.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('No results found for "$query"')),
      );
      return;
    }

    final selected = await showModalBottomSheet<SerperSearchResult>(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.6,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AxonColors.surface,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Select Past Paper',
              style: GoogleFonts.googleSans(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: AxonColors.textPrimary,
              ),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.builder(
                itemCount: results.length,
                itemBuilder: (context, index) {
                  final result = results[index];
                  return ListTile(
                    leading: const Icon(Icons.picture_as_pdf_rounded,
                        color: Colors.red),
                    title: Text(result.title,
                        maxLines: 2, overflow: TextOverflow.ellipsis),
                    subtitle:
                        Text(result.date, style: const TextStyle(fontSize: 12)),
                    onTap: () => Navigator.pop(context, result),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );

    if (selected == null || !mounted) return;

    setState(() => _uploading = true);
    try {
      final response = await http.get(Uri.parse(selected.link));
      if (response.statusCode == 200) {
        final appDir = await getApplicationDocumentsDirectory();
        final pdfDir = Directory('${appDir.path}/pdfs');
        if (!await pdfDir.exists()) await pdfDir.create(recursive: true);
        final fileName = selected.title.endsWith('.pdf')
            ? selected.title
            : '${selected.title}.pdf';
        final file = File(
            '${pdfDir.path}/${DateTime.now().millisecondsSinceEpoch}_$fileName');
        await file.writeAsBytes(response.bodyBytes);
        final subject = _detectSubjectFromFilename(fileName);
        await _addLocalPdf(file, fileName, subject);
      } else {
        throw Exception('Failed to download: ${response.statusCode}');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Download failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _uploading = false);
    }
  }

  Future<void> _showAttachmentSourceSelector() async {
    final source = await showAttachmentSourceSelector(context);
    if (source == null || !mounted) return;

    switch (source) {
      case 'local':
        await _pickLocalFiles();
        break;
      case 'drive':
        await _pickFromGoogleDrive();
        break;
      case 'search':
        await _pickFromSerper();
        break;
    }
  }

  // ==================== Lifecycle & Data Loading ====================

  @override
  void initState() {
    super.initState();
    _loadDocuments();
  }

  Future<void> _loadDocuments() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString('pdf_docs');
      if (raw != null && raw.isNotEmpty) {
        final decoded = jsonDecode(raw) as List;
        final docs =
            decoded.map((e) => Map<String, dynamic>.from(e as Map)).toList();
        if (mounted) {
          setState(() {
            _documents = docs;
            _loading = false;
          });
        }
      } else {
        await _fetchFromGoogleDrive();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = 'Failed to load documents';
        });
      }
    }
  }

  Future<void> _fetchFromGoogleDrive() async {
    try {
      final subjects = ref.read(authStateProvider).user?.subjects ?? [];
      if (subjects.isEmpty) {
        if (mounted) {
          setState(() {
            _documents = [];
            _loading = false;
          });
        }
        return;
      }

      debugPrint('Google Drive sync not implemented - using local files');

      final appDir = await getApplicationDocumentsDirectory();
      final pdfDir = Directory('${appDir.path}/past_papers');

      if (await pdfDir.exists()) {
        final files = <Map<String, dynamic>>[];
        await for (final entity in pdfDir.list(recursive: true)) {
          if (entity is File && entity.path.endsWith('.pdf')) {
            final name = entity.path.split('/').last;
            files.add({
              'id': DateTime.now().millisecondsSinceEpoch.toString(),
              'title': name,
              'subject': _detectSubjectFromFilename(name),
              'board': '',
              'paperYear': '',
              'paperType': '',
              'filePath': entity.path,
              'source': 'google_drive',
              'dateAdded': DateTime.now().toIso8601String(),
            });
          }
        }

        if (mounted) {
          setState(() {
            _documents = files;
            _loading = false;
          });
          await _saveDocs();
        }
      } else {
        if (mounted) {
          setState(() {
            _documents = [];
            _loading = false;
          });
        }
      }
    } catch (e) {
      debugPrint('Error fetching from Google Drive: $e');
      if (mounted) {
        setState(() {
          _loading = false;
          _error = 'Failed to fetch from Google Drive';
        });
      }
    }
  }

  // ==================== Import / Download Helpers ====================

  Future<void> _scrapeFromUrl(String url) async {
    setState(() => _uploading = true);
    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final appDir = await getApplicationDocumentsDirectory();
        final pdfDir = Directory('${appDir.path}/pdfs');
        if (!await pdfDir.exists()) await pdfDir.create(recursive: true);

        final fileName = url.split('/').last.contains('.pdf')
            ? url.split('/').last
            : 'scraped_${DateTime.now().millisecondsSinceEpoch}.pdf';

        final file = File('${pdfDir.path}/$fileName');
        await file.writeAsBytes(response.bodyBytes);

        final subject = _detectSubjectFromFilename(fileName);
        await _addLocalPdf(file, fileName, subject);

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Downloaded from URL')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Scraping failed: $e')),
        );
      }
    } finally {
      setState(() => _uploading = false);
    }
  }

  void _showUrlScraper() {
    final controller = TextEditingController();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AxonColors.surface,
        title: Text('Scrape PDF from URL', style: GoogleFonts.googleSans()),
        content: TextField(
          controller: controller,
          decoration: InputDecoration(
            hintText: 'Enter PDF URL',
            hintStyle: GoogleFonts.googleSans(color: AxonColors.textSecondary),
          ),
          style: GoogleFonts.googleSans(color: AxonColors.textPrimary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text('Cancel', style: GoogleFonts.googleSans()),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(ctx);
              if (controller.text.isNotEmpty) {
                _scrapeFromUrl(controller.text);
              }
            },
            child: Text('Download', style: GoogleFonts.googleSans()),
          ),
        ],
      ),
    );
  }

  // ==================== Local Storage ====================

  Future<void> _addLocalPdf(File file, String name, String subject) async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final pdfDir = Directory('${appDir.path}/pdfs');
      if (!await pdfDir.exists()) await pdfDir.create(recursive: true);

      final newPath =
          '${pdfDir.path}/${DateTime.now().millisecondsSinceEpoch}_$name';
      await file.copy(newPath);

      final doc = {
        'id': DateTime.now().millisecondsSinceEpoch.toString(),
        'title': name,
        'subject': subject,
        'board': '',
        'paperYear': '',
        'paperType': '',
        'filePath': newPath,
        'questionCount': '0',
        'difficulty': '',
        'chapter': '',
        'source': 'local',
        'dateAdded': DateTime.now().toIso8601String(),
      };

      setState(() => _documents.insert(0, doc));
      await _saveDocs();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to add PDF: $e')),
        );
      }
    }
  }

  Future<void> _saveDocs() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('pdf_docs', jsonEncode(_documents));
  }

  // ==================== Navigation ====================

  void _openDocument(Map<String, dynamic> doc) async {
    final filePath = doc['filePath'] as String;
    final title = doc['title'] as String;

    final choice = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: const Color(0xFF1A1A1A),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            leading: const Icon(Icons.quiz, color: Color(0xFF3A86FF)),
            title: Text('Practice Questions',
                style: GoogleFonts.googleSans(color: Colors.white)),
            subtitle: Text('Extract and practice questions',
                style: GoogleFonts.googleSans(
                    color: Colors.white54, fontSize: 12)),
            onTap: () => Navigator.pop(ctx, 'practice'),
          ),
          ListTile(
            leading: const Icon(Icons.visibility, color: Color(0xFF10B981)),
            title: Text('View PDF',
                style: GoogleFonts.googleSans(color: Colors.white)),
            subtitle: Text('Open PDF viewer',
                style: GoogleFonts.googleSans(
                    color: Colors.white54, fontSize: 12)),
            onTap: () => Navigator.pop(ctx, 'view'),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );

    if (choice == null || !mounted) return;

    if (choice == 'view') {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => PdfViewerScreen(
            filePath: filePath,
            title: title,
          ),
        ),
      );
      return;
    }

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => const Center(
        child: CircularProgressIndicator(
          color: Color(0xFF3A86FF),
          strokeWidth: 3,
        ),
      ),
    );

    try {
      final questions = await _pdfService.extractQuestions(filePath);
      if (!mounted) return;

      Navigator.pop(context);
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => PdfResultsScreen(
            title: title,
            questions: questions,
            answers: {},
            filePath: filePath,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      Navigator.pop(context);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to extract questions: $e')),
      );
    }
  }

  // ==================== Build ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      extendBody: true,
      body: SafeArea(
        child: Stack(
          children: [
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: _selectionMode
                            ? Text(
                                '${_selectedDocs.length} selected',
                                style: GoogleFonts.googleSans(
                                  color: AxonColors.textPrimary,
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                ),
                              )
                            : Text(
                                'PDF Library',
                                style: GoogleFonts.googleSans(
                                  color: AxonColors.textPrimary,
                                  fontSize: 24,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                      ),
                      if (_selectionMode) ...[
                        IconButton(
                          onPressed: _selectAll,
                          icon: Icon(Icons.select_all_rounded,
                              color: AxonColors.accent),
                          style: IconButton.styleFrom(
                            backgroundColor:
                                AxonColors.accent.withValues(alpha: 0.2),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.all(10),
                          ),
                        ),
                        IconButton(
                          onPressed: _deselectAll,
                          icon: Icon(Icons.deselect_rounded,
                              color: AxonColors.accent),
                          style: IconButton.styleFrom(
                            backgroundColor:
                                AxonColors.accent.withValues(alpha: 0.2),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.all(10),
                          ),
                        ),
                        IconButton(
                          onPressed: _deleteSelected,
                          icon: Icon(Icons.delete_outline_rounded,
                              color: Colors.redAccent),
                          style: IconButton.styleFrom(
                            backgroundColor:
                                Colors.redAccent.withValues(alpha: 0.2),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.all(10),
                          ),
                        ),
                        IconButton(
                          onPressed: _toggleSelectionMode,
                          icon: Icon(Icons.close_rounded,
                              color: AxonColors.accent),
                          style: IconButton.styleFrom(
                            backgroundColor:
                                AxonColors.accent.withValues(alpha: 0.2),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.all(10),
                          ),
                        ),
                      ] else ...[
                        IconButton(
                          onPressed: _showUrlScraper,
                          icon: Icon(Icons.link_rounded,
                              color: AxonColors.accent),
                          style: IconButton.styleFrom(
                            backgroundColor:
                                AxonColors.accent.withValues(alpha: 0.2),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.all(10),
                          ),
                        ),
                        IconButton(
                          onPressed:
                              _uploading ? null : _showAttachmentSourceSelector,
                          icon:
                              Icon(Icons.add_rounded, color: AxonColors.accent),
                          style: IconButton.styleFrom(
                            backgroundColor:
                                AxonColors.accent.withValues(alpha: 0.2),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12)),
                            padding: const EdgeInsets.all(10),
                          ),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 16),
                  AxonInput(
                    onChanged: _onSearchChanged,
                    hintText: 'Search by title or subject',
                  ),
                  const SizedBox(height: 16),
                  if (_loading)
                    const Expanded(
                      child: Center(
                        child: CircularProgressIndicator(
                          color: Color(0xFF3A86FF),
                          strokeWidth: 3,
                        ),
                      ),
                    )
                  else if (_error != null)
                    Expanded(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.error_outline_rounded,
                                color: Colors.white30, size: 48),
                            const SizedBox(height: 12),
                            Text(
                              _error!,
                              style: GoogleFonts.googleSans(
                                  color: Colors.white60, fontSize: 14),
                            ),
                            const SizedBox(height: 12),
                            CyberButton(
                              label: 'Retry',
                              onTap: _loadDocuments,
                            ),
                          ],
                        ),
                      ),
                    )
                  else if (_documents.isEmpty)
                    Expanded(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.description_rounded,
                                color: Colors.white24, size: 64),
                            const SizedBox(height: 16),
                            Text(
                              'No PDFs yet',
                              style: GoogleFonts.googleSans(
                                color: AxonColors.textSecondary,
                                fontSize: 18,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Add your first past paper to get started',
                              style: GoogleFonts.googleSans(
                                color: AxonColors.textTertiary,
                                fontSize: 13,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                  else if (_displayDocuments.isEmpty && _searchQuery.isNotEmpty)
                    Expanded(
                      child: Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.search_off_rounded,
                                color: AxonColors.textTertiary, size: 48),
                            const SizedBox(height: 12),
                            Text(
                              'No results for "$_searchQuery"',
                              style: GoogleFonts.googleSans(
                                color: AxonColors.textSecondary,
                                fontSize: 14,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                  else
                    Expanded(
                      child: GridView.builder(
                        gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          crossAxisSpacing: 12,
                          mainAxisSpacing: 12,
                          childAspectRatio: 0.6,
                        ),
                        padding: EdgeInsets.only(
                            bottom: bottomDockClearance(context)),
                        itemCount: _displayDocuments.length,
                        itemBuilder: (ctx, i) {
                          final doc = _displayDocuments[i];
                          final isSelected = _isDocumentSelected(doc);
                          return _DocumentCard(
                            title: doc['title'] ?? 'Untitled',
                            subject: doc['subject'] ?? 'Unknown',
                            board: doc['board'] ?? '',
                            year: doc['paperYear'] ?? '',
                            filePath: doc['filePath'] ?? '',
                            isSelected: isSelected,
                            selectionMode: _selectionMode,
                            onTap: () {
                              if (_selectionMode) {
                                _toggleDocumentSelection(doc);
                              } else {
                                _openDocument(doc);
                              }
                            },
                            onLongPress: () {
                              if (!_selectionMode) {
                                setState(() {
                                  _selectionMode = true;
                                });
                              }
                              _toggleDocumentSelection(doc);
                            },
                          );
                        },
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ==================== Utilities ====================

  String _detectSubjectFromFilename(String filename) {
    final lower = filename.toLowerCase();
    final subjects = ref.read(authStateProvider).user?.subjects ?? [];

    for (final subject in subjects) {
      if (lower.contains(subject.toLowerCase())) {
        return subject;
      }
    }

    if (lower.contains('math')) return 'Mathematics';
    if (lower.contains('physics')) return 'Physics';
    if (lower.contains('chemistry')) return 'Chemistry';
    if (lower.contains('bio')) return 'Biology';
    if (lower.contains('computer') || lower.contains('cs')) {
      return 'Computer Science';
    }
    if (lower.contains('english')) return 'English';
    if (lower.contains('econ')) return 'Economics';
    if (lower.contains('business')) return 'Business';

    return 'General';
  }
}

class _DocumentCard extends StatelessWidget {
  final String title;
  final String subject;
  final String board;
  final String year;
  final String filePath;
  final bool isSelected;
  final bool selectionMode;
  final VoidCallback onTap;
  final VoidCallback onLongPress;

  const _DocumentCard({
    required this.title,
    required this.subject,
    required this.board,
    required this.year,
    required this.filePath,
    this.isSelected = false,
    this.selectionMode = false,
    required this.onTap,
    required this.onLongPress,
  });

  @override
  Widget build(BuildContext context) {
    String documentCode = board.isNotEmpty ? board : 'PDF';
    if (title.isNotEmpty && title.contains('_')) {
      final parts = title.split('_');
      if (parts.length >= 3) {
        documentCode =
            '${parts[0]}/${parts[1]}/QP/${parts[2].replaceAll('.pdf', '')}';
      }
    }

    return GestureDetector(
      onTap: () {
        HapticFeedback.lightImpact();
        onTap();
      },
      onLongPress: () {
        HapticFeedback.mediumImpact();
        onLongPress();
      },
      child: Container(
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF1E3A5F) : const Color(0xFF1A1A1A),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color:
                isSelected ? const Color(0xFF3A86FF) : const Color(0xFF2A2A2A),
            width: isSelected ? 2 : 1,
          ),
        ),
        child: Stack(
          children: [
            Positioned.fill(
              child: Container(
                margin: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFF121212),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: const Color(0xFF2A2A2A),
                    width: 0.5,
                  ),
                ),
                child: const Center(
                  child: Icon(
                    Icons.insert_drive_file_rounded,
                    color: Color(0xFF4A90D9),
                    size: 48,
                  ),
                ),
              ),
            ),
            Positioned(
              top: 8,
              right: 8,
              child: selectionMode
                  ? Container(
                      width: 24,
                      height: 24,
                      decoration: BoxDecoration(
                        color: isSelected
                            ? const Color(0xFF3A86FF)
                            : Colors.transparent,
                        border: Border.all(
                          color: isSelected
                              ? const Color(0xFF3A86FF)
                              : Colors.white54,
                          width: 2,
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: isSelected
                          ? const Icon(Icons.check,
                              color: Colors.white, size: 16)
                          : null,
                    )
                  : const SizedBox.shrink(),
            ),
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.bottomCenter,
                    end: Alignment.topCenter,
                    colors: [
                      const Color(0xFF1A1A1A),
                      const Color(0xFF1A1A1A).withValues(alpha: 0.8),
                      Colors.transparent,
                    ],
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      documentCode,
                      style: GoogleFonts.googleSans(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subject,
                      style: GoogleFonts.googleSans(
                        color: const Color(0xFF888888),
                        fontSize: 11,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

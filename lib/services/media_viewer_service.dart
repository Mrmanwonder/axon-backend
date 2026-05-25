// lib/services/media_viewer_service.dart
import 'package:url_launcher/url_launcher.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum MediaType {
  pdf,
  youtube,
  vimeo,
  website,
  image,
  unknown,
}

class MediaViewerService {
  static final MediaViewerService _instance = MediaViewerService._internal();
  factory MediaViewerService() => _instance;
  MediaViewerService._internal();

  static MediaType detectType(String url) {
    final lower = url.toLowerCase();

    if (lower.endsWith('.pdf')) {
      return MediaType.pdf;
    }
    if (lower.contains('youtube.com') || lower.contains('youtu.be')) {
      return MediaType.youtube;
    }
    if (lower.contains('vimeo.com')) {
      return MediaType.vimeo;
    }
    if (lower.endsWith('.png') ||
        lower.endsWith('.jpg') ||
        lower.endsWith('.jpeg')) {
      return MediaType.image;
    }
    if (lower.startsWith('http')) {
      return MediaType.website;
    }

    return MediaType.unknown;
  }

  static String? extractYoutubeId(String url) {
    final lower = url.toLowerCase();

    // youtube.com/watch?v=VIDEO_ID
    var match = RegExp(r'youtube\.com/watch\?v=([^&]+)').firstMatch(lower);
    if (match != null) return match.group(1);

    // youtu.be/VIDEO_ID
    match = RegExp(r'youtu\.be/([^?]+)').firstMatch(lower);
    if (match != null) return match.group(1);

    // youtube.com/embed/VIDEO_ID
    match = RegExp(r'youtube\.com/embed/([^?]+)').firstMatch(lower);
    if (match != null) return match.group(1);

    return null;
  }

  static String? extractVimeoId(String url) {
    final match = RegExp(r'vimeo\.com/(\d+)').firstMatch(url);
    return match?.group(1);
  }

  static Future<bool> canLaunch(String url) async {
    final uri = Uri.parse(url);
    return uri.hasScheme && await canLaunchUrl(uri);
  }

  static Future<void> launch(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }
}

class PdfResumeService {
  static const String _lastPositionKey = 'pdf_last_position_';
  static const String _lastPageKey = 'pdf_last_page_';

  static String _key(String path, String keyType) => '$keyType${path.hashCode}';

  static Future<void> savePosition(
      String path, int page, double scrollOffset) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_key(path, _lastPageKey), page);
    await prefs.setDouble(_key(path, _lastPositionKey), scrollOffset);
  }

  static Future<({int page, double offset})?> getPosition(String path) async {
    final prefs = await SharedPreferences.getInstance();
    final page = prefs.getInt(_key(path, _lastPageKey));
    final offset = prefs.getDouble(_key(path, _lastPositionKey));

    if (page != null || offset != null) {
      return (page: page ?? 0, offset: offset ?? 0.0);
    }
    return null;
  }

  static Future<void> clearPosition(String path) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key(path, _lastPageKey));
    await prefs.remove(_key(path, _lastPositionKey));
  }
}

class VideoResumeService {
  static const String _lastPositionKey = 'video_last_position_';

  static String _key(String videoId) => '$_lastPositionKey$videoId';

  static Future<void> savePosition(String videoId, double seconds) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_key(videoId), seconds);
  }

  static Future<double> getPosition(String videoId) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getDouble(_key(videoId)) ?? 0.0;
  }

  static Future<void> clearPosition(String videoId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key(videoId));
  }
}

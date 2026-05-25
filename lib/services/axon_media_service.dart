import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';
// import 'package:youtube_player_flutter/youtube_player_flutter.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../widgets/common/rose_loader.dart';

enum MediaType {
  youtube,
  mp4,
  mov,
  avi,
  mkv,
  webm,
  m3u8,
  audio,
  unknown,
}

class AxonMediaInfo {
  final String url;
  final MediaType type;
  final String? title;
  final String? thumbnail;
  final Duration? duration;

  AxonMediaInfo({
    required this.url,
    required this.type,
    this.title,
    this.thumbnail,
    this.duration,
  });

  static MediaType detectType(String url) {
    final lowerUrl = url.toLowerCase();

    if (lowerUrl.contains('youtube.com') ||
        lowerUrl.contains('youtu.be') ||
        lowerUrl.contains('youtube.googleapis.com')) {
      return MediaType.youtube;
    }
    if (lowerUrl.contains('.mp4')) return MediaType.mp4;
    if (lowerUrl.contains('.mov')) return MediaType.mov;
    if (lowerUrl.contains('.avi')) return MediaType.avi;
    if (lowerUrl.contains('.mkv')) return MediaType.mkv;
    if (lowerUrl.contains('.webm')) return MediaType.webm;
    if (lowerUrl.contains('.m3u8') || lowerUrl.contains('.hls')) {
      return MediaType.m3u8;
    }
    if (lowerUrl.contains('.mp3') ||
        lowerUrl.contains('.wav') ||
        lowerUrl.contains('.m4a') ||
        lowerUrl.contains('.aac')) {
      return MediaType.audio;
    }

    return MediaType.unknown;
  }

  static AxonMediaInfo fromUrl(String url, {String? title}) {
    return AxonMediaInfo(
      url: url,
      type: detectType(url),
      title: title,
    );
  }

  bool get isYouTube => type == MediaType.youtube;
  bool get isStream => type == MediaType.m3u8;
  bool get isAudio => type == MediaType.audio;
  bool get isVideoFile =>
      type == MediaType.mp4 ||
      type == MediaType.mov ||
      type == MediaType.avi ||
      type == MediaType.mkv ||
      type == MediaType.webm;
}

extension MediaTypeExtension on MediaType {
  bool get isYouTube => this == MediaType.youtube;
  bool get isStream => this == MediaType.m3u8;
  bool get isAudio => this == MediaType.audio;
  bool get isVideoFile =>
      this == MediaType.mp4 ||
      this == MediaType.mov ||
      this == MediaType.avi ||
      this == MediaType.mkv ||
      this == MediaType.webm;
}

class AxonMediaPlayer extends StatefulWidget {
  final String url;
  final String? title;
  final bool autoPlay;
  final bool showControls;
  final double? aspectRatio;
  final VoidCallback? onComplete;

  const AxonMediaPlayer({
    super.key,
    required this.url,
    this.title,
    this.autoPlay = false,
    this.showControls = true,
    this.aspectRatio,
    this.onComplete,
  });

  @override
  State<AxonMediaPlayer> createState() => _AxonMediaPlayerState();
}

class _AxonMediaPlayerState extends State<AxonMediaPlayer> {
  late MediaType _mediaType;
  VideoPlayerController? _videoController;
  ChewieController? _chewieController;
  // YoutubePlayerController? _youtubeController;
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }

  Future<void> _initializePlayer() async {
    try {
      final info = AxonMediaInfo.fromUrl(widget.url, title: widget.title);
      _mediaType = info.type;

      if (_mediaType == MediaType.youtube) {
        await _initYouTube();
      } else if (_mediaType == MediaType.audio) {
        await _initAudio();
      } else if (_mediaType.isVideoFile || _mediaType == MediaType.m3u8) {
        await _initVideo();
      } else {
        setState(() {
          _error = 'Unsupported media type';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  Future<void> _initYouTube() async {
    // YouTube player temporarily disabled due to package compatibility issues
    debugPrint('AxonMedia: YouTube playback requires youtube_player_flutter package');
    setState(() {
      _error = 'YouTube playback is temporarily unavailable';
      _isLoading = false;
    });
  }

  // Future<void> _initYouTube() async {
  //   String? videoId;
  //
  //   if (widget.url.contains('youtu.be/')) {
  //     videoId = widget.url.split('youtu.be/').last.split('?').first;
  //   } else if (widget.url.contains('youtube.com/watch')) {
  //     final uri = Uri.parse(widget.url);
  //     videoId = uri.queryParameters['v'];
  //   } else if (widget.url.contains('youtube.googleapis.com')) {
  //     videoId = widget.url.split('/').last.split('?').first;
  //   }
  //
  //   if (videoId == null) {
  //     setState(() {
  //       _error = 'Invalid YouTube URL';
  //       _isLoading = false;
  //     });
  //     return;
  //   }
  //
  //   _youtubeController = YoutubePlayerController(
  //     initialVideoId: videoId,
  //     flags: const YoutubePlayerFlags(
  //       autoPlay: true,
  //       mute: false,
  //       enableCaption: false,
  //       hideControls: false,
  //     ),
  //   );
  //
  //   setState(() => _isLoading = false);
  // }

  Future<void> _initVideo() async {
    _videoController = VideoPlayerController.networkUrl(
      Uri.parse(widget.url),
    );

    await _videoController!.initialize();

    _chewieController = ChewieController(
      videoPlayerController: _videoController!,
      autoPlay: widget.autoPlay,
      showControls: widget.showControls,
      aspectRatio: widget.aspectRatio ?? _videoController!.value.aspectRatio,
      allowFullScreen: true,
      allowMuting: true,
      allowPlaybackSpeedChanging: true,
      playbackSpeeds: const [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
      materialProgressColors: ChewieProgressColors(
        playedColor: Colors.amber,
        handleColor: Colors.amber,
        backgroundColor: Colors.grey.shade800,
        bufferedColor: Colors.grey.shade600,
      ),
    );

    _videoController!.addListener(() {
      if (_videoController!.value.position >=
              _videoController!.value.duration &&
          _videoController!.value.duration.inSeconds > 0) {
        widget.onComplete?.call();
      }
    });

    setState(() => _isLoading = false);
  }

  Future<void> _initAudio() async {
    _videoController = VideoPlayerController.networkUrl(
      Uri.parse(widget.url),
    );

    await _videoController!.initialize();

    _chewieController = ChewieController(
      videoPlayerController: _videoController!,
      autoPlay: widget.autoPlay,
      showControls: widget.showControls,
      aspectRatio: 1.0,
      materialProgressColors: ChewieProgressColors(
        playedColor: Colors.amber,
        handleColor: Colors.amber,
        backgroundColor: Colors.grey.shade800,
        bufferedColor: Colors.grey.shade600,
      ),
      placeholder: Container(
        color: Colors.black,
        child: const Center(
          child: Icon(Icons.audiotrack, color: Colors.amber, size: 64),
        ),
      ),
    );

    setState(() => _isLoading = false);
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _chewieController?.dispose();
    // _youtubeController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: RoseLoader(size: 24, color: Colors.amber),
        ),
      );
    }

    if (_error != null) {
      return Container(
        color: Colors.black,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(
                _error!,
                style: const TextStyle(color: Colors.white),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _isLoading = true;
                    _error = null;
                  });
                  _initializePlayer();
                },
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (_mediaType == MediaType.youtube) {
      return Container(
        color: Colors.black,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.play_circle_outline, color: Colors.amber, size: 64),
              const SizedBox(height: 16),
              const Text(
                'YouTube playback unavailable',
                style: TextStyle(color: Colors.white),
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: () => launchUrl(Uri.parse(widget.url)),
                child: const Text('Open in Browser'),
              ),
            ],
          ),
        ),
      );
    }

    // debugPrint('AxonMedia: YouTube playback requires youtube_player_flutter package');
    // if (_mediaType == MediaType.youtube && _youtubeController != null) {
    //   return YoutubePlayer(
    //     controller: _youtubeController!,
    //     showVideoProgressIndicator: true,
    //     progressIndicatorColor: Colors.amber,
    //     bottomActions: widget.showControls
    //         ? [
    //             CurrentPosition(),
    //             ProgressBar(
    //               isExpanded: true,
    //               colors: ProgressBarColors(
    //                 playedColor: Colors.amber,
    //                 handleColor: Colors.amber,
    //                 backgroundColor: Colors.grey.shade800,
    //                 bufferedColor: Colors.grey.shade600,
    //               ),
    //             ),
    //             RemainingDuration(),
    //             FullScreenButton(),
    //           ]
    //         : null,
    //   );
    // }

    if (_chewieController != null) {
      return Chewie(controller: _chewieController!);
    }

    return const SizedBox();
  }
}

class AxonMediaCard extends StatelessWidget {
  final String url;
  final String? title;
  final String? subtitle;
  final VoidCallback? onTap;
  final bool showPlayButton;

  const AxonMediaCard({
    super.key,
    required this.url,
    this.title,
    this.subtitle,
    this.onTap,
    this.showPlayButton = true,
  });

  @override
  Widget build(BuildContext context) {
    final mediaInfo = AxonMediaInfo.fromUrl(url, title: title);

    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap ?? () => _openMedia(context),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Stack(
              alignment: Alignment.center,
              children: [
                Container(
                  height: 180,
                  width: double.infinity,
                  color: Colors.grey.shade900,
                  child: mediaInfo.isYouTube
                      ? _YouTubeThumbnail(videoId: _extractYouTubeId(url))
                      : const Icon(Icons.video_library,
                          size: 48, color: Colors.grey),
                ),
                if (showPlayButton)
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      mediaInfo.isYouTube
                          ? Icons.play_circle_fill
                          : Icons.play_arrow,
                      color: Colors.amber,
                      size: 48,
                    ),
                  ),
              ],
            ),
            if (title != null || subtitle != null)
              Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title ?? mediaInfo.title ?? 'Media',
                      style: Theme.of(context).textTheme.titleSmall,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (subtitle != null)
                      Text(
                        subtitle!,
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  String? _extractYouTubeId(String url) {
    if (url.contains('youtu.be/')) {
      return url.split('youtu.be/').last.split('?').first;
    }
    if (url.contains('youtube.com/watch')) {
      return Uri.parse(url).queryParameters['v'];
    }
    return null;
  }

  Future<void> _openMedia(BuildContext context) {
    return Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => Scaffold(
          appBar: AppBar(
            title: Text(title ?? 'Media'),
            backgroundColor: Colors.black,
          ),
          backgroundColor: Colors.black,
          body: Center(
            child: AxonMediaPlayer(url: url, title: title),
          ),
        ),
      ),
    );
  }
}

class _YouTubeThumbnail extends StatelessWidget {
  final String? videoId;

  const _YouTubeThumbnail({this.videoId});

  @override
  Widget build(BuildContext context) {
    if (videoId == null) {
      return const Icon(Icons.error, color: Colors.grey);
    }

    return Image.network(
      'https://img.youtube.com/vi/$videoId/hqdefault.jpg',
      fit: BoxFit.cover,
      errorBuilder: (_, __, ___) => const Icon(Icons.error, color: Colors.grey),
    );
  }
}

class AxonMediaGrid extends StatelessWidget {
  final List<AxonMediaInfo> mediaList;
  final Function(AxonMediaInfo)? onTap;
  final int crossAxisCount;

  const AxonMediaGrid({
    super.key,
    required this.mediaList,
    this.onTap,
    this.crossAxisCount = 2,
  });

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: crossAxisCount,
        childAspectRatio: 16 / 10,
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
      ),
      itemCount: mediaList.length,
      itemBuilder: (context, index) {
        final media = mediaList[index];
        return AxonMediaCard(
          url: media.url,
          title: media.title,
          subtitle: media.type.name,
          onTap: onTap != null ? () => onTap!(media) : null,
        );
      },
    );
  }
}

class AxonVideoDownloader {
  static Future<File?> downloadVideo(String url,
      {Function(double)? onProgress}) async {
    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode != 200) return null;

      final dir = await getApplicationDocumentsDirectory();
      final fileName = url.split('/').last;
      final file = File('${dir.path}/videos/$fileName');

      await file.parent.create(recursive: true);
      await file.writeAsBytes(response.bodyBytes);

      return file;
    } catch (e) {
      return null;
    }
  }

  static Future<void> openExternally(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }
}

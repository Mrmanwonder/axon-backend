import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../services/axon_crawler_service.dart';
import 'rose_loader.dart';

final crawlerStateProvider =
    StateNotifierProvider<CrawlerStateNotifier, CrawlerState>(
  (ref) => CrawlerStateNotifier(),
);

enum CrawlerStatus { idle, crawling, completed, error }

class CrawlerState {
  final CrawlerStatus status;
  final double progress;
  final String message;
  final int filesDownloaded;
  final int totalFiles;
  final String? error;

  const CrawlerState({
    this.status = CrawlerStatus.idle,
    this.progress = 0.0,
    this.message = '',
    this.filesDownloaded = 0,
    this.totalFiles = 0,
    this.error,
  });

  CrawlerState copyWith({
    CrawlerStatus? status,
    double? progress,
    String? message,
    int? filesDownloaded,
    int? totalFiles,
    String? error,
  }) {
    return CrawlerState(
      status: status ?? this.status,
      progress: progress ?? this.progress,
      message: message ?? this.message,
      filesDownloaded: filesDownloaded ?? this.filesDownloaded,
      totalFiles: totalFiles ?? this.totalFiles,
      error: error ?? this.error,
    );
  }
}

class CrawlerStateNotifier extends StateNotifier<CrawlerState> {
  final AxonCrawlerService _crawler = AxonCrawlerService();

  CrawlerStateNotifier() : super(const CrawlerState());

  Future<void> startCrawl() async {
    try {
      state = state.copyWith(
        status: CrawlerStatus.crawling,
        message: 'Initializing crawler...',
        progress: 0.0,
      );

      await _crawler.initialize();

      final results = await _crawler.runFullCrawl(
        onProgress: (msg) => state = state.copyWith(message: msg),
        onProgressPercent: (percent) {
          state = state.copyWith(
            progress: percent,
            filesDownloaded: (percent * 100).toInt(),
          );
        },
      );

      await _crawler.getCacheStats();

      state = state.copyWith(
        status: CrawlerStatus.completed,
        progress: 1.0,
        message: 'Crawl complete!',
        filesDownloaded: results.length,
        totalFiles: results.length,
      );
    } catch (e) {
      state = state.copyWith(
        status: CrawlerStatus.error,
        error: e.toString(),
      );
    }
  }

  Future<void> downloadSpecificResource({
    required String url,
    required CrawlerResourceType type,
  }) async {
    try {
      state = state.copyWith(
        status: CrawlerStatus.crawling,
        message: 'Downloading resource...',
      );

      final result = await _crawler.downloadResource(url, type: type);

      if (result != null) {
        state = state.copyWith(
          status: CrawlerStatus.completed,
          message: 'Downloaded: ${result.fileName}',
          filesDownloaded: state.filesDownloaded + 1,
        );
      } else {
        state = state.copyWith(
          status: CrawlerStatus.error,
          error: 'Download failed or file already exists',
        );
      }
    } catch (e) {
      state = state.copyWith(
        status: CrawlerStatus.error,
        error: e.toString(),
      );
    }
  }

  Future<Map<String, dynamic>> getCacheStats() async {
    return await _crawler.getCacheStats();
  }

  Future<List<File>> getCrawledFiles({CrawlerResourceType? type}) async {
    return await _crawler.getCrawledFiles(type: type);
  }

  Future<void> clearCache() async {
    await _crawler.clearCache();
    state = const CrawlerState();
  }
}

class AxonCrawlButton extends ConsumerWidget {
  final String label;
  final IconData icon;
  final VoidCallback? onPressed;
  final Color? backgroundColor;

  const AxonCrawlButton({
    super.key,
    required this.label,
    this.icon = Icons.download,
    this.onPressed,
    this.backgroundColor,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final crawlerState = ref.watch(crawlerStateProvider);
    final isLoading = crawlerState.status == CrawlerStatus.crawling;

    return ElevatedButton.icon(
      style: ElevatedButton.styleFrom(
        backgroundColor:
            backgroundColor ?? Theme.of(context).colorScheme.primary,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      onPressed: isLoading ? null : onPressed,
      icon: isLoading
          ? const RoseLoader(size: 20, color: Colors.white)
          : Icon(icon),
      label: Text(label),
    );
  }
}

class AxonCrawlWidget extends ConsumerWidget {
  final bool showStats;
  final bool showClearButton;

  const AxonCrawlWidget({
    super.key,
    this.showStats = true,
    this.showClearButton = true,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final crawlerState = ref.watch(crawlerStateProvider);
    final notifier = ref.read(crawlerStateProvider.notifier);

    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Icon(
                  Icons.cloud_download,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Resource Crawler',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (crawlerState.status == CrawlerStatus.crawling) ...[
              LinearProgressIndicator(value: crawlerState.progress),
              const SizedBox(height: 8),
              Text(
                crawlerState.message,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            if (crawlerState.status == CrawlerStatus.completed) ...[
              Icon(Icons.check_circle, color: Colors.green[400], size: 32),
              const SizedBox(height: 8),
              Text(
                'Downloaded ${crawlerState.filesDownloaded} files',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
            if (crawlerState.status == CrawlerStatus.error) ...[
              Icon(Icons.error, color: Colors.red[400], size: 32),
              const SizedBox(height: 8),
              Text(
                crawlerState.error ?? 'Unknown error',
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.red,
                    ),
              ),
            ],
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: AxonCrawlButton(
                    label: crawlerState.status == CrawlerStatus.crawling
                        ? 'Crawling...'
                        : 'Start Crawl',
                    icon: Icons.play_arrow,
                    onPressed: () => notifier.startCrawl(),
                  ),
                ),
                if (showClearButton) ...[
                  const SizedBox(width: 8),
                  IconButton(
                    icon: const Icon(Icons.delete_outline),
                    onPressed: () => notifier.clearCache(),
                    tooltip: 'Clear Cache',
                  ),
                ],
              ],
            ),
            if (showStats) ...[
              const Divider(height: 32),
              FutureBuilder<Map<String, dynamic>>(
                future: notifier.getCacheStats(),
                builder: (context, snapshot) {
                  if (!snapshot.hasData) return const SizedBox();
                  final stats = snapshot.data!;
                  return Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _StatItem(
                        label: 'Files',
                        value: '${stats['totalFiles']}',
                      ),
                      _StatItem(
                        label: 'Size',
                        value: '${stats['totalSizeMB']} MB',
                      ),
                      _StatItem(
                        label: 'Hashes',
                        value: '${stats['processedHashes']}',
                      ),
                    ],
                  );
                },
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String label;
  final String value;

  const _StatItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

// lib/services/image_compression_service.dart
//
// Image compression service that adapts to device capabilities
// Uses device tier to determine compression quality and max dimensions

import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'adaptive_performance_service.dart';

class ImageCompressionService {
  static final ImageCompressionService _instance =
      ImageCompressionService._internal();
  factory ImageCompressionService() => _instance;
  ImageCompressionService._internal();

  final AdaptivePerformanceService _adaptive = AdaptivePerformanceService();

  // Compress and resize image before upload
  Future<Uint8List?> compressImage(
    String imagePath, {
    int? maxDimension,
    int? quality,
    bool stripExif = true,
  }) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) return null;

      final bytes = await file.readAsBytes();
      return compressBytes(
        bytes,
        maxDimension: maxDimension,
        quality: quality,
        stripExif: stripExif,
      );
    } catch (e) {
      debugPrint('ImageCompression: Failed to compress $imagePath: $e');
      return null;
    }
  }

  Future<Uint8List?> compressBytes(
    Uint8List bytes, {
    int? maxDimension,
    int? quality,
    bool stripExif = true,
  }) async {
    try {
      final caps = _adaptive.capabilities;
      final maxDim = maxDimension ?? caps.imageMaxDimension;
      final qual = quality ?? (caps.imageQuality * 100).round();

      // Decode image
      var image = img.decodeImage(bytes);
      if (image == null) return null;

      // Strip EXIF data to reduce size
      if (stripExif) {
        // EXIF stripping not available in this version - skip
      }

      // Resize if needed
      if (image.width > maxDim || image.height > maxDim) {
        if (image.width > image.height) {
          image = img.copyResize(image, width: maxDim);
        } else {
          image = img.copyResize(image, height: maxDim);
        }
      }

      // Encode as JPEG with quality (0-100)
      final compressed = img.encodeJpg(image, quality: qual);
      return Uint8List.fromList(compressed);
    } catch (e) {
      debugPrint('ImageCompression: Failed to compress bytes: $e');
      return null;
    }
  }

  // Thumbnail generation for list views
  Future<Uint8List?> generateThumbnail(
    String imagePath, {
    int size = 200,
    int? quality,
  }) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) return null;

      final bytes = await file.readAsBytes();
      var image = img.decodeImage(bytes);
      if (image == null) return null;

      // Create square thumbnail
      final minDim = image.width < image.height ? image.width : image.height;
      final x = (image.width - minDim) ~/ 2;
      final y = (image.height - minDim) ~/ 2;
      image = img.copyCrop(image, x: x, y: y, width: minDim, height: minDim);
      image = img.copyResize(image, width: size, height: size);

      final qual = quality ?? 70;
      return Uint8List.fromList(img.encodeJpg(image, quality: qual));
    } catch (e) {
      debugPrint('ImageCompression: Failed thumbnail $imagePath: $e');
      return null;
    }
  }

  // Save compressed image to temp file
  Future<String?> saveCompressed(
    String imagePath, {
    int? maxDimension,
    int? quality,
  }) async {
    try {
      final compressed = await compressImage(
        imagePath,
        maxDimension: maxDimension,
        quality: quality,
      );
      if (compressed == null) return null;

      final tempDir = await getTemporaryDirectory();
      final fileName =
          'compressed_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final outPath = '${tempDir.path}/$fileName';
      await File(outPath).writeAsBytes(compressed);
      return outPath;
    } catch (e) {
      debugPrint('ImageCompression: Failed save $imagePath: $e');
      return null;
    }
  }

  // Get file size after compression
  Future<int> getCompressedSize(String imagePath) async {
    final compressed = await compressImage(imagePath);
    return compressed?.length ?? 0;
  }

  // Aspect-ratio-aware resize
  Future<Uint8List?> resizeToFit(
    String imagePath, {
    required int maxWidth,
    required int maxHeight,
    int? quality,
  }) async {
    try {
      final file = File(imagePath);
      if (!await file.exists()) return null;

      final bytes = await file.readAsBytes();
      var image = img.decodeImage(bytes);
      if (image == null) return null;

      if (image.width <= maxWidth && image.height <= maxHeight) {
        return bytes; // Already small enough
      }

      final widthRatio = maxWidth / image.width;
      final heightRatio = maxHeight / image.height;
      final ratio = widthRatio < heightRatio ? widthRatio : heightRatio;

      final newWidth = (image.width * ratio).round();
      final newHeight = (image.height * ratio).round();

      image = img.copyResize(image, width: newWidth, height: newHeight);

      final qual =
          quality ?? (_adaptive.capabilities.imageQuality * 100).round();
      return Uint8List.fromList(img.encodeJpg(image, quality: qual));
    } catch (e) {
      debugPrint('ImageCompression: Failed resize $imagePath: $e');
      return null;
    }
  }
}

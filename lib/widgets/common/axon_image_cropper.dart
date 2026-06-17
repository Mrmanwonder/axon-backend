import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../theme/app_theme.dart';

class AxonImageCropper extends StatefulWidget {
  final String imagePath;
  final Function(File)? onCropComplete;
  final VoidCallback? onCancel;

  const AxonImageCropper({
    super.key,
    required this.imagePath,
    this.onCropComplete,
    this.onCancel,
  });

  static Future<String?> crop({
    required String imagePath,
    required BuildContext context,
    VoidCallback? onCancel,
  }) async {
    final result = await Navigator.push<String>(
      context,
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) =>
            AxonImageCropper(
          imagePath: imagePath,
          onCancel: onCancel,
        ),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(opacity: animation, child: child);
        },
      ),
    );
    return result;
  }

  @override
  State<AxonImageCropper> createState() => _AxonImageCropperState();
}

class _AxonImageCropperState extends State<AxonImageCropper> {
  final GlobalKey _cropAreaKey = GlobalKey();
  Rect _cropRect = Rect.zero;
  double _scale = 1.0;
  Offset _offset = Offset.zero;
  double _minScale = 1.0;
  final double _maxScale = 3.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _initCropRect());
  }

  void _initCropRect() {
    final RenderBox? box =
        _cropAreaKey.currentContext?.findRenderObject() as RenderBox?;
    if (box != null) {
      final size = box.size;
      setState(() {
        final cropSize = size.width * 0.8;
        _cropRect = Rect.fromCenter(
          center: Offset(size.width / 2, size.height / 2),
          width: cropSize,
          height: cropSize,
        );
        _offset = Offset(size.width / 2, size.height / 2);
      });
    }
  }

  void _handleScaleStart(ScaleStartDetails details) {
    _minScale = _scale;
  }

  void _handleScaleUpdate(ScaleUpdateDetails details) {
    setState(() {
      if (details.scale != 1.0) {
        final newScale = (_scale * details.scale).clamp(_minScale, _maxScale);
        _scale = newScale;
      } else {
        _offset += details.focalPointDelta;
      }
    });
  }

  void _onDone() {
    HapticFeedback.mediumImpact();

    Navigator.pop(context, widget.imagePath);
    widget.onCropComplete?.call(File(widget.imagePath));
  }

  void _onCancel() {
    HapticFeedback.lightImpact();
    Navigator.pop(context);
    widget.onCancel?.call();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      body: Stack(
        children: [
          // Image with pan/zoom
          GestureDetector(
            onScaleStart: _handleScaleStart,
            onScaleUpdate: _handleScaleUpdate,
            child: Center(
              child: Transform.scale(
                scale: _scale,
                child: Transform.translate(
                  offset: _offset -
                      Offset(MediaQuery.of(context).size.width / 2,
                          MediaQuery.of(context).size.height / 2),
                  child: Image.file(
                    File(widget.imagePath),
                    fit: BoxFit.contain,
                    width: MediaQuery.of(context).size.width,
                    height: MediaQuery.of(context).size.height,
                  ),
                ),
              ),
            ),
          ),

          // Darkened overlay with crop hole
          CustomPaint(
            painter: _CropOverlayPainter(cropRect: _cropRect),
            size: MediaQuery.of(context).size,
          ),

          // Crop area border
          Positioned(
            left: _cropRect.left,
            top: _cropRect.top,
            child: Container(
              width: _cropRect.width,
              height: _cropRect.height,
              decoration: BoxDecoration(
                border: Border.all(color: AxonColors.accent, width: 2),
              ),
            ),
          ),

          // Crop corner handles
          ..._buildCornerHandles(),

          // Bottom buttons
          Positioned(
            left: 0,
            right: 0,
            bottom: MediaQuery.of(context).padding.bottom + 32,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Cancel button
                _CircleButton(
                  icon: Icons.close,
                  onTap: _onCancel,
                  bgColor: Colors.white.withValues(alpha: 0.1),
                  iconColor: Colors.white,
                ),
                // Done button
                _CircleButton(
                  icon: Icons.check,
                  onTap: _onDone,
                  bgColor: AxonColors.accent,
                  iconColor: Colors.white,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  List<Widget> _buildCornerHandles() {
    const handleOffset = -12.0;

    return [
      // Top-left
      Positioned(
        left: _cropRect.left + handleOffset,
        top: _cropRect.top + handleOffset,
        child: _CornerHandle(position: _CornerPosition.topLeft),
      ),
      // Top-right
      Positioned(
        left: _cropRect.right + handleOffset,
        top: _cropRect.top + handleOffset,
        child: _CornerHandle(position: _CornerPosition.topRight),
      ),
      // Bottom-left
      Positioned(
        left: _cropRect.left + handleOffset,
        top: _cropRect.bottom + handleOffset,
        child: _CornerHandle(position: _CornerPosition.bottomLeft),
      ),
      // Bottom-right
      Positioned(
        left: _cropRect.right + handleOffset,
        top: _cropRect.bottom + handleOffset,
        child: _CornerHandle(position: _CornerPosition.bottomRight),
      ),
    ];
  }
}

enum _CornerPosition { topLeft, topRight, bottomLeft, bottomRight }

class _CornerHandle extends StatelessWidget {
  final _CornerPosition position;

  const _CornerHandle({required this.position});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 24,
      height: 24,
      decoration: BoxDecoration(
        color: AxonColors.accent,
        shape: BoxShape.circle,
      ),
      child: Icon(
        _getIcon(),
        color: Colors.white,
        size: 14,
      ),
    );
  }

  IconData _getIcon() {
    switch (position) {
      case _CornerPosition.topLeft:
        return IconsCrop.square;
      case _CornerPosition.topRight:
        return IconsCrop.square;
      case _CornerPosition.bottomLeft:
        return IconsCrop.square;
      case _CornerPosition.bottomRight:
        return IconsCrop.square;
    }
  }
}

class IconsCrop {
  static const square = Icons.crop_free;
}

class _CircleButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  final Color bgColor;
  final Color iconColor;

  const _CircleButton({
    required this.icon,
    required this.onTap,
    required this.bgColor,
    required this.iconColor,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 64,
        height: 64,
        decoration: BoxDecoration(
          color: bgColor,
          shape: BoxShape.circle,
        ),
        child: Icon(
          icon,
          color: iconColor,
          size: 28,
        ),
      ),
    );
  }
}

class _CropOverlayPainter extends CustomPainter {
  final Rect cropRect;

  _CropOverlayPainter({required this.cropRect});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.black.withValues(alpha: 0.6)
      ..style = PaintingStyle.fill;

    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRect(cropRect)
      ..fillType = PathFillType.evenOdd;

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _CropOverlayPainter oldDelegate) {
    return oldDelegate.cropRect != cropRect;
  }
}

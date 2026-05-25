import 'package:flutter/material.dart';

import '../services/public_resource_service.dart';
import '../theme/app_theme.dart';

class ResourceViewerScreen extends StatelessWidget {
  final PublicResourceEntry resource;

  const ResourceViewerScreen({
    super.key,
    required this.resource,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: Text(resource.title),
      ),
      body: const Center(
        child: Text('Resource Viewer'),
      ),
    );
  }
}

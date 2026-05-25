import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../api_client.dart';
import '../ai_proxy_service.dart';

final apiClientProvider = Provider<ApiClient>((ref) {
  return ApiClient(
    baseUrl: const String.fromEnvironment(
      'API_BASE_URL',
      defaultValue: 'https://api.axon.app',
    ),
  );
});

final aiProxyServiceProvider = Provider<AiProxyService>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return AiProxyService(apiClient: apiClient);
});

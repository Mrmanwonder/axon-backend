import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../api_client.dart';
import '../ai_proxy_service.dart';

final sharedPreferencesProvider = Provider<SharedPreferences>((ref) {
  throw UnimplementedError('Must be overridden in main.dart');
});

final apiClientProvider = Provider<ApiClient>((ref) {
  final prefs = ref.watch(sharedPreferencesProvider);
  return ApiClient(
    baseUrl: const String.fromEnvironment(
      'API_BASE_URL',
      defaultValue: 'https://api.axon.app',
    ),
    prefs: prefs,
  );
});

final aiProxyServiceProvider = Provider<AiProxyService>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return AiProxyService(apiClient: apiClient);
});

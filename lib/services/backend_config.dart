class BackendConfig {
  BackendConfig._();

  static const String baseUrl = String.fromEnvironment(
    'AXON_BACKEND_URL',
    defaultValue: 'https://axon-ml.onrender.com',
  );
}

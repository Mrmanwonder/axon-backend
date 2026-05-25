abstract class ConnectivityGateway {
  Future<bool> isOnline();
}

abstract class OnlineAxonModel {
  Future<String> analyze({
    required String task,
    required Map<String, dynamic> payload,
  });
}

abstract class OfflineAxonModel {
  Future<String> analyze({
    required String task,
    required Map<String, dynamic> payload,
  });
}

class HybridAiOrchestrator {
  const HybridAiOrchestrator({
    required ConnectivityGateway connectivityGateway,
    required OnlineAxonModel onlineModel,
    required OfflineAxonModel offlineModel,
  })  : _connectivityGateway = connectivityGateway,
        _onlineModel = onlineModel,
        _offlineModel = offlineModel;

  final ConnectivityGateway _connectivityGateway;
  final OnlineAxonModel _onlineModel;
  final OfflineAxonModel _offlineModel;

  Future<String> run({
    required String task,
    required Map<String, dynamic> payload,
    bool forceOffline = false,
  }) async {
    final online = forceOffline ? false : await _connectivityGateway.isOnline();
    if (online) {
      return _onlineModel.analyze(task: task, payload: payload);
    }
    return _offlineModel.analyze(task: task, payload: payload);
  }
}

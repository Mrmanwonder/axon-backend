import 'dart:async';
import 'dart:convert';
import 'package:dio/dio.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/foundation.dart';

enum ApiErrorType {
  network,
  auth,
  rateLimited,
  serverError,
  timeout,
  unknown,
}

class ApiResult<T> {
  final T? data;
  final ApiErrorType? errorType;
  final String? message;
  final int? statusCode;

  const ApiResult({this.data, this.errorType, this.message, this.statusCode});

  bool get isSuccess => data != null && errorType == null;
  bool get isError => errorType != null;

  static ApiResult<T> success<T>(T data, {int? statusCode}) =>
      ApiResult(data: data, statusCode: statusCode);

  static ApiResult<T> failure<T>(
    ApiErrorType type, {
    String? message,
    int? statusCode,
  }) =>
      ApiResult(errorType: type, message: message, statusCode: statusCode);
}

class ApiClient {
  final Dio _dio;
  final FirebaseAuth? _auth;

  static const Duration _requestTimeout = Duration(seconds: 45);

  /// Shared Dio instance for services that don't need the full ApiClient wrapper.
  static Dio get sharedDio => _sharedDio;
  static final Dio _sharedDio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 15),
    receiveTimeout: _requestTimeout,
    sendTimeout: const Duration(seconds: 15),
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
  ));

  ApiClient({
    required String baseUrl,
    Dio? dio,
    FirebaseAuth? auth,
  })  : _dio = dio ??
            Dio(BaseOptions(
              baseUrl: baseUrl,
              connectTimeout: const Duration(seconds: 15),
              receiveTimeout: _requestTimeout,
              sendTimeout: const Duration(seconds: 15),
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-Client-Version': '1.0.0',
                'X-Client-Platform': 'flutter',
              },
            )),
        _auth = auth ?? FirebaseAuth.instance {
    _dio.interceptors.addAll([
      _AuthInterceptor(_auth, null),
      _RetryInterceptor(),
      _LoggingInterceptor(),
    ]);
  }

  /// Creates an ApiClient without Firebase auth interceptor.
  ApiClient.anonymous({required String baseUrl})
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl,
          connectTimeout: const Duration(seconds: 15),
          receiveTimeout: _requestTimeout,
          sendTimeout: const Duration(seconds: 15),
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
        )),
        _auth = null {
    _dio.interceptors.addAll([_RetryInterceptor(), _LoggingInterceptor()]);
  }

  Future<ApiResult<Map<String, dynamic>>> post(
    String path, {
    Map<String, dynamic>? body,
    Map<String, String>? queryParams,
    Duration? timeout,
  }) async {
    try {
      final response = await _dio.post<Map<String, dynamic>>(
        path,
        data: body,
        queryParameters: queryParams,
        options: Options(receiveTimeout: timeout ?? _requestTimeout),
      );
      return ApiResult.success(
        response.data ?? <String, dynamic>{},
        statusCode: response.statusCode,
      );
    } on DioException catch (e) {
      return _handleDioError<Map<String, dynamic>>(e);
    } catch (e) {
      return ApiResult.failure(
        ApiErrorType.unknown,
        message: 'Unexpected error: ${e.runtimeType}',
      );
    }
  }

  Future<ApiResult<Map<String, dynamic>>> get(
    String path, {
    Map<String, String>? queryParams,
  }) async {
    try {
      final response = await _dio.get<Map<String, dynamic>>(
        path,
        queryParameters: queryParams,
      );
      return ApiResult.success(
        response.data ?? <String, dynamic>{},
        statusCode: response.statusCode,
      );
    } on DioException catch (e) {
      return _handleDioError<Map<String, dynamic>>(e);
    } catch (e) {
      return ApiResult.failure(
        ApiErrorType.unknown,
        message: 'Unexpected error: ${e.runtimeType}',
      );
    }
  }

  Future<ApiResult<List<dynamic>>> getList(
    String path, {
    Map<String, String>? queryParams,
  }) async {
    try {
      final response = await _dio.get<List<dynamic>>(
        path,
        queryParameters: queryParams,
      );
      return ApiResult.success(
        response.data ?? [],
        statusCode: response.statusCode,
      );
    } on DioException catch (e) {
      return _handleDioError<List<dynamic>>(e);
    } catch (e) {
      return ApiResult.failure(
        ApiErrorType.unknown,
        message: 'Unexpected error: ${e.runtimeType}',
      );
    }
  }

  Future<ApiResult<Response<dynamic>>> postRaw(
    String path, {
    Map<String, dynamic>? body,
    Duration? timeout,
  }) async {
    try {
      final response = await _dio.post(
        path,
        data: body,
        options: Options(receiveTimeout: timeout ?? _requestTimeout),
      );
      return ApiResult.success(response, statusCode: response.statusCode);
    } on DioException catch (e) {
      return _handleDioError<Response<dynamic>>(e);
    } catch (e) {
      return ApiResult.failure(
        ApiErrorType.unknown,
        message: 'Unexpected error: ${e.runtimeType}',
      );
    }
  }

  Future<Stream<String>> postStream(
    String path, {
    required Map<String, dynamic> body,
    Duration? timeout,
  }) async {
    try {
      final token = await _getValidToken();
      final response = await _dio.post<ResponseBody>(
        path,
        data: body,
        options: Options(
          responseType: ResponseType.stream,
          receiveTimeout: timeout ?? const Duration(seconds: 60),
          headers: {
            'Authorization': 'Bearer $token',
            'Accept': 'text/event-stream',
          },
        ),
      );

      if (response.data == null) {
        return Stream.error('Empty stream response');
      }

      return response.data!.stream
          .map<List<int>>((chunk) => chunk)
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .where((line) => line.startsWith('data: '))
          .map((line) {
        final data = line.substring(6).trim();
        if (data == '[DONE]') throw _StreamDoneException();
        try {
          final json = jsonDecode(data);
          return json['choices']?[0]?['delta']?['content'] as String? ?? '';
        } catch (_) {
          return '';
        }
      }).handleError((e) {
        if (e is _StreamDoneException) throw e;
      });
    } on DioException catch (e) {
      return Stream.error(_mapDioExceptionToMessage(e));
    } catch (e) {
      return Stream.error('Stream error: $e');
    }
  }

  Future<String> _getValidToken() async {
    final auth = _auth;
    if (auth == null) {
      throw ApiAuthException('Not authenticated');
    }
    final user = auth.currentUser;
    if (user == null) {
      throw ApiAuthException('Not authenticated');
    }
    final token = await user.getIdToken();
    if (token == null || token.isEmpty) {
      throw ApiAuthException('Failed to get auth token');
    }
    return token;
  }

  ApiResult<T> _handleDioError<T>(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        return ApiResult.failure(
          ApiErrorType.timeout,
          message: 'Request timed out',
          statusCode: e.response?.statusCode,
        );
      case DioExceptionType.connectionError:
        return ApiResult.failure(
          ApiErrorType.network,
          message: 'No internet connection',
        );
      case DioExceptionType.badResponse:
        final status = e.response?.statusCode ?? 0;
        if (status == 401 || status == 403) {
          return ApiResult.failure(
            ApiErrorType.auth,
            message: _extractErrorMessage(e.response, 'Authentication failed'),
            statusCode: status,
          );
        }
        if (status == 429) {
          return ApiResult.failure(
            ApiErrorType.rateLimited,
            message: 'Rate limited',
            statusCode: status,
          );
        }
        if (status >= 500) {
          return ApiResult.failure(
            ApiErrorType.serverError,
            message: _extractErrorMessage(e.response, 'Server error'),
            statusCode: status,
          );
        }
        return ApiResult.failure(
          ApiErrorType.unknown,
          message: _extractErrorMessage(e.response, 'Request failed'),
          statusCode: status,
        );
      case DioExceptionType.cancel:
        return ApiResult.failure(ApiErrorType.unknown,
            message: 'Request cancelled');
      default:
        return ApiResult.failure(
          ApiErrorType.network,
          message: e.message ?? 'Network error',
        );
    }
  }

  String _extractErrorMessage(Response? response, String fallback) {
    if (response?.data is Map) {
      return (response!.data as Map)['detail']?.toString() ??
          (response.data as Map)['message']?.toString() ??
          fallback;
    }
    return fallback;
  }

  String _mapDioExceptionToMessage(DioException e) {
    final result = _handleDioError(e);
    return result.message ?? 'Stream failed';
  }
}

class ApiAuthException implements Exception {
  final String message;
  const ApiAuthException(this.message);

  @override
  String toString() => message;
}

class _StreamDoneException implements Exception {}

class _AuthInterceptor extends Interceptor {
  final FirebaseAuth? _auth;

  _AuthInterceptor(this._auth, _);

  @override
  void onRequest(
    RequestOptions options,
    RequestInterceptorHandler handler,
  ) async {
    if (options.path.startsWith('/api/')) {
      try {
        final auth = _auth;
        if (auth == null) { handler.next(options); return; }
        final user = auth.currentUser;
        if (user != null) {
          final token = await user.getIdToken();
          if (token != null && token.isNotEmpty) {
            options.headers['Authorization'] = 'Bearer $token';
          }
        }
      } catch (_) {}
    }
    handler.next(options);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    handler.next(err);
  }
}

class _RetryInterceptor extends Interceptor {
  static const int _maxRetries = 2;

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    if (_shouldRetry(err) &&
        (err.requestOptions.extra['retryCount'] ?? 0) < _maxRetries) {
      final retryCount =
          (err.requestOptions.extra['retryCount'] as int? ?? 0) + 1;
      err.requestOptions.extra['retryCount'] = retryCount;
      await Future.delayed(Duration(milliseconds: 500 * retryCount));
      try {
        final response = await Dio().fetch(err.requestOptions);
        handler.resolve(response);
        return;
      } catch (_) {}
    }
    handler.next(err);
  }

  bool _shouldRetry(DioException err) {
    return err.type == DioExceptionType.connectionTimeout ||
        err.type == DioExceptionType.connectionError ||
        (err.type == DioExceptionType.badResponse &&
            err.response?.statusCode == 429);
  }
}

class _LoggingInterceptor extends Interceptor {
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    debugPrint('[API] ${options.method} ${options.path}');
    handler.next(options);
  }

  @override
  void onResponse(Response response, ResponseInterceptorHandler handler) {
    debugPrint('[API] ${response.statusCode} ${response.requestOptions.path}');
    handler.next(response);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) {
    debugPrint(
        '[API] ERROR ${err.type} ${err.requestOptions.path}: ${err.message}');
    handler.next(err);
  }
}

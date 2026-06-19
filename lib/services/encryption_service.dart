import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:encrypt/encrypt.dart' as enc;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class EncryptionPayload {
  final List<int> keyBytes;
  final String text;

  EncryptionPayload(this.keyBytes, this.text);
}

String _isolateEncrypt(EncryptionPayload payload) {
  final key = enc.Key(Uint8List.fromList(payload.keyBytes));
  final iv = enc.IV.fromSecureRandom(12);
  final encrypter = enc.Encrypter(enc.AES(key, mode: enc.AESMode.gcm, padding: null));
  final encrypted = encrypter.encrypt(payload.text, iv: iv);
  final ivBase64 = base64Url.encode(iv.bytes);
  final cipherTextBase64 = encrypted.base64;
  return '$ivBase64:$cipherTextBase64';
}

String _isolateDecrypt(EncryptionPayload payload) {
  final key = enc.Key(Uint8List.fromList(payload.keyBytes));
  final parts = payload.text.split(':');
  if (parts.length != 2) {
    throw Exception('Invalid encrypted payload structure');
  }
  final ivBytes = base64Url.decode(parts[0]);
  final cipherText = parts[1];
  final iv = enc.IV(ivBytes);
  final encrypter = enc.Encrypter(enc.AES(key, mode: enc.AESMode.gcm, padding: null));
  return encrypter.decrypt64(cipherText, iv: iv);
}

class EncryptionService {
  static const String _keyStorageKey = 'axon_e2ee_master_key';
  static const _secureStorage = FlutterSecureStorage();
  static enc.Key? _masterKey;

  static Future<void> initialize() async {
    if (_masterKey != null) return;
    String? storedKey = await _secureStorage.read(key: _keyStorageKey);
    if (storedKey == null) {
      final keyBytes = enc.Key.fromSecureRandom(32);
      storedKey = base64Url.encode(keyBytes.bytes);
      await _secureStorage.write(key: _keyStorageKey, value: storedKey);
    }
    _masterKey = enc.Key(base64Url.decode(storedKey));
  }

  static Future<enc.Key> _getKey() async {
    if (_masterKey == null) {
      await initialize();
    }
    return _masterKey!;
  }

  static Future<String> encrypt(String plaintext) async {
    final key = await _getKey();
    final payload = EncryptionPayload(key.bytes, plaintext);
    return compute(_isolateEncrypt, payload);
  }

  static Future<String> decrypt(String ciphertext) async {
    final key = await _getKey();
    final payload = EncryptionPayload(key.bytes, ciphertext);
    return compute(_isolateDecrypt, payload);
  }

  static Future<String> encryptMap(Map<String, dynamic> data) async {
    final jsonStr = jsonEncode(data);
    return encrypt(jsonStr);
  }

  static Future<Map<String, dynamic>> decryptMap(String ciphertext) async {
    final jsonStr = await decrypt(ciphertext);
    return jsonDecode(jsonStr) as Map<String, dynamic>;
  }
}

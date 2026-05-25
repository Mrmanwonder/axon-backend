import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }

    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        return windows;
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not configured for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyCY_rG-xvPOXRhzCNUw-JL3iBhn89dekzw',
    appId: '1:650295450237:web:599c5bfb0db3e6da54af14',
    messagingSenderId: '650295450237',
    projectId: 'axon-34b8c',
    authDomain: 'axon-34b8c.firebaseapp.com',
    storageBucket: 'axon-34b8c.firebasestorage.app',
    measurementId: 'G-RY8H3WB2MM',
  );

  static const FirebaseOptions windows = FirebaseOptions(
    apiKey: 'AIzaSyCY_rG-xvPOXRhzCNUw-JL3iBhn89dekzw',
    appId: '1:650295450237:web:599c5bfb0db3e6da54af14',
    messagingSenderId: '650295450237',
    projectId: 'axon-34b8c',
    authDomain: 'axon-34b8c.firebaseapp.com',
    storageBucket: 'axon-34b8c.firebasestorage.app',
    measurementId: 'G-RY8H3WB2MM',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyCMwuUl9pu8UsnO1OBQ6P_ASvmsZyozYk8',
    appId: '1:650295450237:android:4bf1c05ae32ba69854af14',
    messagingSenderId: '650295450237',
    projectId: 'axon-34b8c',
    storageBucket: 'axon-34b8c.firebasestorage.app',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'REPLACE_WITH_IOS_API_KEY',
    appId: 'REPLACE_WITH_IOS_APP_ID',
    messagingSenderId: 'REPLACE_WITH_SENDER_ID',
    projectId: 'REPLACE_WITH_PROJECT_ID',
    storageBucket: 'REPLACE_WITH_STORAGE_BUCKET',
    iosBundleId: 'com.axon.app',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'REPLACE_WITH_MACOS_API_KEY',
    appId: 'REPLACE_WITH_MACOS_APP_ID',
    messagingSenderId: 'REPLACE_WITH_SENDER_ID',
    projectId: 'REPLACE_WITH_PROJECT_ID',
    storageBucket: 'REPLACE_WITH_STORAGE_BUCKET',
    iosBundleId: 'com.axon.app',
  );
}

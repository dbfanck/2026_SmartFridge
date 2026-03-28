import 'package:flutter/material.dart';
import 'screens/main_screen.dart';

void main() {
  runApp(const SmartFridgeApp());
}

class SmartFridgeApp extends StatelessWidget {
  const SmartFridgeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '스마트 냉장고',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2D5BFF)),
        useMaterial3: true,
      ),
      home: const MainScreen(),
    );
  }
}

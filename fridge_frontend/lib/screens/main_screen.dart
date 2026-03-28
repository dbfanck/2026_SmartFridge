import 'package:flutter/material.dart';
import 'home_screen.dart';
import 'inventory_screen.dart';
import 'analysis_screen.dart';
import '../services/api_service.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  @override
  void initState() {
    super.initState();
    ApiService.updateSpoiled();
  }

  final List<Widget> _screens = const [
    HomeScreen(),
    InventoryScreen(),
    AnalysisScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF0F4FF),
      appBar: AppBar(
        backgroundColor: const Color(0xFF2D5BFF),
        foregroundColor: Colors.white,
        title: const Text(
          '스마트 냉장고',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        selectedItemColor: const Color(0xFF2D5BFF),
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined),
            activeIcon: Icon(Icons.home),
            label: '홈',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.kitchen_outlined),
            activeIcon: Icon(Icons.kitchen),
            label: '재고',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart_outlined),
            activeIcon: Icon(Icons.bar_chart),
            label: '분석',
          ),
        ],
      ),
    );
  }
}

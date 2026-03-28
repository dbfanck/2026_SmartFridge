import 'package:flutter/material.dart';
import '../models/fridge_item.dart';
import '../services/api_service.dart';
import '../widgets/fridge_slot.dart';
import 'item_detail_screen.dart';

class InventoryScreen extends StatefulWidget {
  const InventoryScreen({super.key});

  @override
  State<InventoryScreen> createState() => _InventoryScreenState();
}

class _InventoryScreenState extends State<InventoryScreen> {
  late Future<List<FridgeItem>> _itemsFuture;

  @override
  void initState() {
    super.initState();
    _itemsFuture = ApiService.getLayouts();
  }

  Future<void> _refresh() async {
    setState(() {
      _itemsFuture = ApiService.getLayouts();
    });
    await _itemsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<FridgeItem>>(
      future: _itemsFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.error_outline, size: 48, color: Colors.red),
                const SizedBox(height: 8),
                Text('불러오기 실패: ${snapshot.error}'),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: _refresh,
                  child: const Text('다시 시도'),
                ),
              ],
            ),
          );
        }

        final items = snapshot.data ?? [];
        final slot1Items = items.where((e) => e.slotNumber == 1).toList();
        final slot2Items = items.where((e) => e.slotNumber == 2).toList();

        return RefreshIndicator(
          onRefresh: _refresh,
          child: CustomScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            slivers: [
              SliverFillRemaining(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildFridgeBody(context, slot1Items, slot2Items),
                      const SizedBox(height: 12),
                      Text(
                        '총 ${items.length}개 식재료',
                        style: TextStyle(color: Colors.grey[600], fontSize: 13),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildFridgeBody(
    BuildContext context,
    List<FridgeItem> slot1Items,
    List<FridgeItem> slot2Items,
  ) {
    return Expanded(
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFFD0E4FF),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: const Color(0xFF2D5BFF), width: 3),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.1),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          children: [
            _buildFridgeTop(),
            Expanded(
              child: Column(
                children: [
                  Expanded(
                    child: FridgeSlot(
                      slotNumber: 1,
                      label: '위 칸',
                      items: slot1Items,
                      onItemTap: (item) => _openDetail(context, item),
                      isLeft: true,
                    ),
                  ),
                  Container(
                    height: 3,
                    color: const Color(0xFF2D5BFF),
                  ),
                  Expanded(
                    child: FridgeSlot(
                      slotNumber: 2,
                      label: '아래 칸',
                      items: slot2Items,
                      onItemTap: (item) => _openDetail(context, item),
                      isLeft: false,
                    ),
                  ),
                ],
              ),
            ),
            _buildFridgeBottom(),
          ],
        ),
      ),
    );
  }

  Widget _buildFridgeTop() {
    return Container(
      height: 40,
      decoration: const BoxDecoration(
        color: Color(0xFF2D5BFF),
        borderRadius: BorderRadius.only(
          topLeft: Radius.circular(17),
          topRight: Radius.circular(17),
        ),
      ),
      child: const Center(
        child: Icon(Icons.kitchen, color: Colors.white, size: 20),
      ),
    );
  }

  Widget _buildFridgeBottom() {
    return Container(
      height: 20,
      decoration: const BoxDecoration(
        color: Color(0xFF2D5BFF),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(17),
          bottomRight: Radius.circular(17),
        ),
      ),
    );
  }

  void _openDetail(BuildContext context, FridgeItem item) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ItemDetailScreen(item: item),
      ),
    );
  }
}

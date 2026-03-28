import 'package:flutter/material.dart';
import '../models/fridge_item.dart';

class FridgeSlot extends StatelessWidget {
  final int slotNumber;
  final String label;
  final List<FridgeItem> items;
  final void Function(FridgeItem) onItemTap;
  final bool isLeft;

  const FridgeSlot({
    super.key,
    required this.slotNumber,
    required this.label,
    required this.items,
    required this.onItemTap,
    required this.isLeft,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: const Color(0xFF2D5BFF).withValues(alpha: 0.15),
          ),
          child: Center(
            child: Text(
              label,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 13,
                color: Color(0xFF2D5BFF),
              ),
            ),
          ),
        ),
        Expanded(
          child: items.isEmpty
              ? _buildEmptySlot()
              : ListView.builder(
                  padding: const EdgeInsets.all(8),
                  itemCount: items.length,
                  itemBuilder: (context, index) {
                    return _buildItemCard(items[index]);
                  },
                ),
        ),
      ],
    );
  }

  Widget _buildEmptySlot() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.inbox_outlined, size: 36, color: Colors.grey[400]),
          const SizedBox(height: 8),
          Text(
            '비어있음',
            style: TextStyle(color: Colors.grey[400], fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildItemCard(FridgeItem item) {
    final days = item.daysUntilExpiry;
    final ddayText = _ddayText(days);
    final ddayColor = _ddayColor(days);

    return GestureDetector(
      onTap: () => onItemTap(item),
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(10),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.06),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: const Color(0xFF2D5BFF).withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.fastfood,
                size: 18,
                color: Color(0xFF2D5BFF),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                item.productName,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            if (ddayText != null)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 3),
                decoration: BoxDecoration(
                  color: ddayColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  ddayText,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: ddayColor,
                  ),
                ),
              ),
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right, size: 16, color: Colors.grey),
          ],
        ),
      ),
    );
  }

  String? _ddayText(int? days) {
    if (days == null) return null;
    if (days < 0) return '만료';
    if (days == 0) return 'D-Day';
    return 'D-$days';
  }

  Color _ddayColor(int? days) {
    if (days == null) return Colors.grey;
    if (days < 0) return Colors.red;
    if (days <= 3) return Colors.orange;
    return const Color(0xFF2D5BFF);
  }
}

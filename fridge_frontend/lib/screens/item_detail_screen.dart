import 'package:flutter/material.dart';
import '../models/fridge_item.dart';
import '../services/api_service.dart';

class ItemDetailScreen extends StatefulWidget {
  final FridgeItem item;

  const ItemDetailScreen({super.key, required this.item});

  @override
  State<ItemDetailScreen> createState() => _ItemDetailScreenState();
}

class _ItemDetailScreenState extends State<ItemDetailScreen> {
  late Future<FridgeItemDetail> _detailFuture;

  @override
  void initState() {
    super.initState();
    _detailFuture = ApiService.getItemDetail(widget.item.itemId);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF0F4FF),
      appBar: AppBar(
        backgroundColor: const Color(0xFF2D5BFF),
        foregroundColor: Colors.white,
        title: Text(
          widget.item.productName,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ),
      body: FutureBuilder<FridgeItemDetail>(
        future: _detailFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('오류: ${snapshot.error}'));
          }

          final detail = snapshot.data!;
          return Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeroCard(detail),
                const SizedBox(height: 20),
                _buildInfoCard(detail),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildHeroCard(FridgeItemDetail detail) {
    final days = detail.daysUntilExpiry;
    Color expiryColor;
    String expiryLabel;

    if (days == null) {
      expiryColor = Colors.grey;
      expiryLabel = '유통기한 없음';
    } else if (days < 0) {
      expiryColor = Colors.red;
      expiryLabel = '유통기한 만료';
    } else if (days <= 3) {
      expiryColor = Colors.orange;
      expiryLabel = '$days일 남음';
    } else {
      expiryColor = const Color(0xFF2D5BFF);
      expiryLabel = '$days일 남음';
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: expiryColor.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.fastfood, size: 36, color: expiryColor),
          ),
          const SizedBox(height: 16),
          Text(
            detail.productName,
            style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
            decoration: BoxDecoration(
              color: expiryColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              expiryLabel,
              style: TextStyle(
                color: expiryColor,
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard(FridgeItemDetail detail) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '상세 정보',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          _buildInfoRow(
            Icons.inventory_2_outlined,
            '칸 번호',
            '${widget.item.slotNumber}번 칸',
          ),
          if (detail.category != null) ...[
            const Divider(height: 24),
            _buildInfoRow(
              Icons.category_outlined,
              '카테고리',
              detail.category!,
            ),
          ],
          const Divider(height: 24),
          _buildInfoRow(
            Icons.add_circle_outline,
            '등록일',
            _formatDate(detail.createdAt),
          ),
          if (detail.expiresAt != null) ...[
            const Divider(height: 24),
            _buildInfoRow(
              Icons.event_outlined,
              '유통기한',
              _formatDate(detail.expiresAt!),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildInfoRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 20, color: const Color(0xFF2D5BFF)),
        const SizedBox(width: 12),
        Text(
          label,
          style: TextStyle(fontSize: 14, color: Colors.grey[600]),
        ),
        const Spacer(),
        Text(
          value,
          style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
        ),
      ],
    );
  }

  String _formatDate(DateTime date) {
    final local = date.toLocal();
    return '${local.year}.${local.month.toString().padLeft(2, '0')}.${local.day.toString().padLeft(2, '0')}';
  }
}

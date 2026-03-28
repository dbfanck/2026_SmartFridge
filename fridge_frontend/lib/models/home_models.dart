class ExpiringItem {
  final int itemId;
  final String productName;
  final String category;
  final DateTime expiresAt;

  ExpiringItem({
    required this.itemId,
    required this.productName,
    required this.category,
    required this.expiresAt,
  });

  factory ExpiringItem.fromJson(Map<String, dynamic> json) {
    return ExpiringItem(
      itemId: json['item_id'],
      productName: json['product_name'],
      category: json['category'] ?? '-',
      expiresAt: DateTime.parse(json['expires_at']),
    );
  }

  int get daysUntilExpiry {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final expiry = expiresAt.toLocal();
    final expiryDate = DateTime(expiry.year, expiry.month, expiry.day);
    return expiryDate.difference(today).inDays;
  }
}

class RecentItem {
  final String productName;
  final DateTime createdAt;

  RecentItem({required this.productName, required this.createdAt});

  factory RecentItem.fromJson(Map<String, dynamic> json) {
    return RecentItem(
      productName: json['product_name'],
      createdAt: DateTime.parse(json['created_at']),
    );
  }

  String get timeAgo {
    final diff = DateTime.now().difference(createdAt.toLocal());
    if (diff.inDays > 0) return '${diff.inDays}일 전';
    if (diff.inHours > 0) return '${diff.inHours}시간 전';
    if (diff.inMinutes > 0) return '${diff.inMinutes}분 전';
    return '방금 전';
  }
}

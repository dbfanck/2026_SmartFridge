class FridgeItem {
  final int itemId;
  final String productName;
  final int slotNumber;
  final DateTime? expiresAt;

  FridgeItem({
    required this.itemId,
    required this.productName,
    required this.slotNumber,
    this.expiresAt,
  });

  factory FridgeItem.fromJson(Map<String, dynamic> json) {
    return FridgeItem(
      itemId: json['item_id'],
      productName: json['product_name'],
      slotNumber: json['slot_number'],
      expiresAt: json['expires_at'] != null
          ? DateTime.parse(json['expires_at'])
          : null,
    );
  }

  int? get daysUntilExpiry => _calcDays(expiresAt);
}

class FridgeItemDetail {
  final String productName;
  final String? category;
  final DateTime createdAt;
  final DateTime? expiresAt;

  FridgeItemDetail({
    required this.productName,
    this.category,
    required this.createdAt,
    this.expiresAt,
  });

  factory FridgeItemDetail.fromJson(Map<String, dynamic> json) {
    return FridgeItemDetail(
      productName: json['product_name'],
      category: json['category'],
      createdAt: DateTime.parse(json['created_at']),
      expiresAt: json['expires_at'] != null
          ? DateTime.parse(json['expires_at'])
          : null,
    );
  }

  int? get daysUntilExpiry => _calcDays(expiresAt);
}

int? _calcDays(DateTime? expiresAt) {
  if (expiresAt == null) return null;
  final now = DateTime.now();
  final today = DateTime(now.year, now.month, now.day);
  final expiry = expiresAt.toLocal();
  final expiryDate = DateTime(expiry.year, expiry.month, expiry.day);
  return expiryDate.difference(today).inDays;
}

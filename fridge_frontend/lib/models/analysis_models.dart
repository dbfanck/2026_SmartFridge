class RecipeResult {
  final String name;
  final String description;
  final List<String> ingredients;
  final int minutes;

  RecipeResult({
    required this.name,
    required this.description,
    required this.ingredients,
    required this.minutes,
  });

  factory RecipeResult.fromJson(Map<String, dynamic> json) {
    return RecipeResult(
      name: json['name'] ?? '',
      description: json['description'] ?? '',
      ingredients: List<String>.from(json['ingredients'] ?? []),
      minutes: json['minutes'] ?? 0,
    );
  }
}

class CategoryStat {
  final String category;
  final int purchased;
  final int disposed;

  CategoryStat({
    required this.category,
    required this.purchased,
    required this.disposed,
  });

  factory CategoryStat.fromJson(Map<String, dynamic> json) {
    return CategoryStat(
      category: json['category'] ?? '기타',
      purchased: json['purchased'] ?? 0,
      disposed: json['disposed'] ?? 0,
    );
  }
}

class OverallStats {
  final int totalPurchased;
  final int totalDisposed;
  final double disposeRate;

  OverallStats({
    required this.totalPurchased,
    required this.totalDisposed,
    required this.disposeRate,
  });

  factory OverallStats.fromJson(Map<String, dynamic> json) {
    return OverallStats(
      totalPurchased: json['total_purchased'] ?? 0,
      totalDisposed: json['total_disposed'] ?? 0,
      disposeRate: (json['dispose_rate'] ?? 0).toDouble(),
    );
  }
}

class AnalysisItem {
  final int itemId;
  final String productName;
  final String? category;
  final bool isSpoiled;

  AnalysisItem({
    required this.itemId,
    required this.productName,
    this.category,
    required this.isSpoiled,
  });

  factory AnalysisItem.fromJson(Map<String, dynamic> json) {
    return AnalysisItem(
      itemId: json['item_id'],
      productName: json['product_name'] ?? '',
      category: json['category'],
      isSpoiled: json['is_spoiled'] ?? false,
    );
  }
}

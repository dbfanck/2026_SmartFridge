import 'dart:convert';
import 'dart:async';
import 'package:http/http.dart' as http;
import '../models/fridge_item.dart';
import '../models/home_models.dart';
import '../models/analysis_models.dart';

class ApiService {
  static const String baseUrl = 'https://two026-smart-fridge.onrender.com';

  static Future<List<FridgeItem>> getLayouts() async {
    final response = await http.get(Uri.parse('$baseUrl/layouts'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((e) => FridgeItem.fromJson(e)).toList();
    }
    throw Exception('레이아웃 불러오기 실패');
  }

  static Future<FridgeItemDetail> getItemDetail(int itemId) async {
    final response = await http.get(Uri.parse('$baseUrl/layouts/$itemId'));
    if (response.statusCode == 200) {
      return FridgeItemDetail.fromJson(jsonDecode(response.body));
    }
    throw Exception('상세 정보 불러오기 실패');
  }

  static Future<List<ExpiringItem>> getExpiringItems() async {
    final response = await http.get(Uri.parse('$baseUrl/items/expiring'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((e) => ExpiringItem.fromJson(e)).toList();
    }
    throw Exception('유통기한 임박 항목 불러오기 실패');
  }

  static Future<List<RecentItem>> getRecentItems() async {
    final response = await http.get(Uri.parse('$baseUrl/items/recent'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((e) => RecentItem.fromJson(e)).toList();
    }
    throw Exception('최근 활동 불러오기 실패');
  }

  static Future<List<AnalysisItem>> getAnalysisItems() async {
    final response = await http.get(Uri.parse('$baseUrl/analysis/items'));
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((e) => AnalysisItem.fromJson(e)).toList();
    }
    throw Exception('식재료 목록 불러오기 실패');
  }

  static Future<List<CategoryStat>> getCategoryStats() async {
    final response = await http.get(
      Uri.parse('$baseUrl/analysis/stats/category'),
    );
    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.map((e) => CategoryStat.fromJson(e)).toList();
    }
    throw Exception('카테고리 통계 불러오기 실패');
  }

  static Future<OverallStats> getOverallStats() async {
    final response = await http.get(
      Uri.parse('$baseUrl/analysis/stats/overall'),
    );
    if (response.statusCode == 200) {
      return OverallStats.fromJson(jsonDecode(response.body));
    }
    throw Exception('전체 통계 불러오기 실패');
  }

  static Future<void> updateSpoiled() async {
    await http.patch(Uri.parse('$baseUrl/items/update-spoiled'));
  }

  static Future<RecipeResult> getRecipe() async {
    final response = await http
        .get(
          Uri.parse('$baseUrl/analysis/recipe'),
          headers: {'Content-Type': 'application/json'},
        )
        .timeout(const Duration(seconds: 30));
    if (response.statusCode == 200) {
      return RecipeResult.fromJson(jsonDecode(utf8.decode(response.bodyBytes)));
    }
    final detail =
        jsonDecode(utf8.decode(response.bodyBytes))['detail'] ?? '알 수 없는 오류';
    throw Exception(detail);
  }
}

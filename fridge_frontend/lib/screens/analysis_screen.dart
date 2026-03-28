import 'package:flutter/material.dart';
import '../models/analysis_models.dart';
import '../services/api_service.dart';

// ────────────────────────────── 메인 화면 ────────────────────────────────────

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  late Future<List<CategoryStat>> _categoryFuture;
  late Future<OverallStats> _overallFuture;
  bool _isLoadingRecipe = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  void _load() {
    _categoryFuture = ApiService.getCategoryStats();
    _overallFuture = ApiService.getOverallStats();
  }

  void _refresh() => setState(() => _load());

  void _onRecommendTapped() async {
    setState(() => _isLoadingRecipe = true);
    try {
      final recipe = await ApiService.getRecipe();
      if (!mounted) return;
      _showRecipeBottomSheet(recipe);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('레시피 추천 실패: $e')),
      );
    } finally {
      if (mounted) setState(() => _isLoadingRecipe = false);
    }
  }

  void _showRecipeBottomSheet(RecipeResult recipe) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _RecipeSheet(recipe: recipe),
    );
  }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: () async => _refresh(),
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildRecipeSection(),
          const SizedBox(height: 16),
          _buildChartSection(),
          const SizedBox(height: 16),
          _buildOverallStatsSection(),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  // ── 섹션 공통 카드 래퍼 ───────────────────────────────────────────────────

  Widget _sectionCard({
    required IconData icon,
    required String title,
    required Widget child,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.07),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 18, color: const Color(0xFF2D5BFF)),
              const SizedBox(width: 6),
              Text(
                title,
                style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 14),
          child,
        ],
      ),
    );
  }

  // ── AI 레시피 추천 섹션 ───────────────────────────────────────────────────

  Widget _buildRecipeSection() {
    return _sectionCard(
      icon: Icons.auto_awesome,
      title: 'AI 레시피 추천',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '냉장고 속 식재료를 기반으로 오늘의 메뉴를 추천해드립니다.',
            style: TextStyle(fontSize: 13, color: Colors.grey[600]),
          ),
          const SizedBox(height: 14),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _isLoadingRecipe ? null : _onRecommendTapped,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2D5BFF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 0,
              ),
              icon: _isLoadingRecipe
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(
                          color: Colors.white, strokeWidth: 2),
                    )
                  : const Icon(Icons.restaurant_menu, size: 18),
              label: Text(
                _isLoadingRecipe ? '추천 중...' : '레시피 추천 받기',
                style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── 카테고리별 통계 (세로 바 차트) ───────────────────────────────────────

  Widget _buildChartSection() {
    return _sectionCard(
      icon: Icons.bar_chart,
      title: '카테고리별 통계',
      child: FutureBuilder<List<CategoryStat>>(
        future: _categoryFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError || snapshot.data == null || snapshot.data!.isEmpty) {
            return _emptyMessage('데이터가 없습니다');
          }
          final stats = snapshot.data!;
          return _VerticalBarChart(stats: stats);
        },
      ),
    );
  }

  // ── 전체 통계 섹션 ────────────────────────────────────────────────────────

  Widget _buildOverallStatsSection() {
    return _sectionCard(
      icon: Icons.analytics_outlined,
      title: '전체 통계',
      child: FutureBuilder<OverallStats>(
        future: _overallFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          final stats = snapshot.data;
          final purchased = stats?.totalPurchased ?? 0;
          final disposed = stats?.totalDisposed ?? 0;
          final rate = stats?.disposeRate ?? 0.0;

          return Row(
            children: [
              Expanded(
                child: _statTile(
                  label: '총 구매',
                  value: '$purchased개',
                  icon: Icons.shopping_cart_outlined,
                  color: const Color(0xFF2D5BFF),
                ),
              ),
              Container(width: 1, height: 60, color: Colors.grey[200]),
              Expanded(
                child: _statTile(
                  label: '총 폐기',
                  value: '$disposed개',
                  icon: Icons.delete_outline,
                  color: const Color(0xFFFF6B6B),
                ),
              ),
              Container(width: 1, height: 60, color: Colors.grey[200]),
              Expanded(
                child: _statTile(
                  label: '폐기율',
                  value: '${rate.toStringAsFixed(1)}%',
                  icon: Icons.pie_chart_outline,
                  color: const Color(0xFFFFB703),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _statTile({
    required String label,
    required String value,
    required IconData icon,
    required Color color,
  }) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.1),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: color, size: 22),
        ),
        const SizedBox(height: 8),
        Text(
          value,
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color),
        ),
        const SizedBox(height: 2),
        Text(label, style: TextStyle(fontSize: 12, color: Colors.grey[500])),
      ],
    );
  }

  Widget _emptyMessage(String message) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 20),
      child: Center(
        child: Text(message, style: TextStyle(color: Colors.grey[400], fontSize: 13)),
      ),
    );
  }
}

// ────────────────────────────── 세로 바 차트 ─────────────────────────────────

class _VerticalBarChart extends StatelessWidget {
  final List<CategoryStat> stats;
  const _VerticalBarChart({required this.stats});

  static const double _chartHeight = 160;
  static const Color _purchaseColor = Color(0xFF2D5BFF);
  static const Color _disposeColor = Color(0xFFFF6B6B);

  @override
  Widget build(BuildContext context) {
    final maxVal = stats
        .map((s) => s.purchased)
        .fold(0, (a, b) => a > b ? a : b)
        .toDouble();

    return Column(
      children: [
        // 범례
        Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            _legend(color: _purchaseColor, label: '구매'),
            const SizedBox(width: 12),
            _legend(color: _disposeColor, label: '폐기'),
          ],
        ),
        const SizedBox(height: 12),
        // 차트
        SizedBox(
          height: _chartHeight + 48,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: stats.map((stat) => _barGroup(stat, maxVal)).toList(),
          ),
        ),
      ],
    );
  }

  Widget _legend({required Color color, required String label}) {
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
      ],
    );
  }

  Widget _barGroup(CategoryStat stat, double maxVal) {
    return Expanded(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          // 숫자 레이블 (구매)
          Text(
            '${stat.purchased}',
            style: const TextStyle(
                fontSize: 10, color: _purchaseColor, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 2),
          // 바 묶음
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              _singleBar(
                value: stat.purchased,
                max: maxVal,
                color: _purchaseColor,
              ),
              const SizedBox(width: 3),
              _singleBar(
                value: stat.disposed,
                max: maxVal,
                color: _disposeColor,
              ),
            ],
          ),
          const SizedBox(height: 6),
          // 카테고리 레이블
          Text(
            stat.category,
            style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w500),
            textAlign: TextAlign.center,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  Widget _singleBar({
    required int value,
    required double max,
    required Color color,
  }) {
    final ratio = max == 0 ? 0.0 : value / max;
    final barH = (_chartHeight * ratio).clamp(4.0, _chartHeight);

    return Container(
      width: 14,
      height: barH,
      decoration: BoxDecoration(
        color: color,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
      ),
    );
  }
}

// ────────────────────────────── 레시피 바텀시트 ──────────────────────────────

class _RecipeSheet extends StatelessWidget {
  final RecipeResult recipe;
  const _RecipeSheet({required this.recipe});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: const EdgeInsets.fromLTRB(20, 12, 20, 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xFF2D5BFF).withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.auto_awesome,
                    color: Color(0xFF2D5BFF), size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('AI 추천 레시피',
                        style:
                            TextStyle(fontSize: 12, color: Color(0xFF2D5BFF))),
                    Text(recipe.name,
                        style: const TextStyle(
                            fontSize: 20, fontWeight: FontWeight.bold)),
                  ],
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: const Color(0xFFF0F4FF),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.timer_outlined,
                        size: 14, color: Color(0xFF2D5BFF)),
                    const SizedBox(width: 4),
                    Text('${recipe.minutes}분',
                        style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF2D5BFF))),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Text(recipe.description,
              style: TextStyle(
                  fontSize: 14, color: Colors.grey[600], height: 1.5)),
          const SizedBox(height: 16),
          const Text('필요한 재료',
              style:
                  TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: recipe.ingredients
                .map(
                  (ing) => Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF0F4FF),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(ing,
                        style: const TextStyle(
                            fontSize: 13,
                            color: Color(0xFF2D5BFF),
                            fontWeight: FontWeight.w500)),
                  ),
                )
                .toList(),
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton(
              onPressed: () => Navigator.pop(context),
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                side: const BorderSide(color: Color(0xFF2D5BFF)),
              ),
              child: const Text('닫기',
                  style: TextStyle(
                      color: Color(0xFF2D5BFF),
                      fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
    );
  }
}

# ============================================================
# 💾  db_supabase.py — REST API (Supabase/ngrok) 연동
# ============================================================

import requests
from config import API_URL, USER_AGENT, WEIGHT_TOLERANCE
from scale_service import get_dominant


# ── 상품 정보 조회 ───────────────────────────────────────────

def get_product_info_with_retry(barcode: str) -> dict:
    """
    Open Food Facts API로 상품 정보 조회. 실패 시 1회 재시도.
    반환: {"name", "brand", "category"} 또는 {"error": ...}
    """
    import time
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    for attempt in range(1, 3):
        try:
            res = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == 1:
                    p = data["product"]
                    return {
                        "name":     p.get("product_name") or "상품명 없음",
                        "brand":    p.get("brands")       or "브랜드 없음",
                        "category": p.get("categories")   or "카테고리 없음",
                    }
                return {"error": "DB에 등록되지 않은 바코드"}
            return {"error": f"서버 오류 (HTTP {res.status_code})"}
        except Exception as e:
            print(f"  ⚠️  상품 조회 오류 ({attempt}회차): {e}")
            if attempt < 2:
                print("  🔄 1초 후 재시도합니다...")
                time.sleep(1)
    return {"error": "상품 조회 실패 (네트워크 오류)"}


# ── DB 저장 ──────────────────────────────────────────────────

def save_to_db(barcode: str, product: dict, expiry_date,
               weights: dict, product_type: str = "product",
               shelf_life_days: int = None):
    """
    냉장고 아이템을 백엔드 API에 POST로 저장.
    shelf_life_days: 반찬(분기 B)일 때만 전달, 나머지는 None.
    """
    region_num, weight_val = get_dominant(weights)
    data = {
        "barcode":         barcode,
        "product_name":    product.get("name", "상품명 없음"),
        "category":        product.get("category", "카테고리 없음"),
        "expires_at":      expiry_date or None,
        "weight":          weight_val,
        "slot_number":     region_num,
        "product_type":    product_type,
        "shelf_life_days": shelf_life_days,
    }
    print(f"  📤 DB 저장: {data}")
    try:
        requests.post(f"{API_URL}/items", json=data).raise_for_status()
        print(
            f"  💾 저장 완료! → Region{region_num}: {weight_val}g [{product_type}]"
            + (f" / 유통기한 {shelf_life_days}일" if shelf_life_days else "")
        )
    except requests.exceptions.RequestException as e:
        print(f"  ❌ API 호출 실패: {e}")


# ── DB 삭제 ──────────────────────────────────────────────────

def remove_item_by_weight(weights: dict):
    """
    분기 C: 음수 무게 감지 → 오차범위 내 DB 항목을 찾아 삭제.
    """
    _, weight_val = get_dominant(weights)
    abs_weight    = abs(weight_val)

    print(f"  🗑️  제거 무게: {weight_val:+.1f}g → DB에서 ±{WEIGHT_TOLERANCE}g 범위 조회 중...")
    params = {
        "weight_min": abs_weight - WEIGHT_TOLERANCE,
        "weight_max": abs_weight + WEIGHT_TOLERANCE,
    }
    try:
        res = requests.get(f"{API_URL}/items/search", params=params, timeout=5)
        res.raise_for_status()
        matches = res.json()

        if not matches:
            print("  ⚠️  오차범위 내 일치 항목 없음. 삭제 건너뜀.")
            return

        closest = min(matches, key=lambda x: abs(x.get("weight", 0) - abs_weight))
        item_id = closest.get("id")
        print(
            f"  🎯 삭제 대상: id={item_id}, "
            f"name={closest.get('product_name')}, weight={closest.get('weight')}g"
        )
        requests.delete(f"{API_URL}/items/{item_id}", timeout=5).raise_for_status()
        print(f"  💾 항목 삭제 완료! (id={item_id})")

    except requests.exceptions.RequestException as e:
        print(f"  ❌ API 호출 실패: {e}")

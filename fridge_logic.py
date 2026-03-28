# ============================================================
# 🧠  fridge_logic.py — 냉장고 입출고 비즈니스 로직 (분기 A/B/C)
# ============================================================

import time
from datetime import datetime, timedelta
from picamera2 import Picamera2

from config import SHELF_LIFE_MAP
from scale_service import ArduinoScale, get_dominant, measure_with_retry
from ocr_service import capture_frame_with_retry, get_expiry_with_retry
from ai_service import classify_and_get_expiry
from db_supabase import get_product_info_with_retry, save_to_db, remove_item_by_weight


# ── 결과 콘솔 출력 ───────────────────────────────────────────

def print_result(barcode: str, product: dict, expiry, ocr_raw,
                 weights: dict, product_type: str,
                 shelf_life_days: int = None):
    region_num, weight_val = get_dominant(weights)
    label = {
        "product":   "기존 상품",
        "side_dish": "반찬",
        "removed":   "물품 제거",
    }.get(product_type, product_type)
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  🗂️   처리 유형  : {label}")
    print(f"  📦  바코드    : {barcode}")
    if "error" in product:
        print(f"  ⚠️   상품 조회  : {product['error']}")
    else:
        print(f"  🏷️   상품명    : {product['name']}")
        print(f"  🏢  브랜드    : {product.get('brand', '-')}")
        print(f"  📂  카테고리  : {product['category']}")
    print(f"  📅  유통기한  : {expiry or '해당 없음'}"
          + (f"  (OCR: {ocr_raw})" if expiry and ocr_raw else ""))
    if shelf_life_days is not None:
        print(f"  🗓️   유통 일수  : {shelf_life_days}일")
    print(f"  ⚖️   Region1  : {weights.get('Region1', 0.0)} g")
    print(f"  ⚖️   Region2  : {weights.get('Region2', 0.0)} g")
    print(f"  🏆  저장 구역  : Region{region_num} ({weight_val} g)")
    print(sep)


# ── 분기 A — 기존 바코드 상품 ────────────────────────────────

def handle_product(barcode: str, arduino: ArduinoScale, picam2: Picamera2,
                   reader):
    """
    분기 A: 일반 상품 바코드
      ① Open Food Facts 상품 조회
      ② EasyOCR 유통기한 인식
      ③ 무게 측정
      ④ DB 저장
    """
    print("  [분기 A] 기존 상품 처리")

    print("  🌐 상품 정보 조회 중...")
    product = get_product_info_with_retry(barcode)
    if "error" in product:
        print(f"  ⚠️  {product['error']}")
        print("  👉 바코드를 다시 스캔하거나 상품을 확인해주세요.")

    input("\n  📦 유통기한 면을 카메라 앞에 놓은 뒤 Enter 를 누르세요... ")
    print("  📸 촬영 중...")
    expiry_date, ocr_raw, best_bbox, all_results = get_expiry_with_retry(reader, picam2)

    print("\n  ⚖️  무게 측정 중... (선반에 올려주세요)")
    weights = measure_with_retry(arduino, allow_negative=False)

    print_result(barcode, product, expiry_date, ocr_raw, weights, "product")
    save_to_db(barcode, product, expiry_date, weights, product_type="product")


# ── 분기 B — 반찬 등록 ──────────────────────────────────────

def handle_side_dish(arduino: ArduinoScale, picam2: Picamera2):
    """
    분기 B: SideDish 바코드
      ① 내부 바코드 생성
      ② 카메라 촬영 → TFLite 분류 → 유통기한 산출
      ③ 무게 측정
      ④ DB 저장 (shelf_life_days 포함)
    """
    print("  [분기 B] SideDish — 반찬 등록")

    new_barcode = "INT" + datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"  🏷️  내부 바코드: {new_barcode}")

    input("\n  🍱 반찬을 카메라 앞에 놓은 뒤 Enter 를 누르세요... ")
    print("  📸 촬영 중...")
    frame = capture_frame_with_retry(picam2)

    if frame is not None:
        category, expiry_date, shelf_life_days = classify_and_get_expiry(frame)
    else:
        print("  ❌ 카메라 실패 → 기타(3일) 적용")
        category        = "기타"
        shelf_life_days = SHELF_LIFE_MAP["기타"]
        expiry_date     = (
            datetime.now() + timedelta(days=shelf_life_days)
        ).strftime("%Y-%m-%d")

    product = {"name": f"반찬 ({category})", "category": category}

    print("\n  ⚖️  무게 측정 중... (선반에 올려주세요)")
    weights = measure_with_retry(arduino, allow_negative=False)

    print_result(new_barcode, product, expiry_date, None, weights,
                 "side_dish", shelf_life_days)
    save_to_db(new_barcode, product, expiry_date, weights,
               product_type="side_dish", shelf_life_days=shelf_life_days)


# ── 분기 C — 물품 제거 ──────────────────────────────────────

def handle_remove(arduino: ArduinoScale):
    """
    분기 C: TakeOut 바코드
      ① 음수 무게 측정
      ② DB에서 오차범위 내 항목 삭제
    """
    print("  [분기 C] TakeOut — 물품 제거")

    print("\n  ⚖️  무게 변화 측정 중... (물건을 완전히 꺼내주세요)")
    weights = measure_with_retry(arduino, allow_negative=True)

    _, weight_val = get_dominant(weights)
    if weight_val < 0:
        remove_item_by_weight(weights)
    else:
        print("  ⚠️  양수 무게 감지 — 제거 대상을 특정할 수 없습니다.")
        print("  👉 물건을 꺼낸 뒤 다시 TakeOut 바코드를 스캔해주세요.")

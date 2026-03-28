# ============================================================
# 🤖  ai_service.py — TFLite 음식 카테고리 분류
# ============================================================

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime, timedelta
from config import (
    MODEL_PATH, LABEL_PATH, IMG_SIZE, CONF_THRESHOLD,
    FONT_PATH, AI_CAPTURE_IMAGE, SHELF_LIFE_MAP,
)

# ── 전역 모델 상태 ───────────────────────────────────────────
_food_interpreter = None
_food_labels: list = []
_food_input_idx  = None
_food_output_idx = None
_font            = None


def load_food_model():
    """TFLite 모델 + 라벨 + 폰트를 전역 변수에 로드 (최초 1회)."""
    global _food_interpreter, _food_labels, _food_input_idx, _food_output_idx, _font
    print("⏳ 음식 분류 모델 로딩 중...")
    _food_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    _food_interpreter.allocate_tensors()
    in_det  = _food_interpreter.get_input_details()
    out_det = _food_interpreter.get_output_details()
    _food_input_idx  = in_det[0]["index"]
    _food_output_idx = out_det[0]["index"]
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        _food_labels = [line.strip() for line in f.readlines()]
    try:
        _font = ImageFont.truetype(FONT_PATH, 28)
    except Exception:
        _font = ImageFont.load_default()
    print(f"✅ 음식 분류 모델 준비 완료! ({len(_food_labels)}개 라벨)")


# ── 내부 헬퍼 ────────────────────────────────────────────────

def _preprocess_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 프레임 → 모델 입력 텐서 (float32, 정규화 없음)."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def _predict_food(frame_bgr: np.ndarray) -> tuple:
    """TFLite 추론 → (label: str, confidence: float)."""
    tensor = _preprocess_for_model(frame_bgr)
    _food_interpreter.set_tensor(_food_input_idx, tensor)
    _food_interpreter.invoke()
    output   = _food_interpreter.get_tensor(_food_output_idx)[0]
    pred_idx = int(np.argmax(output))
    conf     = float(output[pred_idx])
    label    = _food_labels[pred_idx] if pred_idx < len(_food_labels) else "기타"
    return label, conf


def _put_korean_text(frame_bgr: np.ndarray, text: str, pos: tuple,
                     color_rgb: tuple) -> np.ndarray:
    """PIL로 한글 텍스트 오버레이 후 BGR 반환."""
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=_font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _manual_category_select() -> tuple:
    """
    자동 분류 실패(confidence 낮음) 시 수동 선택 fallback.
    반환: (category: str, days: int)
    """
    cats    = list(SHELF_LIFE_MAP.keys())
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            print(f"\n  🔄 [재입력 {attempt}회차] 카테고리를 다시 선택해주세요.")
        print("\n  🍱 반찬 카테고리를 선택하세요:")
        for i, cat in enumerate(cats, 1):
            print(f"     {i:2}. {cat:<6} ({SHELF_LIFE_MAP[cat]}일)")
        print("      0. 직접 입력")
        ch = input(f"\n  번호 선택 (0~{len(cats)}): ").strip()
        if ch == "0":
            custom = input("  카테고리 직접 입력: ").strip()
            if not custom:
                print("  ⚠️  입력값이 없습니다. 다시 입력해주세요.")
                continue
            days = SHELF_LIFE_MAP.get(custom, SHELF_LIFE_MAP["기타"])
            print(f"  ✅ '{custom}' 선택 → {days}일")
            return custom, days
        if ch.isdigit() and 1 <= int(ch) <= len(cats):
            cat  = cats[int(ch) - 1]
            days = SHELF_LIFE_MAP[cat]
            print(f"  ✅ '{cat}' 선택 → {days}일")
            return cat, days
        print(f"  ⚠️  '{ch}'은 올바른 번호가 아닙니다. 0~{len(cats)} 사이로 입력해주세요.")


# ── 공개 API ─────────────────────────────────────────────────

def classify_and_get_expiry(frame_bgr: np.ndarray) -> tuple:
    """
    TFLite 모델로 자동 분류 → SHELF_LIFE_MAP에서 유통기한 산출.
    confidence < CONF_THRESHOLD 이면 수동 선택 fallback.
    반환: (category: str, expiry_date: str, shelf_life_days: int)
    """
    print("  🤖 AI 식품 분류 중...")
    label, conf = _predict_food(frame_bgr)
    print(f"  🔍 분류 결과: {label}  (confidence: {conf:.2f})")

    color     = (0, 200, 0) if conf >= CONF_THRESHOLD else (200, 0, 0)
    annotated = _put_korean_text(frame_bgr, f"{label} ({conf:.2f})", (20, 20), color)
    cv2.imwrite(AI_CAPTURE_IMAGE, annotated)
    print(f"  📁 분류 결과 이미지 저장: {AI_CAPTURE_IMAGE}")

    if conf >= CONF_THRESHOLD:
        category = label
        if label not in SHELF_LIFE_MAP:
            matched  = [k for k in SHELF_LIFE_MAP if k in label]
            category = matched[0] if matched else "기타"
            if category != label:
                print(f"  🔗 라벨 '{label}' → SHELF_LIFE_MAP '{category}' 매핑")
        days   = SHELF_LIFE_MAP.get(category, SHELF_LIFE_MAP["기타"])
        expiry = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        print(f"  ✅ 자동 분류 완료: '{category}' → 유통기한 {expiry} ({days}일 후)")
        return category, expiry, days
    else:
        print(f"  ⚠️  신뢰도 부족 ({conf:.2f} < {CONF_THRESHOLD}). 수동으로 선택해주세요.")
        category, days = _manual_category_select()
        expiry = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        print(f"  ✅ '{category}' → 유통기한 {expiry} ({days}일 후)")
        return category, expiry, days

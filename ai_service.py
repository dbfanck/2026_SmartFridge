# ============================================================
# 🤖  ai_service.py — TFLite 음식 카테고리 실시간 자동 분류
#
#  ┌─ 핵심 동작 ──────────────────────────────────────────────┐
#  │  1. Picamera2 실시간 스트림에서 매 프레임 TFLite 추론    │
#  │  2. CONF_THRESHOLD 이상 + REQUIRED_STREAK 프레임 연속    │
#  │     → 해당 카테고리로 자동 확정 (수동 선택 없음)          │
#  │  3. 한 번 확정된 라벨은 ABSENCE_RESET_SEC 동안 재확정 X  │
#  │  4. 확정 시 이미지 저장 + 호출자에게 결과 반환           │
#  └──────────────────────────────────────────────────────────┘
# ============================================================

import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime, timedelta

from config import (
    MODEL_PATH, LABEL_PATH, IMG_SIZE,
    CONF_THRESHOLD, FONT_PATH, AI_CAPTURE_IMAGE,
    SHELF_LIFE_MAP,
    REQUIRED_STREAK, ABSENCE_RESET_SEC,
)

# ── 전역 모델 상태 (최초 1회 로드) ──────────────────────────
_interpreter  = None
_labels: list = []
_input_idx    = None
_output_idx   = None
_font         = None


# ============================================================
# 🔧  초기화
# ============================================================

def load_food_model() -> None:
    """TFLite 모델 + 라벨 + 폰트를 전역 변수에 로드 (최초 1회 호출)."""
    global _interpreter, _labels, _input_idx, _output_idx, _font

    print("⏳ 음식 분류 모델 로딩 중...")
    _interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    _interpreter.allocate_tensors()
    _input_idx  = _interpreter.get_input_details()[0]["index"]
    _output_idx = _interpreter.get_output_details()[0]["index"]

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        _labels = [line.strip() for line in f]

    try:
        _font = ImageFont.truetype(FONT_PATH, 32)
    except Exception:
        _font = ImageFont.load_default()

    print(f"✅ 음식 분류 모델 준비 완료! ({len(_labels)}개 라벨)")


# ============================================================
# 🔬  내부 추론 헬퍼
# ============================================================

def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 프레임 → 모델 입력 텐서 (float32)."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def _predict(frame_bgr: np.ndarray) -> tuple:
    """
    TFLite 단일 프레임 추론.
    반환: (label: str, confidence: float)
    """
    _interpreter.set_tensor(_input_idx, _preprocess(frame_bgr))
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_idx)[0]
    idx    = int(np.argmax(output))
    conf   = float(output[idx])
    label  = _labels[idx] if idx < len(_labels) else "기타"
    return label, conf


def _put_korean_text(frame_bgr: np.ndarray, text: str,
                     pos: tuple, color_rgb: tuple) -> np.ndarray:
    """PIL로 한글 텍스트를 프레임에 오버레이 후 BGR 반환."""
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=_font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _annotate_frame(frame_bgr: np.ndarray, label: str, conf: float,
                    stable_count: int, active_labels: set) -> np.ndarray:
    """
    디버그용 오버레이 3줄:
      1. 현재 라벨 + confidence  (초록: 유효 / 빨강: 미달)
      2. 연속 프레임 카운터
      3. 이미 확정된 active 라벨 목록
    """
    is_valid = conf >= CONF_THRESHOLD
    disp     = frame_bgr.copy()
    disp = _put_korean_text(
        disp, f"{label} ({conf:.2f})", (20, 20),
        (0, 255, 0) if is_valid else (255, 0, 0),
    )
    disp = _put_korean_text(
        disp,
        f"stable: {label if is_valid else '-'} / {stable_count}/{REQUIRED_STREAK}",
        (20, 70), (255, 255, 0),
    )
    active_str = ", ".join(sorted(active_labels)) if active_labels else "-"
    disp = _put_korean_text(
        disp, f"active: {active_str}", (20, 120), (0, 255, 255),
    )
    return disp


def _resolve_category(raw_label: str) -> str:
    """
    TFLite 라벨 → SHELF_LIFE_MAP 키로 매핑.
    직접 매칭 실패 시 부분 문자열 매칭 → 없으면 '기타'.
    """
    if raw_label in SHELF_LIFE_MAP:
        return raw_label
    matched = [k for k in SHELF_LIFE_MAP if k in raw_label]
    if matched:
        mapped = matched[0]
        print(f"  🔗 라벨 '{raw_label}' → SHELF_LIFE_MAP '{mapped}' 매핑")
        return mapped
    return "기타"


# ============================================================
# 🚀  공개 API
# ============================================================

def classify_and_get_expiry(picam2) -> tuple:
    """
    Picamera2 실시간 스트림으로 반찬 카테고리를 자동 분류한다.

    동작 흐름
    ─────────
    • 매 프레임마다 TFLite 추론 수행
    • confidence >= CONF_THRESHOLD 이고 같은 라벨이 REQUIRED_STREAK
      프레임 연속 유지되면 해당 카테고리로 자동 확정
    • 확정된 라벨은 ABSENCE_RESET_SEC 초간 재확정 불가
      (카메라에서 사라진 후 다시 나타나야 재확정 가능)
    • 확정 시 AI_CAPTURE_IMAGE 경로에 어노테이션 이미지 저장
    • q 키: 분류 강제 중단 → '기타' 카테고리 반환

    인자
    ────
    picam2 : 이미 start() 된 Picamera2 인스턴스

    반환
    ────
    (category: str, expiry_date: str "YYYY-MM-DD", shelf_life_days: int)
    """
    # ── 상태 변수 초기화 ────────────────────────────────────
    stable_label       = None   # 현재 연속 추적 중인 라벨
    stable_count       = 0      # 연속 유지 프레임 수
    active_sent_labels = set()  # 이미 확정된 라벨 집합 (중복 방지)
    last_seen_time     = {}     # label → 마지막 유효 감지 unix timestamp

    print("\n  📹 실시간 분류 시작 — 반찬을 카메라 앞에 놓으세요.")
    print(f"     조건: 신뢰도 >= {CONF_THRESHOLD:.2f}  /  {REQUIRED_STREAK}프레임 연속")
    print(f"     {ABSENCE_RESET_SEC:.1f}초 화면에서 사라지면 재확정 가능")
    print("     q 키: 강제 종료 ('기타' 적용)\n")

    while True:
        now = time.time()

        # ① 프레임 캡처 (BGR888 보정)
        raw       = picam2.capture_array()
        frame_bgr = raw[:, :, ::-1].copy()

        # ② 단일 프레임 추론
        label, conf = _predict(frame_bgr)
        is_valid    = conf >= CONF_THRESHOLD

        # ③ 유효 감지이면 마지막 감지 시각 갱신
        if is_valid:
            last_seen_time[label] = now

        # ④ ABSENCE_RESET_SEC 초 이상 미감지된 라벨 → active 해제
        expired = [
            lb for lb in list(active_sent_labels)
            if now - last_seen_time.get(lb, 0.0) > ABSENCE_RESET_SEC
        ]
        for lb in expired:
            active_sent_labels.discard(lb)
            print(f"  ↩️  '{lb}' 화면에서 사라짐 → 재확정 가능")

        # ⑤ 연속 프레임 카운터 갱신
        if is_valid:
            if label == stable_label:
                stable_count += 1
            else:
                stable_label = label
                stable_count = 1
        else:
            stable_label = None
            stable_count = 0

        # ⑥ 디버그 오버레이 렌더링 및 화면 표시
        disp = _annotate_frame(frame_bgr, label, conf,
                               stable_count, active_sent_labels)
        cv2.imshow("Smart Fridge AI — 반찬 분류", disp)

        # ⑦ 자동 확정 조건 판단
        if (stable_label is not None
                and stable_count >= REQUIRED_STREAK
                and stable_label not in active_sent_labels):

            confirmed_label = stable_label
            confirmed_conf  = conf

            # 확정 이미지 저장
            cv2.imwrite(AI_CAPTURE_IMAGE, disp)
            print(f"  📁 분류 결과 이미지 저장: {AI_CAPTURE_IMAGE}")

            # active 등록 + 카운터 리셋
            active_sent_labels.add(confirmed_label)
            last_seen_time[confirmed_label] = now
            stable_label = None
            stable_count = 0

            # 카테고리 매핑 → 유통기한 산출
            category = _resolve_category(confirmed_label)
            days     = SHELF_LIFE_MAP.get(category, SHELF_LIFE_MAP["기타"])
            expiry   = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

            print(f"\n  ✅ 자동 분류 확정!")
            print(f"     라벨      : {confirmed_label}  "
                  f"(conf={confirmed_conf:.2f}, {REQUIRED_STREAK}프레임 연속)")
            print(f"     카테고리  : {category}")
            print(f"     유통기한  : {expiry} ({days}일 후)")

            cv2.destroyAllWindows()
            return category, expiry, days

        # ⑧ q 키: 강제 중단
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  ⚠️  분류 강제 종료 → '기타' 적용")
            cv2.destroyAllWindows()
            category = "기타"
            days     = SHELF_LIFE_MAP["기타"]
            expiry   = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            return category, expiry, days

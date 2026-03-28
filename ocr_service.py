# ============================================================
# 🔍  ocr_service.py — EasyOCR 기반 유통기한 인식
# ============================================================

import re
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from picamera2 import Picamera2
from config import RESULT_IMAGE


# ── 텍스트 정규화 ────────────────────────────────────────────

def clean_text_for_date(text: str) -> str:
    s = text.upper().strip()
    s = s.replace(" ", "").replace("EXP", "").replace("BESTBEFORE", "")
    s = s.replace("USEBY", "").replace("LOT", "").replace("까지", "").replace(",", ".")
    return s.translate(str.maketrans({"O": "0", "Q": "0", "D": "0",
                                      "I": "1", "L": "1", "|": "1", "S": "5", "B": "8"}))


def normalize_date_text(text: str, use_rollover: bool = True):
    """
    다양한 날짜 포맷 문자열을 'YYYY-MM-DD' 형식으로 정규화.
    인식 실패 시 None 반환.
    """
    s   = clean_text_for_date(text).strip(".-/ ")
    now = datetime.now()

    for pat in [r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})",
                r"(20\d{2})(\d{2})(\d{2})"]:
        m = re.search(pat, s)
        if m:
            y, mo, d = m.groups()
            try:
                return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
            except ValueError:
                pass

    m = re.search(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", s)
    if m:
        yy, mo, d = m.groups()
        try:
            return datetime(2000 + int(yy), int(mo), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            pass

    m = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?!\d)", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                cand = datetime(now.year, mo, d)
                if use_rollover and cand.date() < now.date():
                    cand = datetime(now.year + 1, mo, d)
                return cand.strftime("%Y-%m-%d")
            except ValueError:
                pass

    m = re.search(r"(?<!\d)(\d{2})(\d{2})(?!\d)", s)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                cand = datetime(now.year, mo, d)
                if use_rollover and cand.date() < now.date():
                    cand = datetime(now.year + 1, mo, d)
                return cand.strftime("%Y-%m-%d")
            except ValueError:
                pass

    return None


# ── 카메라 캡처 ──────────────────────────────────────────────

def capture_frame_with_retry(picam2: Picamera2):
    """
    Picamera2로 프레임 캡처. 실패 시 재시도 여부를 사용자에게 물어봄.
    반환: BGR numpy array 또는 None
    """
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            ans = input(
                f"  🔄 [재시도 {attempt}회차] 카메라를 다시 촬영하시겠습니까? (y/n): "
            ).strip().lower()
            if ans != "y":
                print("  ⏭️  촬영을 건너뜁니다.")
                return None
        try:
            raw       = picam2.capture_array()
            frame_bgr = raw[:, :, ::-1].copy()
            if frame_bgr is None or frame_bgr.size == 0:
                print("  ❌ 빈 프레임 캡처됨.")
                print("  👉 카메라 앞에 물건을 제대로 놓은 뒤 다시 시도해주세요.")
                continue
            print("  ✅ 촬영 완료.")
            return frame_bgr
        except Exception as e:
            print(f"  ❌ 카메라 오류: {e}")
            print("  👉 카메라 연결을 확인한 뒤 다시 시도해주세요.")
            continue


# ── OCR 핵심 ─────────────────────────────────────────────────

def get_expiry_date(reader, frame) -> tuple:
    """
    EasyOCR readtext 결과에서 가장 신뢰도 높은 날짜를 반환.
    반환: (normalized_date, ocr_raw_text, best_bbox, all_results)
    """
    all_results = reader.readtext(frame)
    rows = []
    for bbox, text, conf in all_results:
        norm = normalize_date_text(text)
        if norm:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            rows.append({
                "ocr_text":   text,
                "normalized": norm,
                "confidence": float(conf),
                "area":       (max(xs) - min(xs)) * (max(ys) - min(ys)),
                "bbox":       bbox,
            })
    if not rows:
        return None, None, None, all_results
    df   = pd.DataFrame(rows).sort_values(
        ["confidence", "area"], ascending=[False, False]
    ).reset_index(drop=True)
    best = df.iloc[0]
    return best["normalized"], best["ocr_text"], best["bbox"], all_results


def save_result_image(frame, all_ocr: list, best_bbox):
    """OCR 결과 박스를 오버레이하여 이미지 저장."""
    output = frame.copy()
    for bbox, _, _ in all_ocr:
        cv2.polylines(output, [np.array(bbox, dtype=np.int32)], True, (255, 0, 0), 1)
    if best_bbox is not None:
        cv2.polylines(output, [np.array(best_bbox, dtype=np.int32)], True, (0, 255, 0), 3)
    cv2.imwrite(RESULT_IMAGE, output)
    print(f"  📁 결과 이미지 저장: {RESULT_IMAGE}")


# ── OCR 재시도 루프 ──────────────────────────────────────────

def get_expiry_with_retry(reader, picam2: Picamera2) -> tuple:
    """
    촬영 → OCR → 날짜 인식 실패 시 재촬영 또는 수동 입력 선택.
    반환: (expiry_date, ocr_raw, best_bbox, all_results)
    """
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            print(f"\n  🔄 [재시도 {attempt}회차] 유통기한 촬영을 다시 시도합니다.")

        frame = capture_frame_with_retry(picam2)
        if frame is None:
            ans = input(
                "  ✏️  유통기한을 직접 입력하시겠습니까? (예: 2025-12-31 / 건너뜀: n): "
            ).strip()
            if ans.lower() == "n" or ans == "":
                return None, None, None, []
            norm = normalize_date_text(ans)
            if norm:
                print(f"  📅 수동 입력 유통기한: {norm}")
                return norm, ans, None, []
            print("  ⚠️  날짜 형식을 인식할 수 없습니다. 건너뜁니다.")
            return None, None, None, []

        print("  🔍 OCR 분석 중...")
        expiry_date, ocr_raw, best_bbox, all_results = get_expiry_date(reader, frame)

        if expiry_date:
            save_result_image(frame, all_results, best_bbox)
            return expiry_date, ocr_raw, best_bbox, all_results

        print("  ⚠️  유통기한을 인식하지 못했습니다.")
        save_result_image(frame, all_results, None)
        ans = input("  👉 재촬영(r) / 직접 입력(e) / 건너뜀(n): ").strip().lower()
        if ans == "r":
            continue
        elif ans == "e":
            manual = input("  ✏️  유통기한 직접 입력 (예: 2025-12-31): ").strip()
            norm   = normalize_date_text(manual)
            if norm:
                print(f"  📅 수동 입력 유통기한: {norm}")
                return norm, manual, None, all_results
            print("  ⚠️  날짜 형식 인식 실패. 유통기한 없이 저장합니다.")
            return None, None, None, all_results
        else:
            print("  ⏭️  유통기한 없이 저장합니다.")
            return None, None, None, all_results

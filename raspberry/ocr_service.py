# ============================================================
# 🔍  ocr_service.py — EasyOCR 기반 유통기한 인식 (라즈베리파이 최적화)
# ============================================================
# 최적화 내역:
#   - pandas 제거 → 순수 리스트/정렬 (import 오버헤드 제거)
#   - matplotlib 제거
#   - 이미지 전처리 추가 (그레이스케일 → CLAHE → 샤프닝)
#     : OCR 정확도 향상 → 재시도 횟수 감소
#   - 최신 날짜 파싱 로직 반영
#     : MM.DD.HH:MM / MM.DD.HH.MM 형태 처리
#     : trailing dot (09.23.) 처리
#     : 콜론(:) 기반 순수 시간 제거
#     : 분(>31) 기반 시간 제거
#     : 날짜 후보 중 가장 늦은 날짜 = 유통기한
# ============================================================

import re
import cv2
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
from config import RESULT_IMAGE


# ── 이미지 전처리 ────────────────────────────────────────────

def preprocess_for_ocr(frame: np.ndarray) -> np.ndarray:
    """
    OCR 정확도를 높이기 위한 전처리.
    그레이스케일 → CLAHE(대비 강화) → 샤프닝 → BGR 복원
    라즈베리파이에서 EasyOCR 재시도 횟수를 줄이는 핵심.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE: 조명 불균일 보정 (잉크젯 인쇄 날짜에 효과적)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # 언샤프 마스킹: 텍스트 경계 선명화
    blur     = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp    = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


# ── 텍스트 정제 ──────────────────────────────────────────────

def clean_text_for_date(text: str) -> str:
    s = str(text).upper().strip()
    for kw in ["EXP", "BESTBEFORE", "USEBY", "LOT", "까지"]:
        s = s.replace(kw, "")
    for old, new in [(" ", ""), (",", "."), (";", "."), ("_", ""),
                     ("(", ""), (")", "")]:
        s = s.replace(old, new)
    trans = str.maketrans({"O": "0", "Q": "0", "D": "0",
                            "I": "1", "L": "1", "|": "1",
                            "S": "5", "B": "8", "Z": "2"})
    return s.translate(trans)


# ── 날짜 파싱 ────────────────────────────────────────────────

def normalize_date_text(text: str, use_rollover: bool = True):
    """
    다양한 날짜 포맷 → 'YYYY-MM-DD' 반환. 실패 시 None.

    지원 형태:
      HH:MM          → None (순수 시간)
      MM.DD.HH:MM    → MM.DD 날짜만 추출 (우유팩 형식)
      MM.DD.HH.MM    → 동일
      YYYY.MM.DD     → 4자리 연도
      YYYYMMDD       → 8자리 붙임
      YY.MM.DD 등    → 6자리 구분자
      MM.DD. / MM.DD → trailing dot 포함
      YYMMDD         → 6자리 붙임
    """
    raw = str(text).strip()
    s   = clean_text_for_date(raw).strip(".-/ ")
    now = datetime.now()

    # ── 0) 시간 제거 ───────────────────────────────────────────

    # 순수 HH:MM (콜론) → 시간만 있는 텍스트
    if re.fullmatch(r"\d{1,2}:\d{2}", raw.strip()):
        return None

    # ★ MM.DD.HH:MM 또는 MM.DD.HH.MM (우유팩 형식)
    # ex) "11.14.11:29" → 11월 14일
    m = re.search(r"(?<!\d)(\d{1,2})[.](\d{1,2})[.](\d{1,2})[:.](\d{2})(?!\d)",
                  raw.strip())
    if m:
        mo, d, hh, mm = map(int, m.groups())
        if 0 <= hh <= 23 and 0 <= mm <= 59 and 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                cand = datetime(now.year, mo, d)
                if use_rollover and cand.date() < now.date():
                    cand = datetime(now.year + 1, mo, d)
                return cand.strftime("%Y-%m-%d")
            except ValueError:
                pass

    # 점(.)만으로 된 순수 HH.MM: 분이 32~59면 시간으로 간주
    # ex) "03.55" → 시간 / "09.23" → 날짜
    m = re.fullmatch(r"(\d{1,2})[.](\d{2})[.]?", s)
    if m:
        _, b = map(int, m.groups())
        if 32 <= b <= 59:
            return None

    # ── 1) YYYY.MM.DD / YYYY-MM-DD / YYYY/MM/DD ───────────────
    m = re.search(r"(?<!\d)(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None

    # ── 2) YYYYMMDD ───────────────────────────────────────────
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None

    # ── 3) YY.MM.DD / MM.DD.YY / DD.MM.YY (구분자 6자리) ──────
    m = re.search(r"(?<!\d)(\d{2})[./-](\d{2})[./-](\d{2})(?!\d)", s)
    if m:
        a, b, c = map(int, m.groups())
        for fmt, args in [("YYMMDD", (2000+a, b, c)),
                          ("DDMMYY", (2000+c, b, a)),
                          ("MMDDYY", (2000+c, a, b))]:
            try:
                return datetime(*args).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    # ── 4) MM.DD. / MM.DD / MM-DD (trailing dot 허용) ─────────
    # ex) "09.23." "09.12."
    m = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[.]?$", s)
    if m:
        mo, d = map(int, m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                cand = datetime(now.year, mo, d)
                if use_rollover and cand.date() < now.date():
                    cand = datetime(now.year + 1, mo, d)
                return cand.strftime("%Y-%m-%d")
            except ValueError:
                return None

    # ── 5) YYMMDD (6자리 붙임) ────────────────────────────────
    m = re.search(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", s)
    if m:
        yy, mo, d = map(int, m.groups())
        try:
            return datetime(2000+yy, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


# ── 카메라 캡처 ──────────────────────────────────────────────

def capture_frame_with_retry(picam2: Picamera2):
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

        print("  📷 카메라 프리뷰 중... [Enter] 로 촬영 / [q] 로 취소")
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        frame_bgr = None

        # ── v3 오토포커스: 프리뷰 시작 전 AF 트리거 ──────────────
        try:
            picam2.set_controls({
                "AfMode":    2,   # 2 = Continuous AF (항상 초점 유지)
                "AfSpeed":   1,   # 1 = Fast
                "LensPosition": 0.0,  # 0.0 = 무한대 (가까운 물체면 높여야 함)
            })
            print("  🔍 오토포커스 활성화 (Continuous)")
        except Exception:
            print("  ⚠️  AF 미지원 카메라 (v1/v2) — 수동 모드로 진행")
        # ────────────────────────────────────────────────────────

        while True:
            try:
                raw       = picam2.capture_array()
                frame_bgr = raw[:, :, ::-1].copy()
            except Exception as e:
                print(f"  ❌ 카메라 오류: {e}")
                cv2.destroyWindow("Preview")
                frame_bgr = None
                break

            preview = frame_bgr.copy()
            h, w    = preview.shape[:2]
            cx, cy  = w // 2, h // 2

            # ── 선명도 점수 실시간 표시 ───────────────────────────
            gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            bx1, by1 = int(w * 0.1), int(h * 0.2)
            bx2, by2 = int(w * 0.9), int(h * 0.8)
            roi      = gray[by1:by2, bx1:bx2]
            sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()

            # 선명도에 따라 색상 변경: 빨강→노랑→초록
            if   sharpness > 300: box_color = (0, 255, 0)    # 선명
            elif sharpness > 100: box_color = (0, 255, 255)   # 보통
            else:                 box_color = (0, 0, 255)     # 흐림

            cv2.line(preview, (cx-30, cy), (cx+30, cy), (0,255,0), 1)
            cv2.line(preview, (cx, cy-30), (cx, cy+30), (0,255,0), 1)
            cv2.rectangle(preview, (bx1, by1), (bx2, by2), box_color, 2)

            cv2.putText(preview, "ENTER: Capture  /  Q: Cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,255), 2, cv2.LINE_AA)

            # 선명도 점수 + 상태 텍스트
            focus_label = ("SHARP" if sharpness > 300
                           else "OK" if sharpness > 100 else "BLURRY")
            cv2.putText(preview,
                        f"Focus: {focus_label}  ({sharpness:.0f})",
                        (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, box_color, 2, cv2.LINE_AA)
            # ────────────────────────────────────────────────────

            cv2.imshow("Preview", preview)
            key = cv2.waitKey(30) & 0xFF

            if key == 13:
                # ── 촬영 시 선명도 경고 ──────────────────────────
                if sharpness < 100:
                    print(f"  ⚠️  초점이 흐립니다 (score={sharpness:.0f}). 그래도 촬영합니까? (y/n): ", end="")
                    if input().strip().lower() != "y":
                        continue
                cv2.destroyWindow("Preview")
                print(f"  ✅ 촬영 완료 (선명도={sharpness:.0f})")
                break
            elif key in (ord('q'), ord('Q'), 27):
                cv2.destroyWindow("Preview")
                print("  ⏭️  촬영을 취소했습니다.")
                return None

        if frame_bgr is None or frame_bgr.size == 0:
            print("  ❌ 빈 프레임 캡처됨.")
            continue

        return preprocess_for_ocr(frame_bgr)

# ── OCR 핵심 ─────────────────────────────────────────────────

def get_expiry_date(reader, frame) -> tuple:
    """
    EasyOCR로 날짜 후보를 모두 추출 → 가장 늦은 날짜 = 유통기한 반환.
    (pandas 없이 순수 리스트 정렬로 처리 — 라즈베리파이 메모리/속도 절약)
    반환: (normalized_date, ocr_raw_text, best_bbox, all_results)
    """
    all_results = reader.readtext(frame)

    candidates = []
    for bbox, text, conf in all_results:
        norm = normalize_date_text(text)
        if norm:
            candidates.append({
                "normalized": norm,
                "ocr_text":   text,
                "confidence": float(conf),
                "bbox":       bbox,
            })

    if not candidates:
        return None, None, None, all_results

    # ★ 가장 늦은 날짜 = 유통기한 (동점 시 confidence 높은 것 우선)
    best = max(candidates, key=lambda x: (x["normalized"], x["confidence"]))
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
            if ans.lower() in ("n", ""):
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

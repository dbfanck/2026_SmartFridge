# ============================================================
# 🏷️  barcode_service.py — 바코드 스캐너 시리얼 통신
# ============================================================

import re
import serial
from config import BARCODE_PORT, BAUD_RATE, BARCODE_SIDE_DISH, BARCODE_REMOVE


def open_barcode_port() -> serial.Serial:
    """
    바코드 스캐너 시리얼 포트를 열고 반환.
    실패 시 SerialException 전파.
    """
    ser = serial.Serial(BARCODE_PORT, BAUD_RATE, timeout=1)
    ser.reset_input_buffer()
    print(f"✅ 바코드 포트 열림: {BARCODE_PORT}")
    return ser


def read_barcode(ser: serial.Serial) -> str | None:
    """
    시리얼 버퍼에서 한 줄 읽어 유효한 바코드이면 반환, 아니면 None.

    유효 패턴:
      - 일반 상품 : 8~13자리 숫자
      - 반찬      : BARCODE_SIDE_DISH 상수값
      - 물품 제거  : BARCODE_REMOVE 상수값
    """
    if ser.in_waiting == 0:
        return None
    raw = ser.readline().decode("utf-8", errors="replace").strip()
    if not raw:
        return None
    if (re.fullmatch(r"\d{8,13}", raw)
            or raw == BARCODE_SIDE_DISH
            or raw == BARCODE_REMOVE):
        return raw
    return None


def classify_barcode(barcode: str) -> str:
    """
    바코드 종류를 문자열로 반환.
    반환값: 'side_dish' | 'remove' | 'product'
    """
    if barcode == BARCODE_SIDE_DISH:
        return "side_dish"
    if barcode == BARCODE_REMOVE:
        return "remove"
    return "product"

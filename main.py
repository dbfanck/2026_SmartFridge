# ============================================================
# 🚀  main.py — 스마트 냉장고 스캐너 진입점
# ============================================================

import time
import serial
import easyocr
from picamera2 import Picamera2

from config import ARDUINO_PORT, BARCODE_SIDE_DISH, BARCODE_REMOVE, WEIGHT_MIN_GRAM
from ai_service import load_food_model
from scale_service import ArduinoScale
from barcode_service import open_barcode_port, read_barcode, classify_barcode
from fridge_logic import handle_product, handle_side_dish, handle_remove


def main():
    # ── 음식 분류 모델 로딩 ──────────────────────────────────
    load_food_model()

    # ── EasyOCR 초기화 ───────────────────────────────────────
    print("⏳ EasyOCR 모델 로딩 중...")
    reader = easyocr.Reader(["en"], gpu=False)
    print("✅ OCR 준비 완료!\n")

    # ── Picamera2 초기화 ─────────────────────────────────────
    picam2  = Picamera2()
    cam_cfg = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"},
        controls={"AeEnable": True, "AwbEnable": True, "NoiseReductionMode": 1},
    )
    picam2.configure(cam_cfg)
    picam2.start()
    time.sleep(2)
    print("✅ Picamera2 준비 완료!\n")

    # ── 아두이노 저울 연결 ───────────────────────────────────
    arduino = ArduinoScale(ARDUINO_PORT)
    if not arduino.connect():
        print("❌ 아두이노 연결 실패. 종료합니다.")
        picam2.stop()
        return

    # ── 바코드 스캐너 포트 열기 ──────────────────────────────
    try:
        barcode_ser = open_barcode_port()
    except serial.SerialException as e:
        print(f"❌ 바코드 포트 오류: {e}")
        arduino.close()
        picam2.stop()
        return

    print("\n" + "=" * 50)
    print("🛒  스마트 냉장고 스캐너 준비 완료!")
    print(f"  SideDish 바코드 : {BARCODE_SIDE_DISH}")
    print(f"  TakeOut  바코드 : {BARCODE_REMOVE}")
    print(f"  최소 유효 무게  : {WEIGHT_MIN_GRAM}g 초과")
    print("  종료: Ctrl+C")
    print("=" * 50 + "\n")

    try:
        while True:
            barcode = read_barcode(barcode_ser)
            if barcode is None:
                time.sleep(0.05)
                continue

            print(f"\n✅ [바코드 인식] {barcode}")
            kind = classify_barcode(barcode)

            if kind == "side_dish":
                handle_side_dish(arduino, picam2)
            elif kind == "remove":
                handle_remove(arduino)
            else:
                handle_product(barcode, arduino, picam2, reader)

    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    finally:
        barcode_ser.close()
        arduino.close()
        picam2.stop()
        print("✅ 모든 연결 종료.")


if __name__ == "__main__":
    main()

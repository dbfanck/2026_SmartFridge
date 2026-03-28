# ============================================================
# ⚖️  scale_service.py — 아두이노 저울 통신 및 무게 측정
# ============================================================

import serial
import time
from config import BAUD_RATE, WEIGHT_MIN_GRAM


class ArduinoScale:
    """아두이노와 시리얼 통신하여 무게 변화량/구역을 측정한다."""

    def __init__(self, port: str, timeout: int = 60):
        self.port    = port
        self.timeout = timeout
        self.ser     = None

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, BAUD_RATE, timeout=2)
            self.ser.reset_input_buffer()
            time.sleep(2)
            deadline = time.time() + 10
            while time.time() < deadline:
                if self._readline() == "READY":
                    print(f"✅ 아두이노 연결 완료: {self.port}")
                    return True
                time.sleep(0.1)
            print("⚠️  아두이노 READY 신호 없음 — 그냥 진행합니다.")
            return True
        except serial.SerialException as e:
            print(f"❌ 아두이노 포트 오류 ({self.port}): {e}")
            return False

    def _readline(self) -> str:
        try:
            if self.ser and self.ser.in_waiting > 0:
                return self.ser.readline().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        return ""

    def read_measure_result(self) -> dict:
        if not self.ser:
            return {}
        results  = {}
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            raw = self._readline()
            if not raw:
                time.sleep(0.1)
                continue
            if raw == "MEASURING":
                print("  ⏳ 기준값 측정 중... (선반을 비워두세요)")
            elif raw == "BASE_READY":
                print("  ✅ 기준값 준비 완료!")
                print("  🧺 지금 상품을 선반에 올려주세요! (최대 20초)")
            elif raw == "DONE":
                print(f"  ✅ 수신 완료: {results}")
                return results
            elif raw == "NO_CHANGE":
                print("  ⚠️  감지된 무게 변화 없음")
            elif "," in raw:
                parts = raw.split(",", 1)
                try:
                    region = parts[0].strip()
                    if region.startswith("BASE"):
                        continue
                    weight = float(parts[1].strip().replace("+", ""))
                    results[region] = weight
                    print(f"  📥 {region}: {weight:+.1f} g")
                except ValueError:
                    print(f"  ⚠️  파싱 실패: '{raw}'")
        print(f"  ⚠️  MEASURE 타임아웃 ({self.timeout}초)")
        return results

    def measure(self) -> dict:
        self.ser.reset_input_buffer()
        self.ser.write(b"MEASURE\n")
        return self.read_measure_result()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ── 헬퍼 함수 ────────────────────────────────────────────────

def get_dominant(weights: dict) -> tuple:
    """
    두 Region 중 절댓값이 더 큰 구역 번호와 무게를 반환한다.
    반환: (region_num: int, weight_val: float)
    """
    w1 = weights.get("Region1", 0.0)
    w2 = weights.get("Region2", 0.0)
    if abs(w1) == 0 and abs(w2) == 0:
        return 0, 0.0
    return (1, w1) if abs(w1) >= abs(w2) else (2, w2)


def measure_with_retry(arduino: ArduinoScale, allow_negative: bool = False) -> dict:
    """
    무게 절댓값이 WEIGHT_MIN_GRAM 이하면 재측정 안내 후 반복.
    allow_negative=True : 분기 C (제거) — 음수가 유효한 경우.
    """
    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            print(f"\n  🔄 [재시도 {attempt}회차] 무게를 다시 측정합니다...")

        weights  = arduino.measure()
        _, w_val = get_dominant(weights)

        if abs(w_val) <= WEIGHT_MIN_GRAM:
            print(f"  ⚠️  감지된 무게 {w_val:+.1f}g — {WEIGHT_MIN_GRAM}g 이하입니다.")
            if allow_negative:
                print("  👉 물건을 완전히 꺼낸 뒤 다시 시도해주세요.")
            else:
                print("  👉 물건을 선반 위에 올린 뒤 다시 시도해주세요.")
            continue

        if not allow_negative and w_val < 0:
            print(f"  ⚠️  음수 무게({w_val:+.1f}g) 감지 — 입고 분기에서는 양수여야 합니다.")
            print("  👉 물건을 다시 올려주세요.")
            continue

        print(f"  ✅ 유효 무게 확인: {w_val:+.1f}g")
        return weights

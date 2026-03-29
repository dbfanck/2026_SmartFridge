#include "HX711.h"

#define calibration_factor2 -56.4
#define calibration_factor1 -90.1

const int DOUT1 = 7, CLK1 = 6;
const int DOUT2 = 3, CLK2 = 2;
HX711 scale1, scale2;

const float THRESHOLD   = 5.0;
const float STABLE_DIFF = 2.0;
const int   STABLE_CNT  = 5;
const unsigned long STABLE_WAIT  = 300;
const unsigned long DETECT_TIMEOUT = 20000;  // 물건 올릴 때까지 대기 20초 (Python 60초 > 아두이노 30초 보장)
const unsigned long STABLE_TIMEOUT = 10000;  // 안정화 타임아웃 10초

// ── 안정화될 때까지 기다린 뒤 무게 반환 ───────────────────
float getStableWeight(HX711 &scale) {
  float cur = scale.get_units(5);
  if (cur < 0) cur = 0;
  unsigned long t = millis();
  int stableCnt = 0;

  while (millis() - t < STABLE_TIMEOUT) {
    delay(STABLE_WAIT);
    float next = scale.get_units(5);
    if (next < 0) next = 0;
    if (abs(next - cur) <= STABLE_DIFF) {
      stableCnt++;
      if (stableCnt >= STABLE_CNT) return next;
    } else {
      stableCnt = 0;
      cur = next;
    }
  }
  float v = scale.get_units(10);
  return v < 0 ? 0 : v;
}

void setup() {
  Serial.begin(9600);
  scale1.begin(DOUT1, CLK1);
  scale1.set_scale(calibration_factor1);
  scale1.tare();
  scale2.begin(DOUT2, CLK2);
  scale2.set_scale(calibration_factor2);
  scale2.tare();
  delay(1000);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // ── MEASURE ────────────────────────────────────────────
    if (cmd == "MEASURE") {
      Serial.println("MEASURING");

      // 1) 기준값 스냅샷 (선반 빈 상태)
      float base1 = scale1.get_units(10); if (base1 < 0) base1 = 0;
      float base2 = scale2.get_units(10); if (base2 < 0) base2 = 0;

      // 2) 기준값 전송 후 BASE_READY 신호 ← Python이 이걸 받고 "올리세요" 표시
      Serial.print("BASE1,"); Serial.println(base1, 1);
      Serial.print("BASE2,"); Serial.println(base2, 1);
      Serial.println("BASE_READY");   // ★ 이 신호 이후에 사용자가 물건 올림

      // 3) 변화 감지 대기 (둘 다 감지될 때까지 계속 루프)
      unsigned long t = millis();
      bool detected1 = false, detected2 = false;
      unsigned long firstDetectTime = 0;  // 첫 감지 시각

      while (millis() - t < DETECT_TIMEOUT) {
        float w1 = scale1.get_units(5); if (w1 < 0) w1 = 0;
        float w2 = scale2.get_units(5); if (w2 < 0) w2 = 0;
        if (!detected1 && abs(w1 - base1) >= THRESHOLD) {
          detected1 = true;
          if (firstDetectTime == 0) firstDetectTime = millis();
        }
        if (!detected2 && abs(w2 - base2) >= THRESHOLD) {
          detected2 = true;
          if (firstDetectTime == 0) firstDetectTime = millis();
        }
        // 첫 감지 후 3초 더 기다려 나머지 구역도 감지
        if ((detected1 || detected2) &&
            firstDetectTime > 0 &&
            millis() - firstDetectTime >= 3000) break;
        delay(200);
      }

      // 4) 변화 구역 안정화 후 delta 전송
      if (detected1) {
        float stable = getStableWeight(scale1);
        float diff   = stable - base1;
        Serial.print("Region1,");
        if (diff >= 0) Serial.print("+");
        Serial.println(diff, 1);
      }
      if (detected2) {
        float stable = getStableWeight(scale2);
        float diff   = stable - base2;
        Serial.print("Region2,");
        if (diff >= 0) Serial.print("+");
        Serial.println(diff, 1);
      }
      if (!detected1 && !detected2) {
        Serial.println("NO_CHANGE");
      }
      Serial.println("DONE");

    // ── TARE ───────────────────────────────────────────────
    } else if (cmd == "TARE") {
      scale1.tare();
      scale2.tare();
      Serial.println("TARE_DONE");

    } else {
      Serial.print("UNKNOWN_CMD:");
      Serial.println(cmd);
    }
  }
}

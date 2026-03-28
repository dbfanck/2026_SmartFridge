# 🧊 Smart Fridge Scanner

라즈베리파이 + 아두이노 기반 스마트 냉장고 관리 시스템.
바코드 스캔 → OCR/AI 분류 → 무게 측정 → DB 자동 저장까지 한 번에 처리합니다.

---

## 📁 파일 트리

```
2026_SmartFridge/
├── config.py              # ⚙️  전체 환경 설정 및 상수
├── db_supabase.py         # 💾  REST API (Supabase/ngrok) 연동
├── barcode_service.py     # 🏷️  바코드 스캐너 시리얼 통신
├── ocr_service.py         # 🔍  EasyOCR 기반 유통기한 인식
├── ai_service.py          # 🤖  TFLite 음식 카테고리 분류
├── scale_service.py       # ⚖️  아두이노 저울 통신 및 무게 측정
├── fridge_logic.py        # 🧠  입출고 비즈니스 로직 (분기 A/B/C)
├── main.py                # 🚀  진입점 — 메인 루프
├── labels.txt             # 🏷️  TFLite 모델 라벨 목록
└── kfood_mobilenetv2.tflite  # 🤖  한국 음식 분류 TFLite 모델
```

---

## 🔄 시스템 알고리즘 (블록 다이어그램)

```
                        ┌──────────────┐
                        │  바코드 인식  │
                        └──────┬───────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐
   │ 물품(기존바코드)│  │ 반찬(새바코드생성)│  │물품제거(새바코드생성)│
   └───────┬───────┘  └────────┬────────┘  └────────┬─────────┘
           │                   │                    │
           ▼                   ▼                    │
      ┌─────────┐    ┌──────────────────┐           │
      │ EasyOCR │    │Food Category Cls.│           │
      └────┬────┘    └────────┬─────────┘           │
           │                   │                    │
           ▼                   ▼                    │
    ┌──────────────┐  ┌─────────────────┐           │
    │ 유통기한 인식 │  │유통기한 데이터셋 │           │
    └──────┬───────┘  │     분류        │           │
           └──────────┴────────┬────────┘           │
                               ▼                    │
                    ┌─────────────────────┐         │
                    │  무게 변화량/구역 인식│◄────────┘
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │    데이터 기록 갱신   │
                    └──────────┬──────────┘
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
       바코드 데이터     유통기한 데이터        동일 무게량 확인
          저장              저장           오차범위 내 정보 삭제
              └────────────────┼────────────────────┘
                               ▼
                             [ DB ]
```

---

## ⚡ 처리 분기 요약

| 분기 | 트리거 바코드 | 주요 처리 |
|------|-------------|-----------|
| **A — 기존 상품** | 8~13자리 숫자 바코드 | Open Food Facts 조회 → EasyOCR 유통기한 → 무게 측정 → DB 저장 |
| **B — 반찬 등록** | `SideDish 1001001001` | 내부 바코드 생성 → TFLite 음식 분류 → 유통기한 산출 → 무게 측정 → DB 저장 |
| **C — 물품 제거** | `TakeOut 0000000000000` | 음수 무게 측정 → DB 오차범위 내 항목 삭제 |

---

## 🛠️ 모듈 역할 상세

### `config.py`
모든 설정값과 상수를 한 곳에서 관리합니다.
- 시리얼 포트, API URL, 모델 경로
- 고정 바코드 문자열
- `SHELF_LIFE_MAP` — 반찬 카테고리별 유통기한 (일)

### `barcode_service.py`
바코드 스캐너 시리얼 포트를 열고, 읽은 값의 유효성을 검사합니다.
- `open_barcode_port()` — 포트 오픈
- `read_barcode()` — 버퍼에서 1줄 읽기 + 패턴 검증
- `classify_barcode()` — `'product'` / `'side_dish'` / `'remove'` 분류

### `ocr_service.py`
Picamera2로 촬영 후 EasyOCR로 유통기한을 인식합니다.
- 다양한 날짜 포맷 정규화 (`normalize_date_text`)
- 인식 실패 시 재촬영 / 수동 입력 fallback

### `ai_service.py`
MobileNetV2 기반 TFLite 모델로 반찬 카테고리를 분류합니다.
- `load_food_model()` — 최초 1회 모델 로드
- `classify_and_get_expiry()` — 분류 → `SHELF_LIFE_MAP`에서 유통기한 산출
- confidence < 0.60 → 수동 카테고리 선택 fallback

### `scale_service.py`
아두이노와 시리얼 통신하여 두 구역(Region1/2)의 무게 변화를 측정합니다.
- `ArduinoScale` 클래스 — 연결/측정/종료
- `get_dominant()` — 더 큰 무게 변화 구역 선택
- `measure_with_retry()` — 10g 이하 측정값 재시도

### `db_supabase.py`
백엔드 REST API와 통신하여 아이템을 저장하거나 삭제합니다.
- `get_product_info_with_retry()` — Open Food Facts 조회
- `save_to_db()` — POST `/items`
- `remove_item_by_weight()` — GET `/items/search` → DELETE `/items/{id}`

### `fridge_logic.py`
세 분기의 비즈니스 로직을 담당합니다.
- `handle_product()` — 분기 A
- `handle_side_dish()` — 분기 B
- `handle_remove()` — 분기 C

### `main.py`
모든 서비스를 초기화하고 바코드 이벤트 루프를 실행합니다.

---

## 🔧 하드웨어 구성

| 장치 | 역할 | 연결 |
|------|------|------|
| Raspberry Pi 4 | 메인 컨트롤러 | — |
| Picamera2 | 유통기한 OCR / 반찬 분류 촬영 | CSI |
| Arduino (HX711 × 2) | 2구역 무게 측정 | `/dev/ttyUSB0` |
| 바코드 스캐너 | 상품/반찬/제거 트리거 | `/dev/ttyACM0` |

---

## 📦 설치

```bash
pip install easyocr tflite-runtime picamera2 pyserial requests pandas opencv-python pillow
```

### 모델 파일 배치
```
2026_SmartFridge/
├── kfood_mobilenetv2.tflite   # 한국 음식 분류 모델
└── labels.txt                  # 카테고리 라벨 (한 줄에 하나)
```

---

## 🚀 실행

```bash
cd 2026_SmartFridge
python main.py
```

---

## ⚙️ 주요 설정 (`config.py`)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `BARCODE_PORT` | `/dev/ttyACM0` | 바코드 스캐너 포트 |
| `ARDUINO_PORT` | `/dev/ttyUSB0` | 아두이노 포트 |
| `CONF_THRESHOLD` | `0.60` | AI 자동 분류 최소 신뢰도 |
| `WEIGHT_MIN_GRAM` | `10.0` | 유효 무게 최솟값 (g) |
| `WEIGHT_TOLERANCE` | `10.0` | 제거 시 DB 조회 오차 범위 (g) |
| `API_URL` | ngrok URL | 백엔드 API 주소 |

# 🧊 Smart Fridge Scanner

라즈베리파이 + 아두이노 기반 스마트 냉장고 관리 시스템.
바코드 스캔 → OCR/AI 분류 → 무게 측정 → DB 자동 저장 → 앱 조회/레시피 추천까지 한 번에 처리합니다.

---

## 📁 프로젝트 구조

```
2026_SmartFridge/
├── raspberry/                  # 🍓 라즈베리파이 실행 코드
│   ├── main.py                 #   진입점 — 메인 루프
│   ├── config.py               #   전체 환경 설정 및 상수
│   ├── barcode_service.py      #   바코드 스캐너 시리얼 통신
│   ├── ocr_service.py          #   EasyOCR 기반 유통기한 인식
│   ├── ai_service.py           #   TFLite 음식 카테고리 분류
│   ├── scale_service.py        #   아두이노 저울 통신 및 무게 측정
│   ├── fridge_logic.py         #   입출고 비즈니스 로직 (분기 A/B/C)
│   └── db_supabase.py          #   백엔드 REST API 연동
│   ├── labels.txt              #   TFLite 모델 라벨 목록
│   └── kfood_mobilenetv2.tflite#  한국 음식 분류 TFLite 모델
│
├── fridge_backend/             # ⚙️  FastAPI 백엔드 서버
│   ├── main.py                 #   앱 초기화 및 라우터 등록
│   ├── requirements.txt        #   Python 의존성
│   ├── db/
│   │   └── connection.py       #   Supabase 클라이언트 초기화
│   └── routers/
│       ├── items.py            #   냉장고 아이템 CRUD
│       ├── layouts.py          #   냉장고 레이아웃 관리
│       └── analysis.py         #   통계 조회 + Gemini 레시피 추천
│
└── fridge_frontend/            # 📱 Flutter 모바일 앱
    ├── lib/
    │   └── main.dart           #   앱 진입점
    └── pubspec.yaml            #   Flutter 의존성
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

## 🍓 raspberry/ — 라즈베리파이 코드

### 모듈 역할

| 파일 | 역할 |
|------|------|
| `config.py` | 시리얼 포트, API URL, 모델 경로, `SHELF_LIFE_MAP` 등 모든 상수 관리 |
| `barcode_service.py` | 포트 오픈, 바코드 1줄 읽기, `product`/`side_dish`/`remove` 분류 |
| `ocr_service.py` | Picamera2 촬영 → EasyOCR → 날짜 포맷 정규화, 실패 시 수동 입력 fallback |
| `ai_service.py` | MobileNetV2 TFLite 모델 로드 → 반찬 분류 → 유통기한 산출, confidence < 0.60 시 수동 fallback |
| `scale_service.py` | 아두이노 시리얼 통신, 2구역(Region1/2) 무게 변화 측정, 10g 이하 재시도 |
| `db_supabase.py` | Open Food Facts 조회, `POST /items` 저장, `DELETE /items/{id}` 삭제 |
| `fridge_logic.py` | 분기 A(`handle_product`) / B(`handle_side_dish`) / C(`handle_remove`) 로직 |
| `main.py` | 전체 서비스 초기화 및 바코드 이벤트 루프 |

### 설치 및 실행

모델 파일을 `raspberry/` 폴더에 배치:
```
raspberry/
├── kfood_mobilenetv2.tflite
└── labels.txt
```

```bash
pip install easyocr tflite-runtime picamera2 pyserial requests pandas opencv-python pillow

cd raspberry
python main.py
```

### 주요 설정 (`config.py`)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `BARCODE_PORT` | `/dev/ttyACM0` | 바코드 스캐너 포트 |
| `ARDUINO_PORT` | `/dev/ttyUSB0` | 아두이노 포트 |
| `CONF_THRESHOLD` | `0.60` | AI 자동 분류 최소 신뢰도 |
| `WEIGHT_MIN_GRAM` | `10.0` | 유효 무게 최솟값 (g) |
| `WEIGHT_TOLERANCE` | `10.0` | 제거 시 DB 조회 오차 범위 (g) |
| `API_URL` | Render URL | 백엔드 API 주소 |

---

## ⚙️ fridge_backend/ — FastAPI 백엔드

Supabase DB와 연동하며 라즈베리파이 및 Flutter 앱의 요청을 처리합니다.

### API 엔드포인트

**Items (`/items`)**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/items` | 냉장고 아이템 추가 |
| `DELETE` | `/items/{item_id}` | 아이템 삭제 |
| `GET` | `/items/search?weight_min=&weight_max=` | 무게 범위로 아이템 조회 |
| `GET` | `/items/expiring` | 유통기한 3일 이내 임박 아이템 조회 |
| `GET` | `/items/recent` | 최근 추가된 아이템 5개 조회 |
| `PATCH` | `/items/update-spoiled` | `expires_at` 기준 상한 여부 일괄 업데이트 |

**Analysis (`/analysis`)**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/analysis/items` | 전체 식재료 조회 |
| `GET` | `/analysis/stats/category` | 카테고리별 구매/폐기 통계 |
| `GET` | `/analysis/stats/overall` | 전체 구매/폐기 통계 + 폐기율 |
| `GET` | `/analysis/recipe` | Gemini AI 레시피 추천 |

### 환경 변수

`fridge_backend/` 루트에 `.env` 파일 생성:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
GEMINI_API_KEY=your-gemini-api-key
```

### 설치 및 실행

```bash
cd fridge_backend
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 📱 fridge_frontend/ — Flutter 앱

백엔드 REST API를 호출하여 냉장고 현황 조회, 통계, 레시피 추천 기능을 제공합니다.

### 설치 및 실행

```bash
cd fridge_frontend
flutter pub get
flutter run
```

---

## 🔧 하드웨어 구성

| 장치 | 역할 | 연결 |
|------|------|------|
| Raspberry Pi 4 | 메인 컨트롤러 | — |
| Picamera2 | 유통기한 OCR / 반찬 분류 촬영 | CSI |
| Arduino (HX711 × 2) | 2구역 무게 측정 | `/dev/ttyUSB0` |
| 바코드 스캐너 | 상품/반찬/제거 트리거 | `/dev/ttyACM0` |

---

## 👥 팀원 소개
<table>
<tr>
    <td align="center">
        <img src="https://github.com/brokenbruise.png" width="100px;" alt="이수진"/><br />
        <b>이수진</b><br />
        <a href="https://github.com/brokenbruise">@brokenbruise</a>
    </td>
    <td align="center">
        <img src="https://github.com/Jeong0922.png" width="100px;" alt="정재헌"/><br />
        <b>정재헌</b><br />
        <a href="https://github.com/Jeong0922">@Jeong0922</a>
    </td>
    <td align="center">
        <img src="https://github.com/hayul0419.png" width="100px;" alt="조하율"/><br />
        <b>조하율</b><br />
        <a href="https://github.com/hayul0419">@hayul0419</a>
    </td>
    <td align="center">
        <img src="https://github.com/dbfanck.png" width="100px;" alt="손민"/><br />
        <b>손민</b><br />
        <a href="https://github.com/dbfanck">@dbfanck</a>
    </td>
</tr>
</table>

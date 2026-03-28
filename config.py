# ============================================================
# ⚙️  config.py — 전체 환경 설정 및 상수
# ============================================================

# ── 시리얼 포트 ──────────────────────────────────────────────
BARCODE_PORT = '/dev/ttyACM0'
ARDUINO_PORT = '/dev/ttyUSB0'
BAUD_RATE    = 9600

# ── API / 네트워크 ───────────────────────────────────────────
USER_AGENT  = "MySmartFridge/1.0 (wjdwogjs10@gmail.com)"
API_URL     = "https://frolicly-hypercivilized-lilah.ngrok-free.dev"

# ── 이미지 저장 경로 ─────────────────────────────────────────
RESULT_IMAGE    = "result_ocr.jpg"
AI_CAPTURE_IMAGE = "ai_captured.jpg"

# ── AI 분류 모델 ─────────────────────────────────────────────
MODEL_PATH     = "kfood_mobilenetv2.tflite"
LABEL_PATH     = "labels.txt"
IMG_SIZE       = (160, 160)
CONF_THRESHOLD = 0.60          # 이 값 미만이면 수동 선택 fallback
FONT_PATH      = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# ── 고정 바코드 ──────────────────────────────────────────────
BARCODE_SIDE_DISH = "SideDish 1001001001"   # 반찬 등록
BARCODE_REMOVE    = "TakeOut 0000000000000" # 물품 제거

# ── 무게 관련 상수 ───────────────────────────────────────────
WEIGHT_MIN_GRAM  = 10.0   # 이 값 이하면 재측정
WEIGHT_TOLERANCE = 10.0   # 제거 시 DB 조회 오차 범위 (g)

# ── 반찬 카테고리별 유통기한 테이블 (단위: 일) ─────────────
SHELF_LIFE_MAP = {
    "구이":   3,
    "국":     3,
    "기타":   3,
    "김치":  30,
    "나물":   3,
    "떡":     2,
    "만두":   3,
    "면":     3,
    "무침":   2,
    "밥":     2,
    "볶음":   4,
    "쌈":     2,
    "음청류": 7,
    "장":    90,
    "장아찌":30,
    "적":     3,
    "전":     3,
    "전골":   3,
    "조림":   5,
    "죽":     3,
    "찌개":   3,
    "찜":     3,
    "탕":     3,
    "튀김":   2,
    "한과":  14,
    "해물":   2,
    "회":     1,
}

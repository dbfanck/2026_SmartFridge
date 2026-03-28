import os
import json
import re
from google import genai
from fastapi import APIRouter, HTTPException
from db.connection import supabase
from collections import defaultdict

router = APIRouter()


# 1. 전체 식재료 조회 (LLM 레시피 추천용)
@router.get("/analysis/items")
def get_all_items():
    response = (
        supabase.table("FRIDGE_ITEM")
        .select("item_id, product_name, category, is_spoiled")
        .execute()
    )
    return response.data


# 2. 카테고리별 구매/폐기 통계
@router.get("/analysis/stats/category")
def get_category_stats():
    response = (
        supabase.table("FRIDGE_ITEM")
        .select("category, is_spoiled")
        .execute()
    )

    stats: dict[str, dict] = defaultdict(lambda: {"purchased": 0, "disposed": 0})
    for item in response.data:
        category = item.get("category") or "기타"
        stats[category]["purchased"] += 1
        if item.get("is_spoiled"):
            stats[category]["disposed"] += 1

    return [
        {"category": cat, "purchased": v["purchased"], "disposed": v["disposed"]}
        for cat, v in stats.items()
    ]


# 3. 전체 구매/폐기 통계 + 폐기율
@router.get("/analysis/stats/overall")
def get_overall_stats():
    response = (
        supabase.table("FRIDGE_ITEM")
        .select("is_spoiled")
        .execute()
    )

    total = len(response.data)
    disposed = sum(1 for item in response.data if item.get("is_spoiled"))
    dispose_rate = round(disposed / total * 100, 1) if total > 0 else 0.0

    return {
        "total_purchased": total,
        "total_disposed": disposed,
        "dispose_rate": dispose_rate,
    }


# 4. Gemini AI 레시피 추천
@router.get("/analysis/recipe")
def get_recipe_recommendation():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "여기에_API_키_입력":
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY가 설정되지 않았습니다.")

    response = (
        supabase.table("FRIDGE_ITEM")
        .select("product_name, category")
        .eq("is_spoiled", False)
        .execute()
    )

    items = response.data
    if not items:
        raise HTTPException(status_code=404, detail="냉장고에 식재료가 없습니다.")

    item_list = ", ".join(
        f"{item['product_name']}({item['category'] or '기타'})" for item in items
    )

    prompt = f"""당신은 요리 전문가입니다. 아래 냉장고 식재료를 최대한 활용해 만들 수 있는 요리 1가지를 추천해 주세요.

냉장고 식재료: {item_list}

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.
{{
  "name": "요리 이름",
  "description": "요리에 대한 한두 문장 설명",
  "ingredients": ["재료1 (분량)", "재료2 (분량)", "..."],
  "minutes": 조리시간(정수)
}}"""

    client = genai.Client(api_key=api_key)
    result = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )
    text = result.text.strip()

    # 마크다운 코드블록 제거
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        recipe = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI 응답 파싱에 실패했습니다.")

    return {
        "name": str(recipe.get("name", "")),
        "description": str(recipe.get("description", "")),
        "ingredients": list(recipe.get("ingredients", [])),
        "minutes": int(recipe.get("minutes", 0)),
    }

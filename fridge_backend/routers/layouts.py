from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db.connection import supabase
from datetime import datetime, timezone, timedelta

router = APIRouter()


# 냉장고 레이아웃 조회 (FRIDGE_ITEM조인)
@router.get("/layouts")
def get_layouts():
    response = supabase.table("FRIDGE_ITEM").select("item_id, product_name, slot_number, expires_at").execute()
    return response.data

# 특정 아이템 상세 조회 (item_id 기준)
@router.get("/layouts/{item_id}")
def get_item_detail(item_id: int):
    response = (
        supabase.table("FRIDGE_ITEM")
        .select("product_name, category, created_at, expires_at")
        .eq("item_id", item_id)
        .single()
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="Item not found")

    return response.data

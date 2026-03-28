from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db.connection import supabase
from datetime import datetime, timezone, timedelta

router = APIRouter()

class ItemCreate(BaseModel):
    barcode: str
    product_name: str
    category: str
    expires_at: Optional[datetime] = None
    weight: float
    slot_number: int

# 냉장고 물건 추가
@router.post("/items", status_code=201)
def create_item(item: ItemCreate):
    response = supabase.table("FRIDGE_ITEM").insert({
        "barcode": item.barcode,
        "product_name": item.product_name,
        "category": item.category,
        "expires_at": item.expires_at.isoformat() if item.expires_at else None,
        "weight": item.weight,
        "slot_number": item.slot_number
    }).execute()

    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to insert item")

    return response.data[0]

# 상한 음식 여부 업데이트 (expires_at 기준)
@router.patch("/items/update-spoiled")
def update_spoiled_items():
    now = datetime.now(timezone.utc)

    response = supabase.table("FRIDGE_ITEM").select("item_id, expires_at").execute()
    if not response.data:
        return {"updated": 0}

    spoiled_ids = [
        item["item_id"] for item in response.data
        if datetime.fromisoformat(item["expires_at"]) <= now
    ]
    not_spoiled_ids = [
        item["item_id"] for item in response.data
        if datetime.fromisoformat(item["expires_at"]) > now
    ]

    if spoiled_ids:
        supabase.table("FRIDGE_ITEM").update({"is_spoiled": True}).in_("item_id", spoiled_ids).execute()
    if not_spoiled_ids:
        supabase.table("FRIDGE_ITEM").update({"is_spoiled": False}).in_("item_id", not_spoiled_ids).execute()

    return {"updated": len(spoiled_ids) + len(not_spoiled_ids), "spoiled": len(spoiled_ids), "fresh": len(not_spoiled_ids)}

# 유통기한 임박 조회 (3일 이내)
@router.get("/items/expiring")
def get_expiring_items():
    now = datetime.now(timezone.utc)
    soon = now + timedelta(days=3)

    response = (
        supabase.table("FRIDGE_ITEM").select("item_id, product_name, category, expires_at")
        .gte("expires_at", now.isoformat()).lte("expires_at", soon.isoformat()).execute()
    )
    return response.data

# 최근 활동 조회 (5개)
@router.get("/items/recent")
def get_recent_items():
    response = supabase.table("FRIDGE_ITEM").select("product_name, created_at").order("created_at", desc=True).limit(5).execute()
    return response.data

# 아이템 삭제
@router.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    response = supabase.table("FRIDGE_ITEM").delete().eq("item_id", item_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Item not found")

# weight 범위로 아이템 조회
@router.get("/items/search")
def search_items_by_weight(weight_min: float, weight_max: float):
    response = (
        supabase.table("FRIDGE_ITEM").select("*")
        .gte("weight", weight_min).lte("weight", weight_max).execute()
    )
    return response.data
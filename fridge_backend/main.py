from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import items, layouts, analysis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items.router)
app.include_router(layouts.router)
app.include_router(analysis.router)
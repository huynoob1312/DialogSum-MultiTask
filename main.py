import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.services.gru_service import gru_service



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Lifespan startup: Đang nạp trước mô hình GRU...")
    gru_service.load_model()
    yield
    print("[*] Lifespan shutdown: Dọn dẹp tài nguyên...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Tóm tắt hội thoại & Trích xuất chủ đề (DialogSum)",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health Check"])
async def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

from fastapi import APIRouter, HTTPException
from app.schemas.dialogue import DialogueRequest, DialogueResponse
from app.services import gru_service, transformer_service
from app.services.bart_service import bart_service

router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post(
    "/predict/gru",
    response_model=DialogueResponse,
    summary="Dự đoán Topic và Summary sử dụng mô hình Seq2Seq-GRU",
    description="Nhận vào chuỗi hội thoại, trả về chủ đề (Topic) và bản tóm tắt (Summary) cùng thời gian phản hồi."
)
async def predict_with_gru(request: DialogueRequest):
    try:
        result = gru_service.predict(
            dialogue=request.dialogue,
            max_summary_len=request.max_summary_len,
            max_topic_len=request.max_topic_len
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình inference GRU: {str(e)}"
        )


@router.post(
    "/predict/transformer",
    response_model=DialogueResponse,
    summary="Dự đoán Topic và Summary sử dụng mô hình Seq2Seq-Transformer",
    description="Nhận vào chuỗi hội thoại, trả về chủ đề (Topic) và bản tóm tắt (Summary) bằng Seq2Seq Transformer custom."
)
async def predict_with_transformer(request: DialogueRequest):
    try:
        result = transformer_service.predict(
            dialogue=request.dialogue,
            max_summary_len=request.max_summary_len,
            max_topic_len=request.max_topic_len
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình inference Transformer: {str(e)}"
        )

@router.post(
    "/predict/bart",
    response_model=DialogueResponse,
    summary="Dự đoán Topic và Summary sử dụng mô hình BART",
    description="Nhận vào chuỗi hội thoại, trả về chủ đề (Topic) và bản tóm tắt (Summary) bằng BART."
)
async def predict_with_bart(request: DialogueRequest):
    try:
        result = bart_service.predict(
            dialogue=request.dialogue,
            max_summary_len=request.max_summary_len,
            max_topic_len=request.max_topic_len
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình inference BART: {str(e)}"
        )
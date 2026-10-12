from pydantic import BaseModel, Field


class DialogueRequest(BaseModel):
    dialogue: str = Field(
        ...,
        description="Đoạn hội thoại đầu vào cần tóm tắt và trích xuất chủ đề",
        min_length=5
    )
    max_summary_len: int = Field(
        default=64,
        description="Độ dài tối đa của tóm tắt (số tokens)",
        ge=10,
        le=256
    )
    max_topic_len: int = Field(
        default=25,
        description="Độ dài tối đa của chủ đề (số tokens)",
        ge=5,
        le=64
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "dialogue": "#Person1#: Happy Birthday, this is for you, Brian.\n#Person2#: I'm so happy you remember, please come in and enjoy the party!\n#Person1#: May I have a pleasure to dance with you?\n#Person2#: Sure!",
                "max_summary_len": 64,
                "max_topic_len": 25
            }
        }
    }


class DialogueResponse(BaseModel):
    model_name: str = Field(..., description="Tên mô hình đã thực thi")
    topic: str = Field(..., description="Chủ đề được AI dự đoán")
    summary: str = Field(..., description="Bản tóm tắt được AI sinh ra")
    execution_time_ms: float = Field(..., description="Thời gian xử lý tính bằng mili-giây (ms)")

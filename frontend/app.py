import os
import sys
import time
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.gru_service import GRUService
from app.services.transformer_service import TransformerService
from app.services.bart_service import BARTService
from app.config import settings

st.set_page_config(
    page_title="DialogSum AI - Multi-Task Inference",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-top: 0.5rem;
    }
    .topic-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 2px 5px rgba(59, 130, 246, 0.3);
    }
    .summary-text {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=" Đang tải mô hình GRU từ Hugging Face Hub (Chỉ tải 1 lần duy nhất)...")
def get_gru_service():
    service = GRUService()
    service.load_model()
    return service


@st.cache_resource(show_spinner=" Đang tải mô hình Transformer từ Hugging Face Hub (Chỉ tải 1 lần duy nhất)...")
def get_transformer_service():
    service = TransformerService()
    service.load_model()
    return service


@st.cache_resource(show_spinner=" Đang tải mô hình BART từ Hugging Face Hub (Chỉ tải 1 lần duy nhất)...")
def get_bart_service():
    service = BARTService()
    service.load_model()
    return service


with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.title("Cấu hình Mô hình")

    selected_model = st.selectbox(
        "Chọn mô hình AI:",
        [
            "Seq2Seq-GRU",
            "Seq2Seq-Transformer",
            "BART",
            "Llama-3.2-3B (Coming soon)"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("Tham số sinh văn bản")
    max_summary_len = st.slider("Độ dài tối đa Summary:", min_value=20, max_value=128, value=64, step=4)
    max_topic_len = st.slider("Độ dài tối đa Topic:", min_value=5, max_value=40, value=25, step=1)

    st.markdown("---")
    st.subheader("Thông tin môi trường")
    if "Transformer" in selected_model:
        active_repo = settings.transformer.repo_id
    elif "BART" in selected_model:
        active_repo = settings.bart.repo_id
    else:
        active_repo = settings.gru.repo_id

    st.write(f"• **Repo ID:** `{active_repo}`")
    st.write(f"• **Thiết bị:** `{settings.DEVICE.upper()}`")

    if st.button("Nạp lại Model"):
        st.cache_resource.clear()
        st.rerun()



st.markdown('<div class="main-title">DialogSum AI: Multi-Task Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Trích xuất đồng thời <b>Chủ đề (Topic)</b> và <b>Tóm tắt hội thoại (Summary)</b> bằng mô hình Deep Learning</div>', unsafe_allow_html=True)

SAMPLE_PRESETS = {
    "Mẫu 1: Tiệc sinh nhật (Brian's Birthday)": (
        "#Person1#: Happy Birthday, this is for you, Brian.\n"
        "#Person2#: I'm so happy you remember, please come in and enjoy the party. Everyone's here, I'm sure you have a good time.\n"
        "#Person1#: Brian, may I have a pleasure to have a dance with you?\n"
        "#Person2#: Ok.\n"
        "#Person1#: This is really wonderful party.\n"
        "#Person2#: Yes, you are always popular with everyone. and you look very pretty today.\n"
        "#Person1#: Thanks, that's very kind of you to say. I hope my necklace goes with my dress, and they both make me look good I feel.\n"
        "#Person2#: You look great, you are absolutely glowing.\n"
        "#Person1#: Thanks, this is a fine party. We should have a drink together to celebrate your birthday"
    ),
    "Mẫu 2: Khám bác sĩ (Doctor Check-up)": (
        "#Person1#: Hi, Mr. Smith. I'm Doctor Hawkins. Why are you here today?\n"
        "#Person2#: I found it would be a good idea to get a check-up.\n"
        "#Person1#: Yes, well, you haven't had one for 5 years. You should have one every year.\n"
        "#Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?\n"
        "#Person1#: Well, the best way to avoid serious illnesses is to find out about them early. Take a deep breath, please. Do you smoke?\n"
        "#Person2#: Yes. I've tried hundreds of times, but I can't quit.\n"
        "#Person1#: Smoking is the leading cause of lung cancer. We have classes and medications that might help."
    ),
    "Mẫu 3: Phỏng vấn xin việc (Job Interview)": (
        "#Person1#: Good morning, please have a seat. Can you tell me a little about your experience in software engineering?\n"
        "#Person2#: Good morning. I have worked as a backend Python developer for 3 years, building APIs and database systems.\n"
        "#Person1#: That sounds great. What made you decide to apply for our company?\n"
        "#Person2#: I'm very interested in your AI solutions and want to challenge myself in a fast-paced environment.\n"
        "#Person1#: Excellent. We'll be reviewing all applications and will notify you by Friday."
    ),
    "Tự nhập hội thoại...": ""
}

selected_preset = st.selectbox(
    "Chọn hội thoại mẫu hoặc tự nhập:",
    list(SAMPLE_PRESETS.keys()),
    index=0
)

default_text = SAMPLE_PRESETS[selected_preset]
dialogue_input = st.text_area(
    "Nhập đoạn hội thoại đầu vào (Format: #Person1#: ... \\n #Person2#: ...):",
    value=default_text,
    height=200,
    placeholder="Nhập hoặc dán đoạn hội thoại vào đây..."
)

col_btn, col_clear = st.columns([1, 6])
with col_btn:
    btn_predict = st.button("Chạy Inference", type="primary", use_container_width=True)

if btn_predict:
    if not dialogue_input.strip():
        st.warning("Vui lòng nhập hoặc chọn một đoạn hội thoại trước khi bấm chạy!")
    else:
        with st.spinner("AI đang suy luận ngữ nghĩa hội thoại..."):
            try:
                if "Transformer" in selected_model:
                    service = get_transformer_service()
                elif "BART" in selected_model:
                    service = get_bart_service()
                else:
                    service = get_gru_service()


                result = service.predict(
                    dialogue=dialogue_input,
                    max_summary_len=max_summary_len,
                    max_topic_len=max_topic_len
                )


                st.success("Dự đoán hoàn tất thành công!")

                col_topic, col_summary = st.columns([1, 2])

                with col_topic:
                    st.markdown("### Chủ đề (Topic)")
                    st.markdown(f'<div class="topic-badge">{result["topic"] if result["topic"] else "N/A"}</div>', unsafe_allow_html=True)

                with col_summary:
                    st.markdown("### Bản tóm tắt (Summary)")
                    st.markdown(f'<div class="result-card"><p class="summary-text">{result["summary"]}</p></div>', unsafe_allow_html=True)

                st.markdown("---")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mô hình thực thi", result["model_name"])
                m2.metric("Thời gian phản hồi", f"{result['execution_time_ms']} ms")
                m3.metric("Số từ đầu vào", len(dialogue_input.split()))
                m4.metric("Thiết bị", settings.DEVICE.upper())

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình suy luận: {str(e)}")

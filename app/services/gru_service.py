import time
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from app.config import settings
from app.models.gru import Seq2SeqGRU, Seq2SeqGRUConfig


class GRUService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(settings.DEVICE)

    def load_model(self):
        """Tải tokenizer và trọng số mô hình từ Hugging Face Hub"""
        if self.model is not None:
            return

        print(f"[*] Dang tai GRU model tu: {settings.gru.repo_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.gru.repo_id)

        config = Seq2SeqGRUConfig.from_pretrained(settings.gru.repo_id)
        self.model = Seq2SeqGRU(config, self.tokenizer)

        weights_path = hf_hub_download(repo_id=settings.gru.repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        print(f"[+] GRU Model da nap thanh cong tren thiet bi: {self.device}")

    def predict(self, dialogue: str, max_summary_len: int = 64, max_topic_len: int = 25) -> dict:
        """Thực hiện inference tóm tắt và trích xuất chủ đề"""
        if self.model is None:
            self.load_model()

        start_time = time.perf_counter()

        with torch.no_grad():
            input_encoding = self.tokenizer(
                dialogue,
                padding="max_length",
                truncation=True,
                max_length=settings.gru.max_dialogue_len,
                return_tensors="pt"
            )
            input_ids = input_encoding["input_ids"].to(self.device)

            _, enc_hidden = self.model.encoder(input_ids)

            summary_ids = [self.tokenizer.bos_token_id]
            curr_hidden = enc_hidden
            curr_input = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

            for _ in range(max_summary_len):
                logits, curr_hidden = self.model.summary_decoder(curr_input, curr_hidden)
                next_word = logits[0, -1, :].argmax().item()
                summary_ids.append(next_word)

                if next_word == self.tokenizer.eos_token_id:
                    break
                curr_input = torch.tensor([[next_word]], device=self.device)

            # 2. Giải mã Topic
            topic_ids = [self.tokenizer.bos_token_id]
            curr_hidden = enc_hidden
            curr_input = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)

            for _ in range(max_topic_len):
                logits, curr_hidden = self.model.topic_decoder(curr_input, curr_hidden)
                next_word = logits[0, -1, :].argmax().item()
                topic_ids.append(next_word)

                if next_word == self.tokenizer.eos_token_id:
                    break
                curr_input = torch.tensor([[next_word]], device=self.device)

        pred_summary = self.tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
        pred_topic = self.tokenizer.decode(topic_ids, skip_special_tokens=True).strip()

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": "Seq2Seq-GRU",
            "topic": pred_topic,
            "summary": pred_summary,
            "execution_time_ms": elapsed_ms
        }


gru_service = GRUService()

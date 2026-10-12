import time
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from app.config import settings
from app.models.transformer import (
    Seq2SeqTransformer,
    Seq2SeqTransformerConfig,
    generate_square_subsequent_mask
)


class TransformerService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(settings.DEVICE)

    def load_model(self):
        """Tải tokenizer và trọng số mô hình từ Hugging Face Hub"""
        if self.model is not None:
            return

        print(f"[*] Dang tai Transformer model tu: {settings.transformer.repo_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.transformer.repo_id)

        config = Seq2SeqTransformerConfig.from_pretrained(settings.transformer.repo_id)
        self.model = Seq2SeqTransformer(config, self.tokenizer)

        weights_path = hf_hub_download(
            repo_id=settings.transformer.repo_id,
            filename=settings.transformer.weights_filename
        )
        state_dict = load_file(weights_path)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        print(f"[+] Transformer Model da nap thanh cong tren thiet bi: {self.device}")

    def predict(self, dialogue: str, max_summary_len: int = 64, max_topic_len: int = 25) -> dict:
        """Thực hiện inference tóm tắt và trích xuất chủ đề bằng Seq2Seq Transformer"""
        if self.model is None:
            self.load_model()

        start_time = time.perf_counter()

        with torch.no_grad():
            input_encoding = self.tokenizer(
                dialogue,
                padding="max_length",
                truncation=True,
                max_length=settings.transformer.max_input_len,
                return_tensors="pt"
            )
            input_ids = input_encoding["input_ids"].to(self.device)


            src_seq_len = input_ids.size(1)
            src_positions = torch.arange(0, src_seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

            src_emb = self.model.embedding(input_ids) + self.model.pos_embedding(src_positions)
            src_padding_mask = (input_ids == self.tokenizer.pad_token_id)

            memory = self.model.transformer.encoder(
                src=src_emb,
                src_key_padding_mask=src_padding_mask
            )


            summary_ids = [self.tokenizer.bos_token_id]
            for _ in range(max_summary_len):
                tgt_tensor = torch.tensor([summary_ids], dtype=torch.long, device=self.device)
                tgt_positions = torch.arange(0, tgt_tensor.size(1), dtype=torch.long, device=self.device).unsqueeze(0)

                tgt_emb = self.model.embedding(tgt_tensor) + self.model.pos_embedding(tgt_positions)
                tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1), self.device)

                out = self.model.transformer.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                logits = self.model.fc_summary(out)
                next_word = logits[0, -1, :].argmax().item()
                summary_ids.append(next_word)

                if next_word == self.tokenizer.eos_token_id:
                    break


            topic_ids = [self.tokenizer.bos_token_id]
            for _ in range(max_topic_len):
                tgt_tensor = torch.tensor([topic_ids], dtype=torch.long, device=self.device)
                tgt_positions = torch.arange(0, tgt_tensor.size(1), dtype=torch.long, device=self.device).unsqueeze(0)

                tgt_emb = self.model.embedding(tgt_tensor) + self.model.pos_embedding(tgt_positions)
                tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1), self.device)

                out = self.model.transformer.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                logits = self.model.fc_topic(out)
                next_word = logits[0, -1, :].argmax().item()
                topic_ids.append(next_word)

                if next_word == self.tokenizer.eos_token_id:
                    break

        pred_summary = self.tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
        pred_topic = self.tokenizer.decode(topic_ids, skip_special_tokens=True).strip()

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "model_name": "Seq2Seq-Transformer",
            "topic": pred_topic,
            "summary": pred_summary,
            "execution_time_ms": elapsed_ms
        }


transformer_service = TransformerService()

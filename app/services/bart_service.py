import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import settings

class BARTService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(settings.DEVICE)

    def load_model(self):
        if self.model is None:
            print(f"[*] Dang tai BART tu {settings.bart.repo_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.bart.repo_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.bart.repo_id).to(self.device)
            self.model.eval()

    def predict(self, dialogue: str, max_summary_len: int = 128, max_topic_len: int = 25) -> dict:
        if self.model is None:
            self.load_model()
            
        start_time = time.perf_counter()
        
        inputs = self.tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=settings.bart.max_dialogue_len).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_summary_len,
                num_beams=settings.bart.num_beams,
                no_repeat_ngram_size=3
            )
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if " | Summary: " in text:
            parts = text.split(" | Summary: ")
            topic = re.sub(r'^(Topic[:\s]*)+', '', parts[0], flags=re.IGNORECASE).strip()
            summary = re.sub(r'^(Summary[:\s]*)+', '', parts[1], flags=re.IGNORECASE).strip()
        else:
            topic, summary = "N/A", text


        return {
            "model_name": "BART",
            "topic": topic,
            "summary": summary,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 2)
        }

bart_service = BARTService()

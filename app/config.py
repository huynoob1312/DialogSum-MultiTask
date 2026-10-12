import torch


class GRUConfig:
    repo_id: str = "huynoob1312/dialogsum-multi-task-gru"
    max_dialogue_len: int = 256
    max_summary_len: int = 64
    max_topic_len: int = 25

class TransformerConfig:
    repo_id: str = "huynoob1312/dialogsum-multi-task-transformer"
    weights_filename: str = "model.safetensors"
    max_input_len: int = 256
    max_summary_len: int = 64
    max_topic_len: int = 25


class BARTConfig:
    repo_id: str = "huynoob1312/dialogsum-multi-task-BART"
    max_dialogue_len: int = 256
    max_summary_len: int = 128
    num_beams: int = 4


class LlamaConfig:
    repo_id: str = "huynoob1312/dialogsum-llama-3.2-3b"
    max_dialogue_len: int = 512
    max_new_tokens: int = 128


class Settings:
    PROJECT_NAME: str = "DialogSum Multi-Task AI"
    VERSION: str = "1.0.0"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    gru = GRUConfig()
    transformer = TransformerConfig()
    bart = BARTConfig()
    llama = LlamaConfig()


settings = Settings()

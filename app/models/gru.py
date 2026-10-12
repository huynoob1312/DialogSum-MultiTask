import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class Seq2SeqGRUConfig(PretrainedConfig):
    model_type = "seq2seq_gru"

    def __init__(self, vocab_size=20000, embedding_dim=128, hidden_size=128, dropout=0.1, loss_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.loss_alpha = loss_alpha


class GRU_Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.gru(x)
        return output, hidden


class GRU_Decoder(nn.Module):
    def __init__(self, tg_vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(tg_vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, tg_vocab_size)
        )

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.gru(x, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2SeqGRU(PreTrainedModel):
    config_class = Seq2SeqGRUConfig

    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.encoder = GRU_Encoder(
            src_vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )
        self.summary_decoder = GRU_Decoder(
            tg_vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size
        )
        self.topic_decoder = GRU_Decoder(
            tg_vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size
        )
        self.tokenizer = tokenizer

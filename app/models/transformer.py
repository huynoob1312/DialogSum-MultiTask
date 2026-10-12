from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz,sz), device=device)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    device = src.device

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = None 
    
    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class Seq2SeqTransformerConfig(PretrainedConfig):
    def __init__(self, vocab_size=20000, max_seq_len=128, embedding_dim=256, n_heads=8, n_layers=6, dropout=0.1, loss_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.loss_alpha = loss_alpha

class Seq2SeqTransformer(PreTrainedModel):
    config_class = Seq2SeqTransformerConfig

    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)

        self.transformer = nn.Transformer(
            d_model = config.embedding_dim,
            nhead = config.n_heads,
            num_encoder_layers = config.n_layers,
            num_decoder_layers = config.n_layers,
            dropout = config.dropout,
            batch_first=True
        )
        self.tokenizer = tokenizer
        self.fc_summary = nn.Linear(config.embedding_dim, config.vocab_size)
        self.fc_topic = nn.Linear(config.embedding_dim, config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index= tokenizer.pad_token_id)
        self.loss_alpha = config.loss_alpha
    
    def forward(self, input_ids, summary_labels, topic_labels):
        tg_summary_input = summary_labels[:, :-1] 
        tg_topic_input = topic_labels[:, :-1]
        
        tg_summary_output = summary_labels[:, 1:] 
        tg_topic_output = topic_labels[:, 1:]

        # mask
        _, summary_tgt_mask, src_padding_mask, summary_padding_mask = create_mask(input_ids, tg_summary_input)
        _, topic_tgt_mask, _, topic_padding_mask = create_mask(input_ids, tg_topic_input)

        # embedding
        src_seq_len = input_ids.size(1)
        src_positions = torch.arange(0, src_seq_len, dtype=torch.long, device=input_ids.device)

        summary_seq_len = tg_summary_input.size(1)
        summary_positions = torch.arange(0, summary_seq_len, dtype=torch.long, device=input_ids.device)

        topic_seq_len = tg_topic_input.size(1)
        topic_positions = torch.arange(0, topic_seq_len, dtype=torch.long, device=input_ids.device)

        src_embedding = self.embedding(input_ids) + self.pos_embedding(src_positions)
        summary_tgt_embedding = self.embedding(tg_summary_input) + self.pos_embedding(summary_positions)
        topic_tgt_embedding = self.embedding(tg_topic_input) + self.pos_embedding(topic_positions)

        # encoder
        memory = self.transformer.encoder(
            src=src_embedding,
            src_key_padding_mask=src_padding_mask
        )

        # decoder
        summary_out = self.transformer.decoder(
            tgt=summary_tgt_embedding,
            memory=memory,
            tgt_mask=summary_tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=summary_padding_mask
        )
        logits_summary = self.fc_summary(summary_out)

        topic_out = self.transformer.decoder(
            tgt=topic_tgt_embedding,
            memory=memory,
            tgt_mask=topic_tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=topic_padding_mask
        )
        logits_topic = self.fc_topic(topic_out)

        # loss
        flatten_logits_summary = logits_summary.reshape(-1, logits_summary.size(-1))
        flatten_target_summary = tg_summary_output.reshape(-1)
        
        flatten_logits_topic = logits_topic.reshape(-1, logits_topic.size(-1))
        flatten_target_topic = tg_topic_output.reshape(-1)

        loss_summary = self.loss(flatten_logits_summary, flatten_target_summary)
        loss_topic = self.loss(flatten_logits_topic, flatten_target_topic)
        total_loss =  self.loss_alpha * loss_summary + (1 - self.loss_alpha) * loss_topic

        return {
            "loss": total_loss,
            "loss_summary": loss_summary,
            "loss_topic": loss_topic,
            "logits_summary": logits_summary,
            "logits_topic": logits_topic
        }
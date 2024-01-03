import torch.nn as nn
from torchcrf import CRF


# 基于LSTM的NER模型
class LstmModel(nn.Module):
    def __init__(self, vocab2idx, label2idx, embed_size=None, hidden_size=None, use_crf=True):
        super(LstmModel, self).__init__()
        self.vocab_size = len(vocab2idx)
        self.n_labels = len(label2idx)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_crf = use_crf

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size // 2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.hidden_size, self.n_labels)
        self.loss_func = nn.CrossEntropyLoss()
        self.crf = CRF(num_tags=self.n_labels, batch_first=True)

    def forwards(self, input_ids, label_ids=None, mask=None):
        embed = self.embedding(input_ids)               # (batch, seq_len, embed_size)
        lstm_out, _ = self.lstm(embed)                  # (batch, seq_len ,hidden_size)
        logits = self.linear(self.dropout(lstm_out))    # (batch, seq_len, n_labels)
        # logits = self.linear(lstm_out)
        if label_ids is not None:
            if self.use_crf:
                loss = -1 * self.crf(emissions=logits, tags=label_ids, mask=mask.bool())
                best_path = self.crf.decode(emissions=logits, mask=mask.bool())
                return best_path, loss
            else:
                preds = logits.view(-1, logits.size(-1))
                targets = label_ids.view(-1)
                loss = self.loss_func(preds, targets)
                return logits, loss
        else:
            return logits


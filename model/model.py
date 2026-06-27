import torch
import torch.nn as nn
from crossattention import CrossAttention


class Code_Note(nn.Module):
    def __init__(self, code_encoder, text_encoder, code_size, text_size, hidden_size, output_size, dropout_prob=0):
        super(Code_Note, self).__init__()
        self.code_encoder = code_encoder
        self.text_encoder = text_encoder

        # Cross-attention layer fusion
        self.attention = CrossAttention(embed_size=code_size, text_size=text_size)

        # Classification MLP layers with dropout
        self.fc1 = torch.nn.Linear(code_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout_prob)

        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout_prob)

        self.fc3 = torch.nn.Linear(output_size, 2)

        # Xavier initialization for classification layers
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs_code_id, inputs_code_mask, inputs_text_id, inputs_text_mask):
        # Extract last hidden states from dual modalities
        code_output = self.code_encoder(inputs_code_id, attention_mask=inputs_code_mask).last_hidden_state
        text_output = self.text_encoder(inputs_text_id, attention_mask=inputs_text_mask).last_hidden_state

        # Multi-modal feature fusion via cross-attention
        output = self.attention(code_output, text_output, text_mask=inputs_text_mask)

        # Extract representation from the [CLS] token
        output = output[:, 0, :]

        # MLP classification head
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout1(output)

        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)

        output = self.fc3(output)
        return output
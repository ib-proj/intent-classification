import torch
import torch.nn as nn
from transformers import BertModel


class GruEncoder(nn.Module):
    def __init__(self, hidden_size, num_classes, device="cpu"):
        super(GruEncoder, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.rnn = nn.GRU(input_size=768, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_text, context):
        input_ids = input_text['input_ids'].to(self.device)
        attention_mask = input_text['attention_mask'].to(self.device)
        context_ids = context['input_ids'].to(self.device)
        context_mask = context['attention_mask'].to(self.device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.pooler_output

        context_outputs = self.bert(input_ids=context_ids, attention_mask=context_mask, return_dict=True)
        pooled_context_output = context_outputs.pooler_output

        output, _ = self.rnn(pooled_output.unsqueeze(1))
        context_output, _ = self.rnn(pooled_context_output.unsqueeze(1))
        output = self.fc1(output)
        output = torch.cat((output.squeeze(1), context_output.squeeze(1)), dim=1)
        output = self.fc2(output)
        return output


class GruSpeakerEncoder(nn.Module):
    def __init__(self, hidden_size, num_classes, num_speakers=2, device="cpu"):
        super(GruSpeakerEncoder, self).__init__()
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.speaker_embedding = nn.Embedding(num_speakers, 768)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.rnn = nn.GRU(input_size=768, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_text, speaker_ids, context):
        input_ids = input_text['input_ids'].to(self.device)
        attention_mask = input_text['attention_mask'].to(self.device)
        context_ids = context['input_ids'].to(self.device)
        prev_speakers = speaker_ids[0].to(self.device)
        speakers = speaker_ids[1].to(self.device)
        context_mask = context['attention_mask'].to(self.device)

        prev_speaker_embeddings = self.speaker_embedding(prev_speakers)
        speaker_embeddings = self.speaker_embedding(speakers)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.pooler_output

        pooled_output = pooled_output + speaker_embeddings

        context_outputs = self.bert(input_ids=context_ids, attention_mask=context_mask, return_dict=True)
        pooled_context_output = context_outputs.pooler_output

        pooled_context_output = pooled_context_output + prev_speaker_embeddings

        output, _ = self.rnn(pooled_output.unsqueeze(1))
        context_output, _ = self.rnn(pooled_context_output.unsqueeze(1))
        output = self.fc1(output)
        output = torch.cat((output.squeeze(1), context_output.squeeze(1)), dim=1)
        output = self.fc2(output)
        return output


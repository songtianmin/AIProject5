import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import *

# 加载Bert
tokenizer = BertTokenizer.from_pretrained('pretrained_bert_models/bert-base-uncased/', do_lower_case=True)


class Self_Attention(nn.Module):
    def __init__(self, input_shape):
        super(Self_Attention, self).__init__()
        self.k = input_shape
        self.W_layer = nn.Sequential(
            nn.Linear(self.k, self.k, bias=True),
            nn.Tanh()
        )
        self.U_weight = nn.Parameter(torch.randn(self.k, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        input, mask = inputs

        x = self.W_layer(input)
        score = torch.matmul(x, self.U_weight)

        mask = torch.unsqueeze(mask, dim=-1)
        # mask为当前文本补padding的位置
        padding = torch.ones_like(mask, dtype=torch.float) * (-2 ** 31 + 1)
        score = torch.where(mask != True, padding, score)
        score = self.softmax(score)

        output = torch.matmul(input.transpose(1, 2), score)
        output /= self.k ** 0.5
        output = torch.squeeze(output, dim=-1)

        return output


class Image_Text_Attention(nn.Module):
    def __init__(self, input_shape):
        super(Image_Text_Attention, self).__init__()
        self.k = input_shape

        # 将图片和文字向量进行线性变换，便于后面计算内积取变换后维度为1
        self.img_layer = nn.Sequential(
            nn.Linear(self.k, 1, bias=True),
            nn.Tanh()
        )
        self.seq_layer = nn.Sequential(
            nn.Linear(self.k, 1, bias=True),
            nn.Tanh()
        )
        self.V_weight = nn.Parameter(torch.randn(1, 1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        seq_emb, image_emb = X

        p = self.img_layer(image_emb)
        q = self.seq_layer(seq_emb)

        emb = torch.matmul(p, q.transpose(1, 2))
        emb = emb + q.transpose(2, 1)
        emb = torch.matmul(emb, self.V_weight)

        score = self.softmax(emb)

        output = torch.matmul(score, seq_emb)
        output /= self.k ** 0.5

        return output


class SentenceBert(BertPreTrainedModel):
    def __init__(self, config):
        super(SentenceBert, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.BiGRU = nn.GRU(
            input_size=config.lstm_embedding_size,  # 768
            hidden_size=config.hidden_size // 2,  # 768 / 2
            batch_first=True,
            num_layers=2,
            bidirectional=True
        )

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        gru_output, _ = self.BiGRU(sequence_output)

        return gru_output


class VistaNet(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(VistaNet, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.word_emb = SentenceBert.from_pretrained('pretrained_bert_models/bert-base-uncased')
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.rs_fc = nn.Linear(1000, hidden_size)
        # self.resnet = resnet50(hidden_size)
        self.text_self_attention = Self_Attention(hidden_size)
        self.img_text_attention = Image_Text_Attention(hidden_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(int(hidden_size / 2), num_labels)

    def forward(self, x, attention_mask=None):
        input_texts, input_images = x

        image_emb = self.resnet(input_images)
        image_emb = self.rs_fc(image_emb)

        text_emb = self.word_emb(input_texts, attention_mask=attention_mask)

        emb_attention_mask = attention_mask.cpu().numpy().tolist()
        # # for i in emb_attention_mask:
        # #     i.pop(0)
        emb_attention_mask = torch.tensor(emb_attention_mask, device=self.device)

        seq_emb = self.text_self_attention((text_emb, emb_attention_mask))

        image_emb = torch.unsqueeze(image_emb, dim=1)
        seq_emb = torch.unsqueeze(seq_emb, dim=1)
        doc_emb = self.img_text_attention((seq_emb, image_emb))

        new_mask = torch.ones((input_texts.shape[0], 1))
        new_mask = new_mask.gt(0).cpu().numpy().tolist()
        new_mask = torch.tensor(new_mask, device=self.device)
        doc_attention_emb = self.text_self_attention((doc_emb, new_mask))

        output = self.fc1(doc_attention_emb)

        # output = self.fc1(seq_emb)
        output = self.fc2(output)

        return output

import torch
import torchvision
import torch.nn as nn
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import *
from transformers import DistilBertModel


class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()

        self.config = BertConfig.from_pretrained('pretrained_bert_models/bert-base-uncased/')
        self.text_emb = BertForPreTraining.from_pretrained('pretrained_bert_models/bert-base-uncased/', config=self.config)
        self.text_emb = self.text_emb.bert
        for param in self.text_emb.parameters():
            param.requires_grad = True
        # self.text_emb = BertModel(config)

        backbone = torchvision.models.resnet50(pretrained=True)
        backbone.eval()
        self.vis_shape = backbone.fc.in_features
        layers = list(backbone.children())[:-1]

        self.visual_extractor = nn.Sequential(*layers)

        self.vis_fusion_layer = nn.Sequential(
            nn.Linear(self.vis_shape, 32),
            nn.ReLU(inplace=True)
        )
        self.txt_fusion_layer = nn.Sequential(
            nn.Linear(self.text_emb.config.hidden_size, 32),
            nn.ReLU(inplace=True)
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )

        self.attention = torch.nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=False, dropout=0.4)
        self.clf_layer = nn.Linear(64, 128)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.clf_layer,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 3)
        )

    def forward(self, x, attention_mask=None):
        text, image = x
        text = self.text_emb(text, attention_mask=attention_mask)
        image = self.visual_extractor(image).flatten(1)

        text = self.txt_fusion_layer(text.last_hidden_state[:, 0, :]).unsqueeze(0)
        image = self.vis_fusion_layer(image).unsqueeze(0)

        x = self.attention(torch.cat((image, text), 2)).squeeze()
        x = self.classifier(x)

        return x

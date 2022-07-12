import numpy
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.models.bert.modeling_bert import *
from torchvision import transforms
from PIL import Image
import torch.nn.utils.rnn as run_utils


class VistaDataset(Dataset):
    def __init__(self, sents, data_id_list, tags, data_root_path):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.sents = sents
        self.data_id_list = data_id_list
        self.tags = tags
        self.data_root_path = data_root_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __getitem__(self, index):
        guid = self.data_id_list[index]
        sent = self.sents[index]
        f = os.path.join(self.data_root_path, str(self.data_id_list[index]) + '.jpg')
        image = Image.open(f)
        tag = self.tags[index]
        return guid, sent, self.transform(image), tag

    def __len__(self):
        return len(self.sents)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
        """
        guids = []
        sentences = []
        images = []
        tags = []

        for guid, sentence, image, tag in batch:
            guids.append(guid)
            sentences.append(torch.tensor(sentence))
            images.append(image.cpu().numpy())
            tags.append(torch.tensor(tag))

        text_mask = [torch.ones_like(i) for i in sentences]

        datas = pad_sequence(sentences, batch_first=True, padding_value=0)
        batch_data = torch.as_tensor(datas, dtype=torch.long, device=self.device)
        text_mask = pad_sequence(text_mask, batch_first=True, padding_value=0)
        batch_text_mask = torch.as_tensor(text_mask, dtype=torch.float, device=self.device)
        batch_images = torch.tensor(np.array(images), dtype=torch.float, device=self.device)
        batch_labels = torch.tensor(tags, dtype=torch.long, device=self.device)

        return [guids, batch_data, batch_text_mask, batch_images, batch_labels]

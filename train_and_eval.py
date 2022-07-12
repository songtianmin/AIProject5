import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer
from torch import nn
from tqdm import tqdm

from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

# 加载Bert
tokenizer = BertTokenizer.from_pretrained('pretrained_bert_models/bert-base-uncased/', do_lower_case=True)


def train_with_model(model, train_dataloader, eval_dataloader, epoch_num, learning_rate):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_f1 = 0
    best_model = None
    for epoch in range(1, epoch_num + 1):
        all_train_loss = 0
        # 训练model
        model.train()

        y_true = []
        y_pred = []
        for Xy in tqdm(train_dataloader):
            guid, x_sentences, x_masks, x_images, y = Xy
            output = model((x_sentences, x_images), attention_mask=x_masks)
            loss = loss_function(output, y)

            all_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            y_true.extend(y.cpu())
            y_pred.extend(torch.max(output, 1)[1].cpu())

        train_loss = float(all_train_loss) / len(train_dataloader)
        print("Epoch: {}, train loss: {}, train f1: {}".format(epoch, train_loss, f1_score(np.array(y_true), np.array(y_pred), average='macro')))

        # 在验证集上验证
        # all_dev_loss = 0
        # predicts = []
        # true_y = []
        #
        # model.eval()
        # with torch.no_grad():
        #     for Xy in tqdm(eval_dataloader):
        #         guid, x_sentences, x_masks, x_images, y = Xy
        #         output = model((x_sentences, x_images), attention_mask=x_masks)
        #
        #         loss = loss_function(output, y)
        #         all_dev_loss += loss.item()
        #
        #         predict = torch.max(output, dim=1)[1]
        #
        #         y = y.to("cpu").numpy().tolist()
        #
        #         predicts.extend(predict.to("cpu").numpy().tolist())
        #         true_y.extend(y)
        #
        # dev_loss = float(all_dev_loss) / len(eval_dataloader)
        # print("Epoch: {}, dev loss: {}".format(epoch, dev_loss))
        # print(classification_report(true_y, predicts))
        #
        # dev_f1 = f1_score(np.array(true_y), np.array(predicts), average='macro')
        dev_f1 = eval_with_ablation(model, eval_dataloader, loss_function, epoch, 2)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_model = model

    print("Save model in best-model.pth")
    torch.save(best_model.state_dict(), 'train_model/best-model.pth')


def eval_with_ablation(model, eval_dataloader, loss_function, epoch, eval_type):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # 正常验证
    all_dev_loss = 0
    predicts = []
    true_y = []

    model.eval()
    with torch.no_grad():
        for Xy in tqdm(eval_dataloader):
            guid, x_sentences, x_masks, x_images, y = Xy
            # 正常验证
            if eval_type == 0:
                pass
            # 消融实验，只输入文本
            elif eval_type == 1:
                x_images = torch.ones_like(x_images, device=device)
            # 消融实验，只输入图片
            elif eval_type == 2:
                x_sentences = torch.ones_like(x_sentences, device=device)

            output = model((x_sentences, x_images), attention_mask=x_masks)

            loss = loss_function(output, y)
            all_dev_loss += loss.item()

            predict = torch.max(output, dim=1)[1]

            y = y.to("cpu").numpy().tolist()

            predicts.extend(predict.to("cpu").numpy().tolist())
            true_y.extend(y)

    dev_loss = float(all_dev_loss) / len(eval_dataloader)
    print("Epoch: {}, dev loss: {}".format(epoch, dev_loss))
    print(classification_report(true_y, predicts))

    dev_f1 = f1_score(np.array(true_y), np.array(predicts), average='macro')
    return dev_f1


def predict(model, test_dataloader):
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    y_pred = []
    guid = []
    with torch.no_grad():
        model.eval()
        for Xy in tqdm(test_dataloader):
            guids, x_sentences, x_masks, x_images, _ = Xy
            output = model((x_sentences, x_images), attention_mask=x_masks)
            y_pred.extend(torch.max(output, dim=1)[1].cpu().numpy().tolist())
            guid.extend(guids)

    with open('./train_model/result.txt', 'w', encoding="utf-8") as f:
        f.write("guid,tag\n")
        label_dic = {0: "positive", 1: "neutral", 2: "negative"}
        for i in range(len(guid)):
            f.write(guid[i] + ',' + label_dic[y_pred[i]] + "\n")

        f.close()


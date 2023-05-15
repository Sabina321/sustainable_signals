import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import torch
import transformers
import random
import os
from torch import nn
from sklearn.model_selection import train_test_split

from datasets import load_dataset, Dataset, load_metric, DatasetDict

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import DistilBertPreTrainedModel, DistilBertConfig
from transformers import DistilBertForSequenceClassification, AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class noCate(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = AutoModel.from_config(config)

        self.pre_review = AutoModel.from_config(config)
        #self.cat_shape = config.dim
        #self.cat_shape = config.hidden_size
        self.cat_shape = 768
        # self.classifier = nn.Linear(config.dim + 20, config.num_labels)
        self.classifier = nn.Linear(768 + self.cat_shape, config.num_labels)
        self.dropout = nn.Dropout(0.1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            rating=None,
            price=None,
            bea=None,
            ba=None,
            hou=None,
            kit=None,
            tfidf=None,
            input_ids0=None,
            attention_mask0=None,
            input_ids1=None,
            attention_mask1=None,
    ):
        # print(input_ids.shape)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(tfidf)
        hidden_state = distilbert_output[0]  # (bs, seq_len, ori_dim)
        pooled_output = hidden_state[:, 0]  # (bs, ori_dim)

        pre_output = self.pre_review(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        )

        pre_hidden_state = pre_output[0]  # (bs, seq_len, ori_dim)
        pre_pooled_output = pre_hidden_state[:, 0]  # (bs, ori_dim)

        pooled_output = torch.cat((pooled_output, pre_pooled_output), 1)  # (bs, new_dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        pooled_output = nn.ReLU()(pooled_output)  # (bs, new_dim)

        logits = self.classifier(pooled_output)  # (bs, new_dim)
        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class Cate(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = AutoModel.from_config(config)
        self.pre_review = AutoModel.from_config(config)
        #self.cat_shape = config.hidden_size
        self.cat_shape = 768
        self.dropout = nn.Dropout(0.1)

        self.house_in = nn.Linear(768 + self.cat_shape, args.house_dim)
        self.house_out = nn.Linear(args.house_dim, 1)

        self.beauty_in = nn.Linear(768 + self.cat_shape, args.beauty_dim)
        self.beauty_out = nn.Linear(args.beauty_dim, 1)

        self.baby_in = nn.Linear(768 + self.cat_shape, args.baby_dim)
        self.baby_out = nn.Linear(args.baby_dim, 1)

        self.kitchen_in = nn.Linear(768 + self.cat_shape, args.kitchen_dim)
        self.kitchen_out = nn.Linear(args.kitchen_dim, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            rating=None,
            price=None,
            bea=None,
            ba=None,
            hou=None,
            kit=None,
            tfidf=None,
            input_ids0=None,
            attention_mask0=None,
            input_ids1=None,
            attention_mask1=None,
    ):
        # print(input_ids.shape)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(tfidf)

        hidden_state = distilbert_output[0]  # (bs, seq_len, ori_dim)
        pooled_output = hidden_state[:, 0]  # (bs, ori_dim)

        pre_output = self.pre_review(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        )

        pre_hidden_state = pre_output[0]  # (bs, seq_len, ori_dim)
        pre_pooled_output = pre_hidden_state[:, 0]  # (bs, ori_dim)

        pooled_output = torch.cat((pooled_output, pre_pooled_output), 1)  # (bs, new_dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        ######################################################
        pooled_output = nn.ReLU()(pooled_output)  # (bs, new_dim)

        hou, ba, kit, bea = hou.bool(), ba.bool(), kit.bool(), bea.bool()

        house = self.house_out(self.house_in(pooled_output[hou]))
        beauty = self.beauty_out(self.beauty_in(pooled_output[bea]))
        baby = self.baby_out(self.baby_in(pooled_output[ba]))
        kitchen = self.kitchen_out(self.kitchen_in(pooled_output[kit]))
        # print(house.shape, hou.int().float().unsqueeze(1).shape)


        logits = torch.zeros(pooled_output.shape[0], 1).half().to('cuda')
        #print(logits.dtype, house.dtype)
        hou_i = torch.where(hou)
        ba_i = torch.where(ba)
        kit_i = torch.where(kit)
        bea_i = torch.where(bea)
        #print(hou_i, logits.shape)
        logits[hou_i] = house
        logits[ba_i] = baby
        logits[kit_i] = kitchen
        logits[bea_i] = beauty

        # logits = house * hou.int().float().unsqueeze(1) + baby * ba.int().float().unsqueeze(1) \
        #          + kitchen * kit.int().float().unsqueeze(1) + beauty * bea.int().float().unsqueeze(1)

        # logits = self.classifier(pooled_output)  # (bs, new_dim)
        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

class Env_add(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = AutoModel.from_config(config)
        self.roberta = AutoModel.from_pretrained(args.pretrain_ro_path)
        self.pre_review = AutoModel.from_config(config)

        self.dropout = nn.Dropout(0.1)

        #self.cat_shape = 2 * config.hidden_size
        self.cat_shape = 2 * 768
        self.house_in = nn.Linear(768 + self.cat_shape, args.house_dim)
        self.house_out = nn.Linear(args.house_dim, 1)

        self.beauty_in = nn.Linear(768 + self.cat_shape, args.beauty_dim)
        self.beauty_out = nn.Linear(args.beauty_dim, 1)

        self.baby_in = nn.Linear(768 + self.cat_shape, args.baby_dim)
        self.baby_out = nn.Linear(args.baby_dim, 1)

        self.kitchen_in = nn.Linear(768 + self.cat_shape, args.kitchen_dim)
        self.kitchen_out = nn.Linear(args.kitchen_dim, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            rating=None,
            price=None,
            bea=None,
            ba=None,
            hou=None,
            kit=None,
            tfidf=None,
            input_ids0=None,
            attention_mask0=None,
            input_ids1=None,
            attention_mask1=None,
    ):
        # print(input_ids.shape)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(tfidf)
        hidden_state = distilbert_output[0]  # (bs, seq_len, ori_dim)
        pooled_output = hidden_state[:, 0]  # (bs, ori_dim)

        ro_output = self.roberta(
            input_ids=input_ids0,
            attention_mask=attention_mask0,
        )

        ro_hidden_state = ro_output[0]  # (bs, seq_len, ori_dim)
        ro_pooled_output = ro_hidden_state[:, 0]  # (bs, ori_dim)

        pre_output = self.pre_review(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        )

        pre_hidden_state = pre_output[0]  # (bs, seq_len, ori_dim)
        pre_pooled_output = pre_hidden_state[:, 0]  # (bs, ori_dim)

        pooled_output = torch.cat((pooled_output, ro_pooled_output, pre_pooled_output), 1)  # (bs, new_dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, new_dim)

        hou, ba, kit, bea = hou.bool(), ba.bool(), kit.bool(), bea.bool()

        house = self.house_out(self.house_in(pooled_output[hou]))
        beauty = self.beauty_out(self.beauty_in(pooled_output[bea]))
        baby = self.baby_out(self.baby_in(pooled_output[ba]))
        kitchen = self.kitchen_out(self.kitchen_in(pooled_output[kit]))
        # print(house.shape, hou.int().float().unsqueeze(1).shape)

        logits = torch.zeros(pooled_output.shape[0], 1).half().to('cuda')
        # print(logits.dtype, house.dtype)
        hou_i = torch.where(hou)
        ba_i = torch.where(ba)
        kit_i = torch.where(kit)
        bea_i = torch.where(bea)
        # print(hou_i, logits.shape)
        logits[hou_i] = house
        logits[ba_i] = baby
        logits[kit_i] = kitchen
        logits[bea_i] = beauty

        # logits = self.classifier(pooled_output)  # (bs, new_dim)
        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class Anno_only(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = AutoModel.from_config(config)

        self.pre_review = AutoModel.from_pretrained(args.pretrain_review)

        self.dropout = nn.Dropout(0.1)

        #self.cat_shape = 2 * config.hidden_size
        self.cat_shape = 768

        self.house_in = nn.Linear(768 + self.cat_shape, args.house_dim)
        self.house_out = nn.Linear(args.house_dim, 1)

        self.beauty_in = nn.Linear(768 + self.cat_shape, args.beauty_dim)
        self.beauty_out = nn.Linear(args.beauty_dim, 1)

        self.baby_in = nn.Linear(768 + self.cat_shape, args.baby_dim)
        self.baby_out = nn.Linear(args.baby_dim, 1)

        self.kitchen_in = nn.Linear(768 + self.cat_shape, args.kitchen_dim)
        self.kitchen_out = nn.Linear(args.kitchen_dim, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            rating=None,
            price=None,
            bea=None,
            ba=None,
            hou=None,
            kit=None,
            tfidf=None,
            input_ids0=None,
            attention_mask0=None,
            input_ids1=None,
            attention_mask1=None,
    ):
        # print(input_ids.shape)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(tfidf)

        hidden_state = distilbert_output[0]  # (bs, seq_len, ori_dim)
        pooled_output = hidden_state[:, 0]  # (bs, ori_dim)

        pre_output = self.pre_review(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        )

        pre_hidden_state = pre_output[0]  # (bs, seq_len, ori_dim)
        pre_pooled_output = pre_hidden_state[:, 0]  # (bs, ori_dim)

        pooled_output = torch.cat((pooled_output, pre_pooled_output), 1)  # (bs, new_dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)


        pooled_output = nn.ReLU()(pooled_output)  # (bs, new_dim)

        hou, ba, kit, bea = hou.bool(), ba.bool(), kit.bool(), bea.bool()

        house = self.house_out(self.house_in(pooled_output[hou]))
        beauty = self.beauty_out(self.beauty_in(pooled_output[bea]))
        baby = self.baby_out(self.baby_in(pooled_output[ba]))
        kitchen = self.kitchen_out(self.kitchen_in(pooled_output[kit]))
        # print(house.shape, hou.int().float().unsqueeze(1).shape)

        logits = torch.zeros(pooled_output.shape[0], 1).half().to('cuda')
        # print(logits.dtype, house.dtype)
        hou_i = torch.where(hou)
        ba_i = torch.where(ba)
        kit_i = torch.where(kit)
        bea_i = torch.where(bea)
        # print(hou_i, logits.shape)
        logits[hou_i] = house
        logits[ba_i] = baby
        logits[kit_i] = kitchen
        logits[bea_i] = beauty

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

class Anno_add(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = AutoModel.from_config(config)
        self.roberta = AutoModel.from_pretrained(args.pretrain_ro_path)
        self.pre_review = AutoModel.from_pretrained(args.pretrain_review)

        self.dropout = nn.Dropout(0.1)

        #self.cat_shape = 2 * config.hidden_size
        self.cat_shape = 2 * 768
        self.house_in = nn.Linear(768 + self.cat_shape, args.house_dim)
        self.house_out = nn.Linear(args.house_dim, 1)

        self.beauty_in = nn.Linear(768 + self.cat_shape, args.beauty_dim)
        self.beauty_out = nn.Linear(args.beauty_dim, 1)

        self.baby_in = nn.Linear(768 + self.cat_shape, args.baby_dim)
        self.baby_out = nn.Linear(args.baby_dim, 1)

        self.kitchen_in = nn.Linear(768 + self.cat_shape, args.kitchen_dim)
        self.kitchen_out = nn.Linear(args.kitchen_dim, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            rating=None,
            price=None,
            bea=None,
            ba=None,
            hou=None,
            kit=None,
            tfidf=None,
            input_ids0=None,
            attention_mask0=None,
            input_ids1=None,
            attention_mask1=None,
    ):
        # print(input_ids.shape)

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(tfidf)

        hidden_state = distilbert_output[0]  # (bs, seq_len, ori_dim)
        pooled_output = hidden_state[:, 0]  # (bs, ori_dim)

        ro_output = self.roberta(
            input_ids=input_ids0,
            attention_mask=attention_mask0,
        )

        ro_hidden_state = ro_output[0]  # (bs, seq_len, ori_dim)
        ro_pooled_output = ro_hidden_state[:, 0]  # (bs, ori_dim)

        pre_output = self.pre_review(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        )

        pre_hidden_state = pre_output[0]  # (bs, seq_len, ori_dim)
        pre_pooled_output = pre_hidden_state[:, 0]  # (bs, ori_dim)

        pooled_output = torch.cat((pooled_output, ro_pooled_output, pre_pooled_output), 1)  # (bs, new_dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)


        pooled_output = nn.ReLU()(pooled_output)  # (bs, new_dim)

        hou, ba, kit, bea = hou.bool(), ba.bool(), kit.bool(), bea.bool()

        house = self.house_out(self.house_in(pooled_output[hou]))
        beauty = self.beauty_out(self.beauty_in(pooled_output[bea]))
        baby = self.baby_out(self.baby_in(pooled_output[ba]))
        kitchen = self.kitchen_out(self.kitchen_in(pooled_output[kit]))
        # print(house.shape, hou.int().float().unsqueeze(1).shape)

        logits = torch.zeros(pooled_output.shape[0], 1).half().to('cuda')
        # print(logits.dtype, house.dtype)
        hou_i = torch.where(hou)
        ba_i = torch.where(ba)
        kit_i = torch.where(kit)
        bea_i = torch.where(bea)
        # print(hou_i, logits.shape)
        logits[hou_i] = house
        logits[ba_i] = baby
        logits[kit_i] = kitchen
        logits[bea_i] = beauty

        outputs = (logits,) + distilbert_output[1:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
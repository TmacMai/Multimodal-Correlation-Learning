import logging
from typing import Optional, Tuple
from modules.transformer import TransformerEncoder

import torch
import torch.utils.checkpoint
from torch import nn
#from torch.nn import CrossEntropyLoss, MSELoss

from torch.nn import L1Loss, MSELoss

from transformers.modeling_bert import BertPreTrainedModel
from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig

import torch.optim as optim
from itertools import chain

from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM, DEVICE

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_new": gelu_new,
    "mish": mish,
}


BertLayerNorm = torch.nn.LayerNorm


class MCL_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, d_l):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.d_l = d_l
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        # visual,
        # acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]

        outputs = sequence_output.transpose(1, 2)
        outputs = self.proj_l(outputs)
        pooled_output = outputs[:, :, -1]

        return pooled_output

import numpy as np
class MCL(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, args = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.d_l = args.d_l  ###############40 for mosi, 50 for mosei

        self.bert = MCL_BertModel(config, multimodal_config, self.d_l)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.d_l*3, config.num_labels)

        self.classifier2 = nn.Linear(self.d_l*2, 1)
        self.classifier3 = nn.Linear(self.d_l*3, 1)  ###correlation predictor

        self.attn_dropout = args.attn_dropout  
        self.num_heads = args.num_heads
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.proj_a = nn.Conv1d(ACOUSTIC_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_v = nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.transa = self.get_network(self_type='a', layers=args.layers) ###3
        self.transv = self.get_network(self_type='v', layers=args.layers)  ###3

        self.llr =  args.correlation_lr  #correlation learning rate
        self.lr = args.learning_rate     #main task learning rate
        self.optimizer_all = getattr(optim, 'Adam')(chain(self.bert.parameters(), self.transa.parameters(), self.transv.parameters(), self.classifier.parameters(), self.proj_a.parameters(), self.proj_v.parameters()), lr=self.lr)
        self.optimizer_lav = getattr(optim, 'Adam')(chain(self.bert.parameters(), self.transa.parameters(), self.transv.parameters(), self.proj_a.parameters(), self.proj_v.parameters()), lr=self.llr)
        self.optimizer_c3 = getattr(optim, 'Adam')(chain(self.classifier2.parameters(), self.classifier3.parameters()), lr=self.llr*0.1)


        self.ratio = args.ratio   ######megative sample factor, default 4
        self.alpha = args.alpha   #####hyperparameter for score assigniment function
        self.gamma = args.gamma   ###hyperparameter for score assigniment function

        self.loss = L1Loss()
        self.init_weights()

    def get_network(self, self_type='l', layers=5):
        if self_type in ['a', 'v']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads= self.num_heads,
                                  layers=layers,
                                  attn_dropout= attn_dropout,
                                  relu_dropout=self.relu_dropout,   
                                  res_dropout= self.res_dropout,    
                                  embed_dropout=self.embed_dropout,  
                                  attn_mask= False)#self.attn_mask)

    def generate_supervised_trimodal_loss(self, x_l, x_a, x_v, label):
        
        batch = [i for i in range(x_l.shape[0])]
        index1 = np.random.choice(batch, x_l.shape[0] * (self.ratio+1), replace=True)
        index2 = np.random.choice(batch, x_l.shape[0] * (self.ratio+1), replace=True)
      #  index3 = list(np.random.choice(batch, x_l.shape[0] * 2, replace=True))
        index3 = np.where(index1 != index2)[0]
        index1 = index1[index3]
        index2 = index2[index3]

        index3 = np.random.choice(batch, len(index1), replace=True)

        index4 = np.where(index1 != index3)[0]
        index5 = np.where(index2 != index3)[0]
        index4 = set(index4).intersection(set(index5))
        index4 = list(index4)
        index1 = index1[index4]
        index2 = index2[index4]
        index3 = index3[index4]
        
        total_len = x_l.shape[0] * self.ratio
        if len(index1) > total_len:
            index1 = index1[:total_len]
            index2 = index2[:total_len]
            index3 = index3[:total_len]
            

        label_p = torch.ones((x_l.shape[0],1),).float().cuda()

        negative_pair1 = torch.cat([x_l[index1], x_v[index2], x_a[index3]], dim = -1)
        positive_pair1 = torch.cat([x_l, x_v, x_a], dim = -1)
        pair1 = torch.cat([negative_pair1, positive_pair1], dim=0)

        y1 = self.classifier3(pair1)

        
        label_s = (label[index1] - label[index2])**2 + (label[index1] - label[index3])**2 + (label[index3] - label[index2])**2
        label_s = torch.sqrt(label_s)
        label_s = (label_s+self.alpha)/(self.gamma+self.alpha)


        label = torch.cat([label_s.squeeze(2), label_p-1], dim=0)
        loss1 = self.loss(y1, label)
        return loss1 



    def generate_trimodal_loss(self, x_l, x_a, x_v, label = True):
        
        batch = [i for i in range(x_l.shape[0])]
        index1 = np.random.choice(batch, x_l.shape[0] * (self.ratio+1), replace=True)
        index2 = np.random.choice(batch, x_l.shape[0] * (self.ratio+1), replace=True)
      #  index3 = list(np.random.choice(batch, x_l.shape[0] * 2, replace=True))
        index3 = np.where(index1 != index2)[0]
        index1 = index1[index3]
        index2 = index2[index3]

        index3 = np.random.choice(batch, len(index1), replace=True)

        index4 = np.where(index1 != index3)[0]
        index5 = np.where(index2 != index3)[0]
        index4 = set(index4).intersection(set(index5))
        index4 = list(index4)
        index1 = index1[index4]
        index2 = index2[index4]
        index3 = index3[index4]
        
        total_len = x_l.shape[0] * self.ratio
        if len(index1) > total_len:
            index1 = index1[:total_len]
            index2 = index2[:total_len]
            index3 = index3[:total_len]


        label_n  = torch.zeros((len(index1),1),).float().cuda()
        label_p = torch.ones((x_l.shape[0],1),).float().cuda()

        negative_pair1 = torch.cat([x_l[index1], x_v[index2], x_a[index3]], dim = -1)
        positive_pair1 = torch.cat([x_l, x_v, x_a], dim = -1)
        pair1 = torch.cat([negative_pair1, positive_pair1], dim=0)

        y1 = self.classifier3(pair1)

        label = torch.cat([label_n+1, label_p-1], dim=0)
        loss1 = self.loss(y1, label)
      
        return loss1 


    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
  
        output_l = outputs
     
        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        output_a = outputa[-1]  
        output_v = outputv[-1]


        ####################correlation learning################################
        #loss_1 = self.generate_supervised_loss(output_l, output_a, output_v, label_ids)  ###unsupervised correlation learning
        loss_1 = self.generate_supervised_trimodal_loss(output_l, output_a, output_v, label_ids)  ####supervised correlation learning
        
        ###################fusion##############################################
     
        fusion = torch.cat([output_l, output_a, output_v], dim = -1)
        outputf = self.classifier(fusion)
        
        #####update according to correlation loss
        self.optimizer_c3.zero_grad()
        self.optimizer_lav.zero_grad()
        loss_1.backward(retain_graph = True)
        self.optimizer_lav.step()
        self.optimizer_c3.step()


        ############update according to main task loss
        self.optimizer_all.zero_grad()
        loss_all = self.loss(outputf.view(-1), label_ids.view(-1))
        loss_all.backward()
        self.optimizer_all.step()

        return outputf



    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        output_l = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)


        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        output_a = outputa[-1]  # 48 50
        output_v = outputv[-1]

        fusion = torch.cat([output_l, output_a, output_v], dim = -1)
        outputs = self.classifier(fusion)

        return outputs
  

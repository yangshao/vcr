"""
Let's get the relationships yo
"""

from typing import Dict

import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention, DotProductMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, replace_masked_values, get_final_encoder_states
from allennlp.nn import InitializerApplicator
import config
from compact_bilinear_pooling import CompactBilinearPooling
from models.multiatt.sample_utils import *
from torch.distributions.categorical import Categorical
import math
import numpy as np
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class PositionalEncoder(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

@Model.register("MultiHopAttentionQA")
class AttentionQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQA, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.reasoning_encoder = TimeDistributed(reasoning_encoder)

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        dim = sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        (span_encoder.get_output_dim(), self.pool_answer),
                                        (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout and self.training:
            span_rep = self.rnn_input_dropout(span_rep)

        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                ind: torch.LongTensor = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]

        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # Now get the question representations
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        ####################################
        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]
        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                            a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:,None,None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))


        reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                           (attended_o, self.reasoning_use_obj),
                                                           (attended_q, self.reasoning_use_question)]
                                      if to_pool], -1)

        if self.rnn_input_dropout is not None and self.training:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)
        reasoning_output = self.reasoning_encoder(reasoning_inp, answer_mask)


        ###########################################
        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (a_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)

        pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]
        logits = self.final_mlp(pooled_rep).squeeze(2)

        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}



@Model.register("CRF_QA_FINAL")
class CRFQA_FINAL(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 seq_encoder: Seq2SeqEncoder,
                 factor_lstm: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 att_flag: bool = False,
                 multi_flag: bool = False,
                 adaptive_flag: bool = False,
                 wo_qa: bool = False,
                 wo_qr: bool = False,
                 wo_ar: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(CRFQA_FINAL, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.att_flag = att_flag
        self.multi_flag = multi_flag
        self.adaptive_flag = adaptive_flag
        self.wo_qa = wo_qa
        self.wo_qr = wo_qr
        self.wo_ar = wo_ar
        print('att flag: ', self.att_flag)
        print('wo_qa: ', self.wo_qa)
        print('wo_qr: ', self.wo_qr)
        print('wo_ar: ', self.wo_ar)
        print('multi flag: ', self.multi_flag)
        print('adaptive flag: ', self.adaptive_flag)
        self.seq_encoder = seq_encoder
        if self.adaptive_flag:
            self.factor_lstm = factor_lstm
            self.factor_layer = torch.nn.Linear(512, 3)
        self.co_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.text_image_attention = BilinearMatrixAttention(
            matrix_1_dim=768,
            matrix_2_dim=1024,
        )
        self.summary_image_attention = BilinearMatrixAttention(
            matrix_1_dim=1024,
            matrix_2_dim=1024,
        )

        self.bilinear_layer = torch.nn.Bilinear(1024, 512,1, bias=False)

        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(1024, 512)

        self.input_mlp = torch.nn.Linear(1280, 512)
        # self.input_mlp = torch.nn.Linear(1792, 512)
        self.co_input_mlp = torch.nn.Linear(768, 512)
        # self.input_mlp = torch.nn.Linear(768, 512)

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(4*512, 1),
            # torch.nn.Linear(2*512, 1),
        )

        self.vqa_weight_layer = torch.nn.Linear(1536, 1)
        self.co_weight_layer = torch.nn.Linear(4*512, 1)
        # self.co_weight_layer = torch.nn.Linear(2*512, 1)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.margin_loss = torch.nn.MultiMarginLoss(margin=0.5, p=1)
        initializer(self)



    def get_vqa_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        img_feats = obj_reps['img_feats']
        # img_feats = obj_reps['new_2d_objs'][:,0,:,:]
        q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)
        text_summary = torch.cat([self_attended_q, self_attended_a], dim=-1)

        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.summary_image_attention(
            text_summary,
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        v_att = F.softmax(text_image_similarity, dim=2)
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
        att_img_feats = self.image_mlp(att_img_feats)

        concat_summary = torch.cat([text_summary, att_img_feats], dim=-1)
        final_weight = self.vqa_weight_layer(concat_summary).squeeze()

        logits = self.bilinear_layer(text_summary.contiguous(), att_img_feats.contiguous()).squeeze()

        # logits = self.final_mlp(self_attended_q+self_attended_a).squeeze()
        return logits, v_att, final_weight
    def get_co_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        # q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        # a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)
        #not using visual information in answer explanation module
        if config.double_flag:
            qa_question['bert'] = qa_question['bert'].double()
            qa_answer['bert'] = qa_answer['bert'].double()
        q_rep = F.relu(self.co_input_mlp(qa_question['bert']))
        a_rep = F.relu(self.co_input_mlp(qa_answer['bert']))

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)

        final_rep = torch.cat([self_attended_q, self_attended_a, torch.abs(self_attended_q-self_attended_a), self_attended_q*self_attended_a], dim=-1)
        # final_rep = torch.cat([self_attended_q, self_attended_a], dim=-1)
        logits = self.final_mlp(final_rep).squeeze()
        final_weight = self.co_weight_layer(final_rep).squeeze()
        return logits, final_weight

    def get_factor_weights(self, answers, answers_tag, answers_mask):
        answer_input = answers['bert']
        batch_size, ct, sent_len, emb_size = answer_input.size()
        answer_emb = self.factor_lstm(answer_input.view(batch_size*ct, sent_len, emb_size), answers_mask.view(batch_size*ct, sent_len))
        answer_emb = get_final_encoder_states(answer_emb, answers_mask.view(batch_size*ct, sent_len), bidirectional=True)
        answer_emb = answer_emb.view(batch_size, ct, 512)
        factor_weight = F.relu(self.factor_layer(answer_emb))
        return F.softmax(factor_weight, dim=-1)


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if config.double_flag:
            images = images.double()
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))


        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)


        if(not self.wo_qa):
            qa_logits, qa_att, qa_weight_logit = self.get_vqa_logits(qa_question, qa_question_tags, qa_question_mask, qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        if not self.wo_qr:
            qr_logits, qr_att, qr_weight_logit = self.get_vqa_logits(qr_question, qr_question_tags, qr_question_mask, qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        if not self.wo_ar:
            ar_logits, ar_weight_logit  = self.get_co_logits(ar_answers, ar_answer_tags, ar_answer_mask, ar_rationales, ar_rationale_tags, ar_rationale_mask, obj_reps)
        batch_size =images.shape[0]
        a_ct = qa_logits.size(1)
        r_ct = qr_logits.size(1)

        if self.adaptive_flag:
            # factor_weights = self.get_factor_weights(qa_answers, qa_answer_tags, qa_answer_mask)
            qa_weight_logit = qa_weight_logit.unsqueeze(2).expand(batch_size, a_ct, r_ct).contiguous().view(batch_size, a_ct*r_ct, 1)
            qr_weight_logit = qr_weight_logit.unsqueeze(1).expand(batch_size, a_ct, r_ct).contiguous().view(batch_size, a_ct*r_ct, 1)
            ar_weight_logit = ar_weight_logit.unsqueeze(2)
            factor_weights_temp = torch.cat([qa_weight_logit, qr_weight_logit, ar_weight_logit], dim=2)
            factor_weights = F.softmax(factor_weights_temp, dim=-1)
        if(not self.wo_qa):
            new_qa_logits = qa_logits.unsqueeze(2).expand(batch_size, a_ct, r_ct)
        if not self.wo_qr:
            new_qr_logits = qr_logits.unsqueeze(1).expand(batch_size, a_ct, r_ct)
        if not self.wo_ar:
            new_ar_logits = ar_logits.view(batch_size, a_ct, r_ct)
        #rearange into 16 situations
        #situation 1:
        if self.wo_qa:
            logits = (new_qr_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_qr:
            logits = (new_qa_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_ar:
            logits = (new_qa_logits + new_qr_logits).view(batch_size, -1)
        else:
            if self.adaptive_flag:
                # qa_weight =  factor_weights[:,:,0].unsqueeze(2).expand_as(new_qa_logits)
                # qr_weight =  factor_weights[:,:,1].unsqueeze(2).expand_as(new_qr_logits)
                # ar_weight = factor_weights[:,:,2].unsqueeze(2).expand_as(new_ar_logits)
                qa_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                qr_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                ar_weight = factor_weights[:,:,2].view(batch_size, a_ct, r_ct)
                logits = (qa_weight*new_qa_logits + qr_weight*new_qr_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_qr_logits + new_ar_logits).view(batch_size, -1)

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       'qa_logits': new_qa_logits,
                       'qr_logits': new_qr_logits,
                       'ar_logits': new_ar_logits,
                       'qa_att': qa_att,
                       'qr_att': qr_att,
                       'test': qa_logits
                       # 'test': obj_reps['ori_images']
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        #augmented joint modeling
        if answer_label is not None and rationale_label is not None:
            final_label = answer_label*r_ct+rationale_label
            loss = self._loss(logits, final_label.long().view(-1))
            # loss = self.margin_loss(logits, final_label.long().view(-1))
            self._accuracy(logits, final_label)
            output_dict["loss"] = loss[None]
            if self.multi_flag:
                output_dict['answer_loss'] = self._loss(qa_logits, answer_label.long().view(-1))
                output_dict['rationale_loss'] = self._loss(qr_logits, rationale_label.long().view(-1))
            if self.adaptive_flag:
                output_dict['adaptive_weight'] = factor_weights


        return output_dict

    def _image_text_co_attention_feats(self, span, img_feats):
        text = span['bert']
        if config.double_flag:
            text = text.double()
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.text_image_attention(
            text.view(text.shape[0],text.shape[1]*text.shape[2], text.shape[3]),
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att = F.softmax(text_image_similarity,dim=-1)
        att_text = torch.bmm(att, flat_img_feats)
        # print(text_image_similarity.size())
        att_text = att_text.view(text.shape[0], text.shape[1], text.shape[2], -1)
        att_text = F.relu(self.image_mlp(att_text))
        return att_text, text_image_similarity


    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        # image_feats = object_reps['new_2d_objs'][:, 0, :,:]
        image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, object_reps['img_feats'])
        # image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, image_feats)
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps['obj_reps'], image_attention_feats)
        # retrieved_feats = self._collect_obj_reps(span_tags, object_reps['new_1d_obj_reps'], image_attention_feats)

        if config.double_flag:
            span['bert'] = span['bert'].double()
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        return F.relu(self.input_mlp(span_rep)), retrieved_feats, text_image_similarity
    def _collect_obj_reps(self, span_tags, object_reps, image_attention_feats):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        #image not as a box
        span_mask = (span_tags>=0).float()
        #image as a box
        # span_mask = (span_tags>0).float()
        image_mask = 1.0-span_mask

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        span_visual_feats =  object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

        if config.double_flag:
            span_mask = span_mask.double()
            image_mask = image_mask.double()
        final_fests = span_mask.unsqueeze(-1).expand_as(span_visual_feats)*span_visual_feats + image_mask.unsqueeze(-1).expand_as(image_attention_feats)*image_attention_feats
        return final_fests
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("CRF_QA_Context")
class CRFQA_Context(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 ar_seq_encoder: Seq2SeqEncoder,
                 vqa_seq_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 gd_flag: bool = False,
                 att_flag: bool = False,
                 multi_flag: bool = False,
                 adaptive_flag: bool = False,
                 wo_qa: bool = False,
                 wo_qr: bool = False,
                 wo_ar: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(CRFQA_Context, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.gd_flag = gd_flag
        self.att_flag = att_flag
        self.multi_flag = multi_flag
        self.adaptive_flag = adaptive_flag
        self.wo_qa = wo_qa
        self.wo_qr = wo_qr
        self.wo_ar = wo_ar
        print('att flag: ', self.att_flag)
        print('wo_qa: ', self.wo_qa)
        print('wo_qr: ', self.wo_qr)
        print('wo_ar: ', self.wo_ar)
        print('multi flag: ', self.multi_flag)
        print('adaptive flag: ', self.adaptive_flag)
        self.ar_seq_encoder = ar_seq_encoder
        self.vqa_seq_encoder = vqa_seq_encoder
        if self.adaptive_flag:
            self.factor_layer = torch.nn.Linear(512*4, 3)
        self.co_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.text_image_attention = BilinearMatrixAttention(
            matrix_1_dim=768,
            matrix_2_dim=1024,
        )
        self.text_image_attention1 = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=1024,
        )
        self.text_text_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.co_mcb = CompactBilinearPooling(512, 512, 2048).cuda()
        self.co_mcb_linear = torch.nn.Linear(2048, 512)
        self.image_text_mcb = CompactBilinearPooling(512, 1024, 512).cuda()

        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(1024, 512)

        self.input_mlp = torch.nn.Linear(1280, 512)
        self.co_input_mlp = torch.nn.Linear(768, 512)

        self.ar_score_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
        )
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            # torch.nn.Linear(4*512, 1),
            torch.nn.Linear(2*512, 1),
        )
        self.vqa_score_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.kl_loss = torch.nn.KLDivLoss()
        initializer(self)



    def get_vqa_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask,
                       obj_reps, att, module='answer'):
        image_feats = obj_reps['new_2d_objs'][:, 0, :,:]
        ori_q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        ori_a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)

        batch_size, ct, sent_len, emb_size = ori_q_rep.size()
        if self.rnn_input_dropout and self.training:
            ori_q_rep = self.rnn_input_dropout(ori_q_rep)
            ori_a_rep  = self.rnn_input_dropout(ori_a_rep)

        q_rep = self.vqa_seq_encoder(ori_q_rep.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        q_rep = get_final_encoder_states(q_rep, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        q_rep = q_rep.view(batch_size, ct, emb_size)
        #q_rep: the lstm representation of word grounded questions

        #attended_a: get attented answer representation from the visual answer ratinoale module
        a_batch_size, a_ct, a_sent_len, a_emb_size = ori_a_rep.size()
        if(module == 'answer'):
            qa_answer_mask = qa_answer_mask.unsqueeze(2).expand(a_batch_size, a_ct, 16, a_sent_len).contiguous().view(a_batch_size, -1, a_sent_len)
            ori_a_rep = ori_a_rep.unsqueeze(2).expand(a_batch_size, a_ct, 16, a_sent_len, a_emb_size).contiguous().view(
                a_batch_size, -1, a_sent_len, a_emb_size
            )
            q_rep = q_rep.unsqueeze(2).expand(batch_size, ct, 16, emb_size).contiguous().view(batch_size, -1, emb_size)
        if(module == 'rationale'):
            qa_answer_mask = qa_answer_mask.unsqueeze(1).expand(a_batch_size, 4, a_ct, a_sent_len).contiguous().view(a_batch_size, -1, a_sent_len)
            ori_a_rep = ori_a_rep.unsqueeze(1).expand(a_batch_size, 4, a_ct, a_sent_len, a_emb_size).contiguous().view(
                a_batch_size, -1, a_sent_len, a_emb_size
            )
            q_rep = q_rep.unsqueeze(1).expand(batch_size, 4, ct, emb_size).contiguous().view(batch_size, -1, emb_size)

        a_attention_weights = masked_softmax(att, qa_answer_mask[..., None], dim=2, double_flag=config.double_flag)
        attended_a = torch.einsum('bnqa,bnqd->bnad', (a_attention_weights, ori_a_rep)).squeeze()

        #get summary vector to combine grounded question representation and attented_answer representation
        batch_size, ct, sent_len, emb_size = ori_a_rep.size()
        mcb_emb = self.co_mcb(q_rep.view(-1, emb_size), attended_a.view(-1, emb_size))
        mcb_emb = F.relu(self.co_mcb_linear(mcb_emb)).view(batch_size, ct, 512)
        summary_vec =  mcb_emb

        #using summary vector to attend on image
        image_feats = image_feats.permute(0, 2, 3, 1)
        bs, ct1, ct2, dim = image_feats.size()
        image_feats = image_feats.view(bs, -1, dim).unsqueeze(1)
        summary_image_sim = self.get_image_attention(summary_vec, image_feats)
        summary_image_sim = F.softmax(summary_image_sim, -1)
        summary_image = torch.einsum('bnv, bcvd->bnd', (summary_image_sim, image_feats))

        summary_summary = self.image_text_mcb(summary_vec, summary_image)
        logits = self.vqa_score_layer(summary_summary).squeeze()
        return logits, summary_image_sim
    def get_text_attention(self, summary, text):
        bs, ct, len, dim = text.size()
        summary = summary.unsqueeze(2)
        sim = self.text_text_attention(
            summary.view(summary.shape[0]*summary.shape[1], summary.shape[2], summary.shape[3]),
            text.view(text.shape[0]*text.shape[1], text.shape[2], text.shape[3])
        ).view(bs, ct, summary.shape[2], text.shape[2])
        return torch.transpose(sim, -1, -2)

    def get_image_attention(self, summary, image):
        bs, ct, len, dim = image.size()
        summary = summary.unsqueeze(2)
        summary_bs, summary_ct, summary_len, summary_dim = summary.size()
        sim = self.text_image_attention1(
            summary.view(summary.shape[0], summary.shape[1]*summary.shape[2], summary.shape[3]),
            image.view(image.shape[0], image.shape[1]*image.shape[2], image.shape[3])
        )
        return sim


    def get_co_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        #non_linear layer to transform bert layer
        if config.double_flag:
            qa_question['bert'] = qa_question['bert'].double()
            qa_answer['bert'] = qa_answer['bert'].double()
        q_rep = F.relu(self.co_input_mlp(qa_question['bert']))
        a_rep = F.relu(self.co_input_mlp(qa_answer['bert']))

        #answer rationale co-attention
        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        #combined new answer representation
        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag=config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        #combined new question representation
        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag=config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        #bilstm for answer representation
        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.ar_seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        # self_attended_q = self_attended_q.view(batch_size, ct, emb_size)

        #bilstm for question representation
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.ar_seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        # self_attended_a = self_attended_a.view(batch_size, ct, emb_size)
        mcb_emb = self.co_mcb(self_attended_a, self_attended_q)
        mcb_emb = F.relu(self.co_mcb_linear(mcb_emb)).view(batch_size, ct, 512)

        #use mcb_emb to attend attended_a and attended_q
        q_att = self.get_text_attention(mcb_emb, attended_a)
        q_attention_weights = masked_softmax(q_att, qa_question_mask[..., None], dim=2, double_flag=config.double_flag)
        final_q = torch.einsum('bnqa,bnqd->bnad', (q_attention_weights, attended_a)).squeeze()

        a_att = self.get_text_attention(mcb_emb, attended_q)
        a_attention_weights = masked_softmax(a_att, qa_answer_mask[..., None], dim=2, double_flag=config.double_flag)
        final_a = torch.einsum('bnqa,bnqd->bnad', (a_attention_weights, attended_q)).squeeze()

        final_rep = torch.cat([final_q, final_a, torch.abs(final_q-final_a), final_q*final_a], dim=-1)
        logits = self.final_mlp(final_rep).squeeze()

        return logits, q_att, a_att, final_rep

    def get_factor_weights(self, answer_emb):
        factor_weight = F.relu(self.factor_layer(answer_emb))
        return F.softmax(factor_weight, dim=-1)


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))


        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)


        if not self.wo_ar:
            ar_logits, ar_a_att, ar_r_att, ar_rep  = self.get_co_logits(ar_answers, ar_answer_tags, ar_answer_mask,
                                                                ar_rationales, ar_rationale_tags, ar_rationale_mask, obj_reps)
        if(not self.wo_qa):
            qa_logits, qa_att = self.get_vqa_logits(qa_question, qa_question_tags, qa_question_mask,
                                                    qa_answers, qa_answer_tags, qa_answer_mask,
                                                    obj_reps, ar_a_att, module='answer')
        if not self.wo_qr:
            qr_logits, qr_att = self.get_vqa_logits(qr_question, qr_question_tags, qr_question_mask,
                                                    qr_rationales, qr_rationale_tags, qr_rationale_mask,
                                                    obj_reps, ar_r_att, module='rationale')
        batch_size =images.shape[0]

        if(self.multi_flag):
            qa_prob = F.softmax(qa_logits, dim=-1)
            qr_prob = F.softmax(qr_logits, dim=-1)

        if self.adaptive_flag:
            factor_weights = self.get_factor_weights(ar_rep)
        a_ct = 4
        r_ct = 16
        if(not self.wo_qa):
            new_qa_logits = qa_logits.view(batch_size, a_ct, r_ct)
        if not self.wo_qr:
            new_qr_logits = qr_logits.view(batch_size, a_ct, r_ct)
        if not self.wo_ar:
            new_ar_logits = ar_logits.view(batch_size, a_ct, r_ct)
        #rearange into 16 situations
        #situation 1:
        if self.wo_qa:
            logits = (new_qr_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_qr:
            logits = (new_qa_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_ar:
            logits = (new_qa_logits + new_qr_logits).view(batch_size, -1)
        else:
            if self.adaptive_flag:
                qa_weight =  factor_weights[:,:,0].view(new_qa_logits.size())
                qr_weight =  factor_weights[:,:,1].view(new_qr_logits.size())
                ar_weight = factor_weights[:,:,2].view(new_ar_logits.size())
                logits = (qa_weight*new_qa_logits + qr_weight*new_qr_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_qr_logits + new_ar_logits).view(batch_size, -1)

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       'qa_logits': new_qa_logits,
                       'qr_logits': new_qr_logits,
                       'ar_logits': new_ar_logits,
                       'test': obj_reps['img_feats']
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        if answer_label is not None and rationale_label is not None:
            final_label = answer_label*r_ct+rationale_label
            loss = self._loss(logits, final_label.long().view(-1))
            self._accuracy(logits, final_label)
            output_dict["loss"] = loss[None]
            if self.multi_flag:
                output_dict['answer_loss'] = self._loss(qa_logits, answer_label.long().view(-1))
                output_dict['rationale_loss'] = self._loss(qr_logits, rationale_label.long().view(-1))
            if self.adaptive_flag:
                output_dict['adaptive_weight'] = factor_weights

            #add jsd divergence between qa_att and qr_att
            eps = 1e-16
            if self.att_flag:
                mean_att = (qa_att+qr_att)/2.0
                qa_att = qa_att.view(-1, 196)
                qr_att = qr_att.view(-1, 196)
                mean_att = mean_att.view(-1, 196)
                # kl_loss = 0.5*self.kl_loss(torch.log(qa_att+eps), torch.log(mean_att+eps))+0.5*self.kl_loss(torch.log(qr_att+eps), torch.log(mean_att+eps))
                # kl_loss = self.kl_div_loss(qa_att, qr_att)
                kl_loss = 0.5*self.kl_div_loss(qa_att, mean_att)+0.5*self.kl_div_loss(qr_att, mean_att)
                # log_qa_att = torch.log(qa_att)
                # log_qr_att = torch.log(qr_att)
                # kl_loss = self.kl_loss(log_qa_att.view(-1, 196), log_qr_att.view(-1, 196))
                output_dict['kl_loss'] = kl_loss


        return output_dict

    def kl_div_loss(self, p, q):
        eps = 1e-16
        return torch.sum(p*torch.log(p/(q+eps)), dim=-1)

    def _image_text_co_attention_feats(self, span, img_feats):
        text = span['bert']
        if(config.double_flag):
            text = text.double()
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.text_image_attention(
            text.view(text.shape[0],text.shape[1]*text.shape[2], text.shape[3]),
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att = F.softmax(text_image_similarity,dim=-1)
        att_text = torch.bmm(att, flat_img_feats)
        # print(text_image_similarity.size())
        att_text = att_text.view(text.shape[0], text.shape[1], text.shape[2], -1)
        att_text = F.relu(self.image_mlp(att_text))
        return att_text, text_image_similarity


    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        image_feats = object_reps['new_2d_objs'][:, 0, :,:]
        # image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, object_reps['img_feats'])
        image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, image_feats)
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps['obj_reps'], image_attention_feats)

        if config.double_flag:
            span['bert'] = span['bert'].double()
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        return F.relu(self.input_mlp(span_rep)), retrieved_feats, text_image_similarity
    def _collect_obj_reps(self, span_tags, object_reps, image_attention_feats):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        #add image as box as false
        # span_mask = (span_tags>=0).float()
        #add image as box as true
        span_mask = (span_tags>0).float()

        image_mask = 1.0-span_mask

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        span_visual_feats =  object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

        if config.double_flag:
            span_mask = span_mask.double()
            image_mask = image_mask.double()
        final_fests = span_mask.unsqueeze(-1).expand_as(span_visual_feats)*span_visual_feats + image_mask.unsqueeze(-1).expand_as(image_attention_feats)*image_attention_feats
        return final_fests
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("MLP_QA")
class MLPQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 att_flag: bool = False,
                 multi_flag: bool = False,
                 adaptive_flag: bool = False,
                 wo_qa: bool = False,
                 wo_qr: bool = False,
                 wo_ar: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(MLPQA, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.dropout = input_dropout

        self.att_flag = att_flag
        self.multi_flag = multi_flag
        self.adaptive_flag = adaptive_flag
        self.wo_qa = wo_qa
        self.wo_qr = wo_qr
        self.wo_ar = wo_ar
        print('att flag: ', self.att_flag)
        print('wo_qa: ', self.wo_qa)
        print('wo_qr: ', self.wo_qr)
        print('wo_ar: ', self.wo_ar)
        print('multi flag: ', self.multi_flag)
        print('adaptive flag: ', self.adaptive_flag)
        self.span_encoder = span_encoder

        self.pre_final_layer = torch.nn.Linear(512*4, 512)
        self.final_layer = torch.nn.Linear(512, 1)


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)



    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if config.double_flag:
            images = images.double()
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))


        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        q_rep, q_obj_reps = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps['obj_reps'])
        r_rep, r_obj_reps = self.embed_span(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps['obj_reps'])
        bs, ct, sent_len, dim = q_rep.size()
        q_final = get_final_encoder_states(q_rep.view(bs*ct, sent_len, -1), qa_question_mask.view(bs*ct, sent_len), bidirectional=True)
        q_final = q_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = a_rep.size()
        a_final = get_final_encoder_states(a_rep.view(bs*ct, sent_len, -1), qa_answer_mask.view(bs*ct, sent_len), bidirectional=True)
        a_final = a_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = r_rep.size()
        r_final = get_final_encoder_states(r_rep.view(bs*ct, sent_len, -1), qr_rationale_mask.view(bs*ct, sent_len), bidirectional=True)
        r_final = r_final.view(bs, ct, -1)
        q_final =  torch.mean(q_final, dim=1, keepdim=True)
        a_ct = 4
        r_ct = 16
        q_final = q_final.expand(bs, a_ct*r_ct, dim)
        a_final = a_final.unsqueeze(2).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        r_final = r_final.unsqueeze(1).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        img_final = obj_reps['obj_reps'][:,0:1,:].expand(bs, a_ct*r_ct, dim)
        final_rep = torch.cat([img_final, q_final, a_final, r_final], dim=-1)

        pre_final_output = F.relu(self.pre_final_layer(final_rep))
        pre_final_output = F.dropout(pre_final_output, p=self.dropout, training=self.training)
        logits  = self.final_layer(pre_final_output).squeeze()


        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # 'test': obj_reps['ori_images']
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        #augmented joint modeling
        if answer_label is not None and rationale_label is not None:
            final_label = answer_label*r_ct+rationale_label
            loss = self._loss(logits, final_label.long().view(-1))
            self._accuracy(logits, final_label)
            output_dict["loss"] = loss[None]

        return output_dict


    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout and self.training:
            span_rep = self.rnn_input_dropout(span_rep)

        bs, ct, len, sz = span_rep.size()
        encoder_res = self.span_encoder(span_rep.view(-1, len, sz),span_mask.view(-1, len))
        return encoder_res.view(bs, ct,len, -1), retrieved_feats

        # return self.span_encoder(span_rep, span_mask), retrieved_feats
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("CRF_QA_weight1")
class CRFQA_W_QI(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 seq_encoder: Seq2SeqEncoder,
                 factor_lstm: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 att_flag: bool = False,
                 multi_flag: bool = False,
                 adaptive_flag: bool = False,
                 wo_qa: bool = False,
                 wo_qr: bool = False,
                 wo_ar: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(CRFQA_W_QI, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.att_flag = att_flag
        self.multi_flag = multi_flag
        self.adaptive_flag = adaptive_flag
        self.wo_qa = wo_qa
        self.wo_qr = wo_qr
        self.wo_ar = wo_ar
        print('att flag: ', self.att_flag)
        print('wo_qa: ', self.wo_qa)
        print('wo_qr: ', self.wo_qr)
        print('wo_ar: ', self.wo_ar)
        print('multi flag: ', self.multi_flag)
        print('adaptive flag: ', self.adaptive_flag)
        self.seq_encoder = seq_encoder
        if self.adaptive_flag:
            self.factor_lstm = factor_lstm
            self.factor_layer = torch.nn.Linear(1024, 3)
        self.co_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.text_image_attention = BilinearMatrixAttention(
            matrix_1_dim=768,
            matrix_2_dim=1024,
        )
        self.summary_image_attention = BilinearMatrixAttention(
            matrix_1_dim=1024,
            matrix_2_dim=1024,
        )
        self.weight_image_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=1024,
        )

        self.bilinear_layer = torch.nn.Bilinear(1024, 512,1, bias=False)

        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(1024, 512)

        self.input_mlp = torch.nn.Linear(1280, 512)
        # self.input_mlp = torch.nn.Linear(1792, 512)
        self.co_input_mlp = torch.nn.Linear(768, 512)
        # self.input_mlp = torch.nn.Linear(768, 512)

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(4*512, 1),
            # torch.nn.Linear(2*512, 1),
        )

        self.vqa_weight_layer = torch.nn.Linear(1536, 1)
        self.co_weight_layer = torch.nn.Linear(4*512, 1)
        # self.co_weight_layer = torch.nn.Linear(2*512, 1)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.margin_loss = torch.nn.MultiMarginLoss(margin=0.5, p=1)
        initializer(self)


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if config.double_flag:
            images = images.double()
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)


        if(not self.wo_qa):
            qa_logits, qa_att, qa_weight_logit = self.get_vqa_logits(qa_question, qa_question_tags, qa_question_mask, qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        if not self.wo_qr:
            qr_logits, qr_att, qr_weight_logit = self.get_vqa_logits(qr_question, qr_question_tags, qr_question_mask, qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        if not self.wo_ar:
            ar_logits, ar_weight_logit  = self.get_co_logits(ar_answers, ar_answer_tags, ar_answer_mask, ar_rationales, ar_rationale_tags, ar_rationale_mask, obj_reps)
        batch_size =images.shape[0]
        a_ct = qa_logits.size(1)
        r_ct = qr_logits.size(1)

        if self.adaptive_flag:
            factor_weights = self.get_factor_weights(obj_reps, qa_question, qa_question_tags, qa_question_mask)
            factor_weights = factor_weights.unsqueeze(2).expand(batch_size, a_ct, r_ct, 3).contiguous().view(batch_size, -1, 3)
        if(not self.wo_qa):
            new_qa_logits = qa_logits.unsqueeze(2).expand(batch_size, a_ct, r_ct)
        if not self.wo_qr:
            new_qr_logits = qr_logits.unsqueeze(1).expand(batch_size, a_ct, r_ct)
        if not self.wo_ar:
            new_ar_logits = ar_logits.view(batch_size, a_ct, r_ct)
        #rearange into 16 situations
        #situation 1:
        if self.wo_qa:
            logits = (new_qr_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_qr:
            logits = (new_qa_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_ar:
            logits = (new_qa_logits + new_qr_logits).view(batch_size, -1)
        else:
            if self.adaptive_flag:
                # qa_weight =  factor_weights[:,:,0].unsqueeze(2).expand_as(new_qa_logits)
                # qr_weight =  factor_weights[:,:,1].unsqueeze(2).expand_as(new_qr_logits)
                # ar_weight = factor_weights[:,:,2].unsqueeze(2).expand_as(new_ar_logits)
                qa_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                qr_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                ar_weight = factor_weights[:,:,2].view(batch_size, a_ct, r_ct)
                logits = (qa_weight*new_qa_logits + qr_weight*new_qr_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_qr_logits + new_ar_logits).view(batch_size, -1)

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       'qa_logits': new_qa_logits,
                       'qr_logits': new_qr_logits,
                       'ar_logits': new_ar_logits,
                       'qa_att': qa_att,
                       'qr_att': qr_att,
                       'test': qa_logits
                       # 'test': obj_reps['ori_images']
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        #augmented joint modeling
        if answer_label is not None and rationale_label is not None:
            final_label = answer_label*r_ct+rationale_label
            loss = self._loss(logits, final_label.long().view(-1))
            # loss = self.margin_loss(logits, final_label.long().view(-1))
            self._accuracy(logits, final_label)
            output_dict["loss"] = loss[None]
            if self.multi_flag:
                output_dict['answer_loss'] = self._loss(qa_logits, answer_label.long().view(-1))
                output_dict['rationale_loss'] = self._loss(qr_logits, rationale_label.long().view(-1))
            if self.adaptive_flag:
                output_dict['adaptive_weight'] = factor_weights
        return output_dict

    def get_vqa_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        img_feats = obj_reps['img_feats']
        # img_feats = obj_reps['new_2d_objs'][:,0,:,:]
        q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)
        text_summary = torch.cat([self_attended_q, self_attended_a], dim=-1)

        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.summary_image_attention(
            text_summary,
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        v_att = F.softmax(text_image_similarity, dim=2)
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
        att_img_feats = self.image_mlp(att_img_feats)

        concat_summary = torch.cat([text_summary, att_img_feats], dim=-1)
        final_weight = self.vqa_weight_layer(concat_summary).squeeze()

        logits = self.bilinear_layer(text_summary.contiguous(), att_img_feats.contiguous()).squeeze()

        # logits = self.final_mlp(self_attended_q+self_attended_a).squeeze()
        return logits, v_att, final_weight
    def get_co_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        # q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        # a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)
        #not using visual information in answer explanation module
        if config.double_flag:
            qa_question['bert'] = qa_question['bert'].double()
            qa_answer['bert'] = qa_answer['bert'].double()
        q_rep = F.relu(self.co_input_mlp(qa_question['bert']))
        a_rep = F.relu(self.co_input_mlp(qa_answer['bert']))

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)

        final_rep = torch.cat([self_attended_q, self_attended_a, torch.abs(self_attended_q-self_attended_a), self_attended_q*self_attended_a], dim=-1)
        # final_rep = torch.cat([self_attended_q, self_attended_a], dim=-1)
        logits = self.final_mlp(final_rep).squeeze()
        final_weight = self.co_weight_layer(final_rep).squeeze()
        return logits, final_weight
    def get_factor_weights(self, obj_reps, questions, question_tags, question_masks):
        img_feats = obj_reps['img_feats']
        q_rep, q_obj_reps, qi_sim = self.embed_span(questions, question_tags, question_masks, obj_reps)
        batch_size, ct, sent_len, emb_size = q_rep.size()
        if self.rnn_input_dropout and self.training:
            q_rep = self.rnn_input_dropout(q_rep)
        q_rep = self.seq_encoder(q_rep.view(batch_size*ct, sent_len, emb_size), question_masks.view(batch_size*ct, sent_len))
        q_rep = get_final_encoder_states(q_rep, question_masks.view(batch_size*ct, sent_len), bidirectional=True)
        q_rep = q_rep.view(batch_size, ct, emb_size)
        text_summary = q_rep

        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.weight_image_attention(
            text_summary,
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        v_att = F.softmax(text_image_similarity, dim=2)
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
        att_img_feats = self.image_mlp(att_img_feats)

        concat_summary = torch.cat([text_summary, att_img_feats], dim=-1)
        factor_weight = F.relu(self.factor_layer(concat_summary))
        return F.softmax(factor_weight, dim=-1)
    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        # image_feats = object_reps['new_2d_objs'][:, 0, :,:]
        image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, object_reps['img_feats'])
        # image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, image_feats)
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps['obj_reps'], image_attention_feats)
        # retrieved_feats = self._collect_obj_reps(span_tags, object_reps['new_1d_obj_reps'], image_attention_feats)

        if config.double_flag:
            span['bert'] = span['bert'].double()
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        return F.relu(self.input_mlp(span_rep)), retrieved_feats, text_image_similarity
    def _collect_obj_reps(self, span_tags, object_reps, image_attention_feats):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        #image not as a box
        span_mask = (span_tags>=0).float()
        #image as a box
        # span_mask = (span_tags>0).float()
        image_mask = 1.0-span_mask

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        span_visual_feats =  object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

        if config.double_flag:
            span_mask = span_mask.double()
            image_mask = image_mask.double()
        final_fests = span_mask.unsqueeze(-1).expand_as(span_visual_feats)*span_visual_feats + image_mask.unsqueeze(-1).expand_as(image_attention_feats)*image_attention_feats
        return final_fests
    def _image_text_co_attention_feats(self, span, img_feats):
        text = span['bert']
        if config.double_flag:
            text = text.double()
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.text_image_attention(
            text.view(text.shape[0],text.shape[1]*text.shape[2], text.shape[3]),
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att = F.softmax(text_image_similarity,dim=-1)
        att_text = torch.bmm(att, flat_img_feats)
        # print(text_image_similarity.size())
        att_text = att_text.view(text.shape[0], text.shape[1], text.shape[2], -1)
        att_text = F.relu(self.image_mlp(att_text))
        return att_text, text_image_similarity
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("CRF_QA_weight2")
class CRFQA_W_ALL(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 seq_encoder: Seq2SeqEncoder,
                 factor_lstm: Seq2SeqEncoder,
                 # text_seq_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 att_flag: bool = False,
                 multi_flag: bool = False,
                 adaptive_flag: bool = False,
                 wo_qa: bool = False,
                 wo_qr: bool = False,
                 wo_ar: bool = False,
                 wo_im: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(CRFQA_W_ALL, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.att_flag = att_flag
        self.multi_flag = multi_flag
        self.adaptive_flag = adaptive_flag
        self.wo_qa = wo_qa
        self.wo_qr = wo_qr
        self.wo_ar = wo_ar
        self.wo_im = wo_im
        print('att flag: ', self.att_flag)
        print('wo_qa: ', self.wo_qa)
        print('wo_qr: ', self.wo_qr)
        print('wo_ar: ', self.wo_ar)
        print('wo_im: ', self.wo_im)
        print('multi flag: ', self.multi_flag)
        print('adaptive flag: ', self.adaptive_flag)
        self.seq_encoder = seq_encoder
        # self.text_seq_encoder = text_seq_encoder
        if self.adaptive_flag:
            self.factor_lstm = factor_lstm
            # self.factor_layer = torch.nn.Linear(512*3, 3)
            self.factor_layer = torch.nn.Linear(512*4, 3)
            if self.wo_qa or self.wo_qr or self.wo_ar:
                self.partial_factor_layer = torch.nn.Linear(512*4, 2)
            # self.partial_factor_layer = torch.nn.Linear(512*4, 2)
        self.co_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.text_image_attention = BilinearMatrixAttention(
            matrix_1_dim=768,
            matrix_2_dim=1024,
        )
        self.summary_image_attention = BilinearMatrixAttention(
            matrix_1_dim=1024,
            matrix_2_dim=1024,
        )
        self.weight_image_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=1024,
        )

        if self.wo_im:
            self.wo_im_vqa_hidden_layer = torch.nn.Linear(1024, 512)
            self.wo_im_vqa_logit_layer = torch.nn.Linear(512, 1)

        self.bilinear_layer = torch.nn.Bilinear(1024, 512,1, bias=False)

        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(1024, 512)

        self.input_mlp = torch.nn.Linear(1280, 512)
        # self.input_mlp = torch.nn.Linear(1792, 512)
        self.co_input_mlp = torch.nn.Linear(768, 512)
        # self.input_mlp = torch.nn.Linear(768, 512)

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(4*512, 1),
            # torch.nn.Linear(2*512, 1),
        )

        self.vqa_weight_layer = torch.nn.Linear(1536, 1)
        self.co_weight_layer = torch.nn.Linear(4*512, 1)
        # self.co_weight_layer = torch.nn.Linear(2*512, 1)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.margin_loss = torch.nn.MultiMarginLoss(margin=0.5, p=1)
        initializer(self)


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if config.double_flag:
            images = images.double()
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)


        if(not self.wo_qa):
            qa_logits, qa_att, qa_weight_logit, qa_visual_f = self.get_vqa_logits(qa_question, qa_question_tags, qa_question_mask, qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        if not self.wo_qr:
            qr_logits, qr_att, qr_weight_logit, qr_visual_f = self.get_vqa_logits(qr_question, qr_question_tags, qr_question_mask, qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        if not self.wo_ar:
            ar_logits, ar_weight_logit, ar_att, inv_ar_att  = self.get_co_logits(ar_answers, ar_answer_tags, ar_answer_mask, ar_rationales, ar_rationale_tags, ar_rationale_mask, obj_reps)

        batch_size =images.shape[0]
        a_ct = 4
        r_ct = 16

        if self.adaptive_flag:
            factor_weights = self.get_factor_weights(obj_reps, qa_question, qa_question_tags, qa_question_mask,
                                                     qa_answers, qa_answer_tags, qa_answer_mask,
                                                     qr_rationales, qr_rationale_tags, qr_rationale_mask)
        if(not self.wo_qa):
            new_qa_logits = qa_logits.unsqueeze(2).expand(batch_size, a_ct, r_ct)
        if not self.wo_qr:
            new_qr_logits = qr_logits.unsqueeze(1).expand(batch_size, a_ct, r_ct)
        if not self.wo_ar:
            new_ar_logits = ar_logits.view(batch_size, a_ct, r_ct)
        #rearange into 16 situations
        #situation 1:
        if self.wo_qa:
            if self.adaptive_flag:
                qr_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                ar_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                logits = (qr_weight*new_qr_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qr_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_qr:
            if self.adaptive_flag:
                qa_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                ar_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                logits = (qa_weight*new_qa_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_ar:
            if self.adaptive_flag:
                qa_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                qr_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                logits = (qa_weight*new_qa_logits + qr_weight*new_qr_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_qr_logits).view(batch_size, -1)
        else:
            if self.adaptive_flag:
                qa_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                qr_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                ar_weight = factor_weights[:,:,2].view(batch_size, a_ct, r_ct)
                logits = (qa_weight*new_qa_logits + qr_weight*new_qr_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_qr_logits + new_ar_logits).view(batch_size, -1)

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # 'qa_logits': new_qa_logits,
                       # 'qr_logits': new_qr_logits,
                       # 'ar_logits': new_ar_logits,
                       # 'qa_att': qa_att,
                       # 'qr_att': qr_att,
                       # 'ar_att': ar_att,
                       # 'inv_ar_att': inv_ar_att,
                       # 'qa_logits': new_qa_logits,
                       # 'qr_logits': new_qr_logits,
                       # 'ar_logits': new_ar_logits,
                       # 'factor_weights': factor_weights
                       }
        #augmented joint modeling
        if answer_label is not None and rationale_label is not None:
            final_label = answer_label*r_ct+rationale_label
            loss = self._loss(logits, final_label.long().view(-1))
            self._accuracy(logits, final_label)
            output_dict["loss"] = loss[None]
            if self.multi_flag:
                if not self.wo_qa:
                    output_dict['answer_loss'] = self._loss(qa_logits, answer_label.long().view(-1))
                if not self.wo_qr:
                    output_dict['rationale_loss'] = self._loss(qr_logits, rationale_label.long().view(-1))
            if self.adaptive_flag:
                output_dict['adaptive_weight'] = factor_weights
        return output_dict

    def get_vqa_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        img_feats = obj_reps['img_feats']
        # img_feats = obj_reps['new_2d_objs'][:,0,:,:]
        q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)
        text_summary = torch.cat([self_attended_q, self_attended_a], dim=-1)

        v_att = None
        final_weight = None
        if not self.wo_im:
            trans_img_fests = img_feats.permute(0, 2, 3, 1)
            text_image_similarity = self.summary_image_attention(
                text_summary,
                trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
            )
            v_att = F.softmax(text_image_similarity, dim=2)
            flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
            att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
            att_img_feats = self.image_mlp(att_img_feats)

            concat_summary = torch.cat([text_summary, att_img_feats], dim=-1)
            final_weight = self.vqa_weight_layer(concat_summary).squeeze()

            logits = self.bilinear_layer(text_summary.contiguous(), att_img_feats.contiguous()).squeeze()
        else:
            hidden = F.relu(self.wo_im_vqa_hidden_layer(text_summary))
            logits = self.wo_im_vqa_logit_layer(hidden).squeeze()


        # logits = self.final_mlp(self_attended_q+self_attended_a).squeeze()
        return logits, v_att, final_weight, att_img_feats
    def get_co_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        # q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        # a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)
        #not using visual information in answer explanation module
        if config.double_flag:
            qa_question['bert'] = qa_question['bert'].double()
            qa_answer['bert'] = qa_answer['bert'].double()
        q_rep = F.relu(self.co_input_mlp(qa_question['bert']))
        a_rep = F.relu(self.co_input_mlp(qa_answer['bert']))

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        # self_attended_q = self.text_seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        # self_attended_a = self.text_seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)

        final_rep = torch.cat([self_attended_q, self_attended_a, torch.abs(self_attended_q-self_attended_a), self_attended_q*self_attended_a], dim=-1)
        # final_rep = torch.cat([self_attended_q, self_attended_a], dim=-1)
        logits = self.final_mlp(final_rep).squeeze()
        final_weight = self.co_weight_layer(final_rep).squeeze()
        return logits, final_weight, qa_attention_weights, inverse_qa_attention_weights
    def get_factor_weights(self, obj_reps, qa_question, qa_question_tags, qa_question_mask,
                            qa_answers, qa_answer_tags, qa_answer_mask,
                           qr_rationales, qr_rationale_tags, qr_rationale_mask,
                           ):
        q_rep, _, _ = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        a_rep, _, _ = self.embed_span(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        r_rep, _, _ = self.embed_span(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        bs, ct, sent_len, dim = q_rep.size()
        q_final = get_final_encoder_states(q_rep.view(bs*ct, sent_len, -1), qa_question_mask.view(bs*ct, sent_len), bidirectional=True)
        q_final = q_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = a_rep.size()
        a_final = get_final_encoder_states(a_rep.view(bs*ct, sent_len, -1), qa_answer_mask.view(bs*ct, sent_len), bidirectional=True)
        a_final = a_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = r_rep.size()
        r_final = get_final_encoder_states(r_rep.view(bs*ct, sent_len, -1), qr_rationale_mask.view(bs*ct, sent_len), bidirectional=True)
        r_final = r_final.view(bs, ct, -1)
        q_final =  torch.mean(q_final, dim=1, keepdim=True)
        a_ct = 4
        r_ct = 16
        q_final = q_final.expand(bs, a_ct*r_ct, dim)
        a_final = a_final.unsqueeze(2).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        r_final = r_final.unsqueeze(1).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        img_final = obj_reps['obj_reps'][:,0:1,:].expand(bs, a_ct*r_ct, dim)
        if self.wo_im:
            img_final = torch.cuda.FloatTensor(bs, a_ct*r_ct, dim).fill_(0)
        final_rep = torch.cat([img_final, q_final, a_final, r_final], dim=-1)
        # final_rep = torch.cat([q_final, a_final, r_final], dim=-1)

        if self.wo_qa or self.wo_qr  or self.wo_ar:
            factor_weight = F.relu(self.partial_factor_layer(final_rep))
        else:
            factor_weight = F.relu(self.factor_layer(final_rep))
        return F.softmax(factor_weight, dim=-1)
    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        # image_feats = object_reps['new_2d_objs'][:, 0, :,:]
        image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, object_reps['img_feats'])
        # image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, image_feats)
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps['obj_reps'], image_attention_feats)
        # retrieved_feats = self._collect_obj_reps(span_tags, object_reps['new_1d_obj_reps'], image_attention_feats)

        batch_size, ct, l, emb_size = retrieved_feats.size()
        if self.wo_im:
            retrieved_feats = torch.cuda.FloatTensor(batch_size, ct, l, emb_size).fill_(0)

        if config.double_flag:
            span['bert'] = span['bert'].double()
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        return F.relu(self.input_mlp(span_rep)), retrieved_feats, text_image_similarity
    def _collect_obj_reps(self, span_tags, object_reps, image_attention_feats):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        #image not as a box
        span_mask = (span_tags>=0).float()
        #image as a box
        # span_mask = (span_tags>0).float()
        image_mask = 1.0-span_mask

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        span_visual_feats =  object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

        if config.double_flag:
            span_mask = span_mask.double()
            image_mask = image_mask.double()
        final_fests = span_mask.unsqueeze(-1).expand_as(span_visual_feats)*span_visual_feats + image_mask.unsqueeze(-1).expand_as(image_attention_feats)*image_attention_feats
        return final_fests
    def _image_text_co_attention_feats(self, span, img_feats):
        text = span['bert']
        if config.double_flag:
            text = text.double()
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.text_image_attention(
            text.view(text.shape[0],text.shape[1]*text.shape[2], text.shape[3]),
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att = F.softmax(text_image_similarity,dim=-1)
        att_text = torch.bmm(att, flat_img_feats)
        # print(text_image_similarity.size())
        att_text = att_text.view(text.shape[0], text.shape[1], text.shape[2], -1)
        att_text = F.relu(self.image_mlp(att_text))
        return att_text, text_image_similarity
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("CRF_QA_SRL")
class CRFQA_SRL(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 seq_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 att_flag: bool = False,
                 multi_flag: bool = False,
                 adaptive_flag: bool = False,
                 wo_qa: bool = False,
                 wo_qr: bool = False,
                 wo_ar: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(CRFQA_SRL, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.srl_embed = torch.nn.Embedding(num_embeddings=84, embedding_dim=128)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.att_dropout = torch.nn.Dropout(input_dropout)

        self.srl_mapping = torch.nn.Linear(768, 128);


        self.position_encoding = PositionalEncoding(768)

        self.att_flag = att_flag
        self.multi_flag = multi_flag
        self.adaptive_flag = adaptive_flag
        self.wo_qa = wo_qa
        self.wo_qr = wo_qr
        self.wo_ar = wo_ar
        print('att flag: ', self.att_flag)
        print('wo_qa: ', self.wo_qa)
        print('wo_qr: ', self.wo_qr)
        print('wo_ar: ', self.wo_ar)
        print('multi flag: ', self.multi_flag)
        print('adaptive flag: ', self.adaptive_flag)
        self.seq_encoder = seq_encoder
        if self.adaptive_flag:
            self.factor_layer = torch.nn.Linear(2048, 3)
        self.co_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.text_image_attention = BilinearMatrixAttention(
            matrix_1_dim=768,
            matrix_2_dim=1024,
        )
        self.summary_image_attention = BilinearMatrixAttention(
            matrix_1_dim=1024,
            matrix_2_dim=1024,
        )
        # self.weight_image_attention = BilinearMatrixAttention(
        #     matrix_1_dim=512,
        #     matrix_2_dim=1024,
        # )

        self.bilinear_layer = torch.nn.Bilinear(1024, 512,1, bias=False)

        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(1024, 512)

        self.input_mlp = torch.nn.Linear(1280, 512)
        # self.input_mlp = torch.nn.Linear(1792, 512)
        self.co_input_mlp = torch.nn.Linear(768, 512)
        # self.input_mlp = torch.nn.Linear(768, 512)

        self.srl_mlp = torch.nn.Linear(768+128, 768)

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            # torch.nn.Linear(4*512, 1),
            torch.nn.Linear(2*512, 1),
        )

        # self.vqa_weight_layer = torch.nn.Linear(1536, 1)
        # self.co_weight_layer = torch.nn.Linear(4*512, 1)

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def embed_srl(self, sent):
        '''
        :param sent: batch_size, sent_ct, verb_ct, sent_len
        :return:
        sent_emb: embedding for each word's sematic role
        sent_mask: mask for srl padding, 1: non-padding  0: padding
        O_sent_mask: mask for O semantic role 1: non O srl, 0: O srl
        '''
        sent_mask = 1.0 - (sent==-1).long()
        # O_sent_mask = (sent!=0 and sent != -1).long()
        O_sent_mask = (sent > 0).long()
        clamp_sent = torch.clamp(sent, min=0)
        sent_emb = self.srl_embed(clamp_sent)
        # sent_emb = torch.mean(sent_emb, 1)
        return sent_emb, sent_mask, O_sent_mask

    def batched_index_select(self, input, dim, index):
        views = [input.shape[0]] + \
                [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def bert_srl_combination(self, sent, srl_emb,
                              srl_mask, srl_verb_index, srl_o_mask):
        '''
        :param sent: sent['bert'], batch size, sent_ct, sent_len, bert_emb_size
        :param sent_tags:
        :param sent_masks:
        :param srl_emb: batch size, sent_ct, verb_ct, sent_len, srl_emb_size
        :param srl_mask: batch_size, sent_ct, verb_ct
        :param srl_verb_index:batch_size, sent_ct, verb_ct
        :param srl_o_mask: batch_size, sent_ct, verb_ct, sent_len
        :return:
        '''

        def self_attention_func(input, mask):
            '''

            :param input:  batch_size, sent_ct, sent_len, emb_size
            :param mask: batch_size, sent_ct, sent_len
            :return:
            '''
            batch_size, sent_ct, sent_len, emb_size = input.size()
            reshape_input = input.contiguous().view(batch_size*sent_ct, sent_len, emb_size)
            reshape_mask = mask.contiguous().view(batch_size*sent_ct, sent_len)
            temp1 = reshape_mask.float().unsqueeze(2)
            temp2 = reshape_mask.unsqueeze(1).float()
            new_mask = torch.bmm(temp1, temp2)
            scores = torch.bmm(reshape_input, reshape_input.transpose(-2, -1))/math.sqrt(emb_size)
            scores = scores.masked_fill(new_mask==0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
            p_attn = self.att_dropout(p_attn)
            res = torch.matmul(p_attn, reshape_input)
            res = res.view(batch_size, sent_ct, sent_len, emb_size)
            return res

        #first step: composite the verb's bert representation with the srl_tag representation for each verb
        batch_size, sent_ct, sent_len, bert_emb_size =  sent['bert'].size()
        new_bert_emb_size = 128
        batch_size, sent_ct, verb_ct = srl_mask.size()
        all_verbs_srl_emb  = torch.cuda.FloatTensor(verb_ct, batch_size, sent_ct, sent_len, bert_emb_size+new_bert_emb_size).fill_(0)
        for i in range(verb_ct):
            srl_tag_emb = srl_emb[:, :, i, :, :] # batch_size,  sent_ct, sent_len, 128
            srl_o_tag_mask = srl_o_mask[:,:, i,:] # batch_size, sent_ct, sent_len

            combined_input = torch.cat([sent['bert'], srl_tag_emb], dim=-1)
            combined_output = self_attention_func(combined_input, srl_o_tag_mask)
            all_verbs_srl_emb[i,:,:,:,:] = combined_output

        all_verbs_srl_emb = all_verbs_srl_emb.permute(1,2,3,0,4)
        expand_srl_mask = srl_mask.unsqueeze(2).unsqueeze(-1).expand(batch_size, sent_ct, sent_len, verb_ct, new_bert_emb_size+bert_emb_size).float()
        # expand_srl_verb_mask = srl_mask
        final_srl_emb = torch.sum(all_verbs_srl_emb*expand_srl_mask, dim=-2)

        combined_emb = F.relu(self.srl_mlp(final_srl_emb), inplace=True)
        bs, sent_ct, sent_len, emb_size = combined_emb.size()
        combined_emb = self.position_encoding(combined_emb.view(bs*sent_ct, sent_len, emb_size)).view(bs, sent_ct, sent_len, emb_size)
        sent['bert'] = combined_emb
        return combined_emb


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                question_srl:torch.LongTensor,
                question_srl_mask:torch.LongTensor,
                question_srl_verb:torch.LongTensor,
                answer_srl:torch.LongTensor,
                answer_srl_mask:torch.LongTensor,
                answer_srl_verb:torch.LongTensor,
                rationale_srl:torch.LongTensor,
                rationale_srl_mask:torch.LongTensor,
                rationale_srl_verb:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param question_srl_mask: batch_size, verb_ct  mask on the question srl verbs
        :param answer_srl_mask, batch_size, 4, verb_ct  mask on the answer srl verbs
        :param rationale_srl_mask batch_size, 16, verb_ct mask on the rationale srl verbs

        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if config.double_flag:
            images = images.double()
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        #srl_embedding for question, answer and rationale
        #the rerturn mask is the mask for the padding
        question_srl_emb, question_srl_emb_mask, question_srl_o_mask = self.embed_srl(question_srl)
        answer_srl_emb, answer_srl_emb_mask, answer_srl_o_mask = self.embed_srl(answer_srl)
        rationale_srl_emb, rationale_srl_emb_mask, rationale_srl_o_mask = self.embed_srl(rationale_srl)


        q_bs, q_v_ct, q_sent_len, q_emb_sz = question_srl_emb.size()
        qa_question_srl_emb = question_srl_emb.unsqueeze(1).expand(q_bs, 4, q_v_ct, q_sent_len, q_emb_sz)
        qa_question_srl_mask = question_srl_mask.unsqueeze(1).expand(q_bs, 4, q_v_ct)
        qa_question_srl_verb = question_srl_verb.unsqueeze(1).expand(q_bs, 4, q_v_ct)
        qa_question_srl_o_mask = question_srl_o_mask.unsqueeze(1).expand(q_bs, 4,
                                                                         q_v_ct, q_sent_len)
        self.bert_srl_combination(qa_question,
                                  qa_question_srl_emb, qa_question_srl_mask, qa_question_srl_verb, qa_question_srl_o_mask)

        qa_answer_srl_mask = answer_srl_mask
        qa_answer_srl_verb = answer_srl_verb
        qa_answer_srl_o_mask = answer_srl_o_mask
        self.bert_srl_combination(qa_answers,
                                  answer_srl_emb, qa_answer_srl_mask, qa_answer_srl_verb, qa_answer_srl_o_mask)


        qr_question_srl_emb = question_srl_emb.unsqueeze(1).expand(q_bs, 16, q_v_ct, q_sent_len, q_emb_sz)
        qr_question_srl_mask = question_srl_mask.unsqueeze(1).expand(q_bs, 16, q_v_ct)
        qr_question_srl_verb = question_srl_verb.unsqueeze(1).expand(q_bs, 16, q_v_ct)
        qr_question_srl_o_mask = question_srl_o_mask.unsqueeze(1).expand(q_bs, 16,
                                                                         q_v_ct, q_sent_len)
        self.bert_srl_combination(qr_question,
                                  qr_question_srl_emb, qr_question_srl_mask, qr_question_srl_verb, qr_question_srl_o_mask)


        qr_rationale_srl_mask = rationale_srl_mask
        qr_rationale_srl_verb = rationale_srl_verb
        qr_rationale_srl_o_mask = rationale_srl_o_mask
        self.bert_srl_combination(qr_rationales,
                                  rationale_srl_emb, qr_rationale_srl_mask, qr_rationale_srl_verb, qr_rationale_srl_o_mask)

        a_bs, a_sent_ct, a_v_ct, a_sent_len, a_emb_sz = answer_srl_emb.size()
        ar_answer_srl_emb = answer_srl_emb.unsqueeze(2).expand(a_bs, a_sent_ct, 16, a_v_ct, a_sent_len, a_emb_sz).contiguous().view(a_bs, 64, a_v_ct, a_sent_len, a_emb_sz)
        ar_answer_srl_mask = answer_srl_mask.unsqueeze(2).expand(a_bs, a_sent_ct, 16, a_v_ct).contiguous().view(a_bs, 64, a_v_ct)
        ar_answer_srl_verb = answer_srl_verb.unsqueeze(2).expand(a_bs, a_sent_ct, 16, a_v_ct).contiguous().view(a_bs, 64, a_v_ct)
        ar_answer_srl_o_mask = answer_srl_o_mask.unsqueeze(2).expand(a_bs, a_sent_ct, 16,
                                                                     a_v_ct, a_sent_len).contiguous().view(a_bs, 64, a_v_ct, a_sent_len)
        self.bert_srl_combination(ar_answers,
                                  ar_answer_srl_emb, ar_answer_srl_mask, ar_answer_srl_verb, ar_answer_srl_o_mask)

        r_bs, r_sent_ct, r_v_ct, r_sent_len, r_emb_sz = rationale_srl_emb.size()
        ar_rationale_srl_emb = rationale_srl_emb.unsqueeze(1).expand(r_bs, 4, r_sent_ct, r_v_ct, r_sent_len, r_emb_sz).contiguous().view(r_bs, 64, r_v_ct, r_sent_len, r_emb_sz)
        ar_rationale_srl_mask = rationale_srl_mask.unsqueeze(1).expand(r_bs, 4, r_sent_ct, r_v_ct).contiguous().view(r_bs, 64, r_v_ct)
        ar_rationale_srl_verb = rationale_srl_verb.unsqueeze(1).expand(r_bs, 4, r_sent_ct, r_v_ct).contiguous().view(r_bs, 64, r_v_ct)
        ar_rationale_srl_o_mask = rationale_srl_o_mask.unsqueeze(1).expand(r_bs, 4, r_sent_ct,
                                                                           r_v_ct, r_sent_len).contiguous().view(r_bs, 64, r_v_ct, r_sent_len)
        self.bert_srl_combination(ar_rationales,
                                  ar_rationale_srl_emb, ar_rationale_srl_mask, ar_rationale_srl_verb, ar_rationale_srl_o_mask)

        qa_q_rep, qa_q_obj_reps, qa_qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        qa_a_rep, qa_a_obj_reps, qa_ai_sim = self.embed_span(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        qr_q_rep, qr_q_obj_reps, qr_qi_sim = self.embed_span(qr_question, qr_question_tags, qr_question_mask, obj_reps)
        qr_r_rep, qr_r_obj_reps, qr_ri_sim = self.embed_span(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        if(not self.wo_qa):
            qa_logits, qa_att  = self.get_vqa_logits(qa_question, qa_question_tags, qa_question_mask,
                                                                     qa_answers, qa_answer_tags, qa_answer_mask, obj_reps, qa_q_rep, qa_a_rep)

        if not self.wo_qr:
            qr_logits, qr_att  = self.get_vqa_logits(qr_question, qr_question_tags, qr_question_mask, qr_rationales, qr_rationale_tags,
                                                     qr_rationale_mask, obj_reps, qr_q_rep, qr_r_rep)
        if not self.wo_ar:
            ar_logits = self.get_co_logits(ar_answers, ar_answer_tags, ar_answer_mask, ar_rationales, ar_rationale_tags, ar_rationale_mask, obj_reps)
        batch_size =images.shape[0]
        a_ct = qa_logits.size(1)
        r_ct = qr_logits.size(1)

        if self.adaptive_flag:
            factor_weights = self.get_factor_weights(obj_reps, qa_question, qa_question_tags, qa_question_mask,
                                                     qa_answers, qa_answer_tags, qa_answer_mask,
                                                     qr_rationales, qr_rationale_tags, qr_rationale_mask,
                                                     qa_q_rep, qa_a_rep, qr_r_rep)
        if(not self.wo_qa):
            new_qa_logits = qa_logits.unsqueeze(2).expand(batch_size, a_ct, r_ct)
        if not self.wo_qr:
            new_qr_logits = qr_logits.unsqueeze(1).expand(batch_size, a_ct, r_ct)
        if not self.wo_ar:
            new_ar_logits = ar_logits.view(batch_size, a_ct, r_ct)
        #rearange into 16 situations
        #situation 1:
        if self.wo_qa:
            logits = (new_qr_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_qr:
            logits = (new_qa_logits + new_ar_logits).view(batch_size, -1)
        elif self.wo_ar:
            logits = (new_qa_logits + new_qr_logits).view(batch_size, -1)
        else:
            if self.adaptive_flag:
                # qa_weight =  factor_weights[:,:,0].unsqueeze(2).expand_as(new_qa_logits)
                # qr_weight =  factor_weights[:,:,1].unsqueeze(2).expand_as(new_qr_logits)
                # ar_weight = factor_weights[:,:,2].unsqueeze(2).expand_as(new_ar_logits)
                qa_weight = factor_weights[:,:,0].view(batch_size, a_ct, r_ct)
                qr_weight = factor_weights[:,:,1].view(batch_size, a_ct, r_ct)
                ar_weight = factor_weights[:,:,2].view(batch_size, a_ct, r_ct)
                logits = (qa_weight*new_qa_logits + qr_weight*new_qr_logits + ar_weight*new_ar_logits).view(batch_size, -1)
            else:
                logits = (new_qa_logits + new_qr_logits + new_ar_logits).view(batch_size, -1)

        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # 'qa_logits': new_qa_logits,
                       # 'qr_logits': new_qr_logits,
                       # 'ar_logits': new_ar_logits,
                       # 'qa_att': qa_att,
                       # 'qr_att': qr_att,
                       # 'test': qa_logits
                       # 'test': obj_reps['ori_images']
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }
        #augmented joint modeling
        if answer_label is not None and rationale_label is not None:
            final_label = answer_label*r_ct+rationale_label
            loss = self._loss(logits, final_label.long().view(-1))
            # loss = self.margin_loss(logits, final_label.long().view(-1))
            self._accuracy(logits, final_label)
            output_dict["loss"] = loss[None]
            if self.multi_flag:
                output_dict['answer_loss'] = self._loss(qa_logits, answer_label.long().view(-1))[None]
                output_dict['rationale_loss'] = self._loss(qr_logits, rationale_label.long().view(-1))[None]
            if self.adaptive_flag:
                output_dict['adaptive_weight'] = factor_weights
        return output_dict

    def get_vqa_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps, q_rep, a_rep):
        img_feats = obj_reps['img_feats']
        # img_feats = obj_reps['new_2d_objs'][:,0,:,:]

        # q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        # a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)
        text_summary = torch.cat([self_attended_q, self_attended_a], dim=-1)

        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.summary_image_attention(
            text_summary,
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        v_att = F.softmax(text_image_similarity, dim=2)
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
        att_img_feats = self.image_mlp(att_img_feats)

        concat_summary = torch.cat([text_summary, att_img_feats], dim=-1)
        # final_weight = self.vqa_weight_layer(concat_summary).squeeze()

        logits = self.bilinear_layer(text_summary.contiguous(), att_img_feats.contiguous()).squeeze()

        # logits = self.final_mlp(self_attended_q+self_attended_a).squeeze()
        return logits, v_att
    def get_co_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        # q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        # a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)
        #not using visual information in answer explanation module
        if config.double_flag:
            qa_question['bert'] = qa_question['bert'].double()
            qa_answer['bert'] = qa_answer['bert'].double()
        q_rep = F.relu(self.co_input_mlp(qa_question['bert']), inplace=True)
        a_rep = F.relu(self.co_input_mlp(qa_answer['bert']), inplace=True)

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)), inplace=True)+a_rep

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)),inplace=True)+q_rep

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_q = self_attended_q.view(batch_size, ct, emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, emb_size)

        # final_rep = torch.cat([self_attended_q, self_attended_a, torch.abs(self_attended_q-self_attended_a), self_attended_q*self_attended_a], dim=-1)
        final_rep = torch.cat([self_attended_q, self_attended_a], dim=-1)
        logits = self.final_mlp(final_rep).squeeze()
        # final_weight = self.co_weight_layer(final_rep).squeeze()
        return logits
    def get_factor_weights(self, obj_reps, qa_question, qa_question_tags, qa_question_mask,
                           qa_answers, qa_answer_tags, qa_answer_mask,
                           qr_rationales, qr_rationale_tags, qr_rationale_mask,
                           q_rep, a_rep, r_rep):
        # q_rep, _, _ = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        # a_rep, _, _ = self.embed_span(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        # r_rep, _, _ = self.embed_span(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        bs, ct, sent_len, dim = q_rep.size()
        q_final = get_final_encoder_states(q_rep.view(bs*ct, sent_len, -1), qa_question_mask.view(bs*ct, sent_len), bidirectional=True)
        q_final = q_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = a_rep.size()
        a_final = get_final_encoder_states(a_rep.view(bs*ct, sent_len, -1), qa_answer_mask.view(bs*ct, sent_len), bidirectional=True)
        a_final = a_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = r_rep.size()
        r_final = get_final_encoder_states(r_rep.view(bs*ct, sent_len, -1), qr_rationale_mask.view(bs*ct, sent_len), bidirectional=True)
        r_final = r_final.view(bs, ct, -1)
        q_final =  torch.mean(q_final, dim=1, keepdim=True)
        a_ct = 4
        r_ct = 16
        q_final = q_final.expand(bs, a_ct*r_ct, dim)
        a_final = a_final.unsqueeze(2).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        r_final = r_final.unsqueeze(1).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        img_final = obj_reps['obj_reps'][:,0:1,:].expand(bs, a_ct*r_ct, dim)
        final_rep = torch.cat([img_final, q_final, a_final, r_final], dim=-1)

        factor_weight = F.relu(self.factor_layer(final_rep), inplace=True)
        return F.softmax(factor_weight, dim=-1)
    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        # image_feats = object_reps['new_2d_objs'][:, 0, :,:]
        image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, object_reps['img_feats'])
        # image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, image_feats)
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps['obj_reps'], image_attention_feats)
        # retrieved_feats = self._collect_obj_reps(span_tags, object_reps['new_1d_obj_reps'], image_attention_feats)

        if config.double_flag:
            span['bert'] = span['bert'].double()
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        return F.relu(self.input_mlp(span_rep),inplace=True), retrieved_feats, text_image_similarity
    def _collect_obj_reps(self, span_tags, object_reps, image_attention_feats):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        #image not as a box
        span_mask = (span_tags>=0).float()
        #image as a box
        # span_mask = (span_tags>0).float()
        image_mask = 1.0-span_mask

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        span_visual_feats =  object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

        if config.double_flag:
            span_mask = span_mask.double()
            image_mask = image_mask.double()
        final_fests = span_mask.unsqueeze(-1).expand_as(span_visual_feats)*span_visual_feats + image_mask.unsqueeze(-1).expand_as(image_attention_feats)*image_attention_feats
        return final_fests
    def _image_text_co_attention_feats(self, span, img_feats):
        text = span['bert']
        if config.double_flag:
            text = text.double()
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.text_image_attention(
            text.view(text.shape[0],text.shape[1]*text.shape[2], text.shape[3]),
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att = F.softmax(text_image_similarity,dim=-1)
        att_text = torch.bmm(att, flat_img_feats)
        # print(text_image_similarity.size())
        att_text = att_text.view(text.shape[0], text.shape[1], text.shape[2], -1)
        att_text = F.relu(self.image_mlp(att_text), inplace=True)
        return att_text, text_image_similarity
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

@Model.register("VAE_QA")
class VAE_QA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 seq_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 class_embs: bool=True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(VAE_QA, self).__init__(vocab)
        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.dropout_layer = torch.nn.Dropout(input_dropout)

        self.seq_encoder = seq_encoder
        self.co_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=512,
        )
        self.text_image_attention = BilinearMatrixAttention(
            matrix_1_dim=768,
            matrix_2_dim=1024,
        )
        self.summary_image_attention = BilinearMatrixAttention(
            matrix_1_dim=1024,
            matrix_2_dim=1024,
        )
        self.recog_summary_image_attention = BilinearMatrixAttention(
            matrix_1_dim=1536,
            matrix_2_dim=1024,
        )
        self.weight_image_attention = BilinearMatrixAttention(
            matrix_1_dim=512,
            matrix_2_dim=1024,
        )

        self.bilinear_layer = torch.nn.Bilinear(1024, 512,1, bias=False)

        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(1024, 512)

        self.input_mlp = torch.nn.Linear(1280, 512)
        # self.input_mlp = torch.nn.Linear(1792, 512)
        self.co_input_mlp = torch.nn.Linear(768, 512)
        # self.input_mlp = torch.nn.Linear(768, 512)

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(4*512, 1),
            # torch.nn.Linear(2*512, 1),
        )

        self.vqa_weight_layer = torch.nn.Linear(1536, 1)
        self.co_weight_layer = torch.nn.Linear(4*512, 1)
        # self.co_weight_layer = torch.nn.Linear(2*512, 1)


        self.prior_logit_hidden = torch.nn.Linear(1536, 512)
        self.prior_logit_layer = torch.nn.Linear(512, 1)

        self.recog_logit_hidden = torch.nn.Linear(512*4, 512)
        self.recog_logit_layer = torch.nn.Linear(512, 1)

        self.perform_logit_hidden = torch.nn.Linear(512*3, 512)
        self.perform_logit_layer = torch.nn.Linear(512, 1)


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        self.margin_loss = torch.nn.MultiMarginLoss(margin=0.5, p=1)
        initializer(self)

    def evidence_embedding(self, qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps):
        r_rep, a_obj_reps, ai_sim = self.embed_span(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        batch_size, ct, sent_len, emb_size = r_rep.size()
        r_rep = self.seq_encoder(r_rep.view(batch_size*ct, sent_len, emb_size), qr_rationale_mask.view(batch_size*ct, sent_len))
        final_r_rep = get_final_encoder_states(r_rep, qr_rationale_mask.view(batch_size*ct, sent_len), bidirectional=True)
        final_r_rep = final_r_rep.view(batch_size, ct, emb_size)
        return final_r_rep


    def forward(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if self.training:
            if config.double_flag:
                images = images.double()
            # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
            # not needed
            max_len = int(box_mask.sum(1).max().item())
            objects = objects[:, :max_len]
            box_mask = box_mask[:, :max_len]
            boxes = boxes[:, :max_len]
            # segms = segms[:, :max_len]


            for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
                if int(the_tags.max()) > max_len:
                    raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                        tag_type, int(the_tags.max()), objects.shape, the_tags
                    ))

            obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
            img_feats = obj_reps['img_feats']

            #first step: get all rationales embedding
            #return size: batch_size x 16 x emb_dim
            rationale_embedding = self.evidence_embedding(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)

            #second step: get gold answer embedding
            #return size: batch_size x 4 x emb_dim
            answer_embedding = self.evidence_embedding(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
            #third step: compute classification loss for different answers
            batch_size = answer_embedding.size(0)
            gold_answer_one_hot = torch.cuda.FloatTensor(batch_size, 4)
            gold_answer_one_hot.zero_()
            answer_label = answer_label.unsqueeze(1)
            gold_answer_one_hot.scatter_(1, answer_label, 1)
            gold_answer_one_hot_expand = gold_answer_one_hot.unsqueeze(-1)
            transposed_answer_embedding = torch.transpose(answer_embedding, 1, 2)
            gold_answer_embedding = torch.bmm(transposed_answer_embedding, gold_answer_one_hot_expand).squeeze()

            #third step: get question embedding
            question_embeddings = self.evidence_embedding(qr_question, qr_question_tags, qr_question_mask, obj_reps)




            #fourth step: build prior network: given question, and rationale, prdict 1 of K distribution
            concat_prior_input = torch.cat([question_embeddings, rationale_embedding], dim=-1)
            trans_img_fests = img_feats.permute(0, 2, 3, 1)
            text_image_similarity = self.summary_image_attention(
                concat_prior_input,
                trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
            )
            v_att = F.softmax(text_image_similarity, dim=2)
            flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
            att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
            att_img_feats = self.image_mlp(att_img_feats)

            prior_concat_summary = torch.cat([concat_prior_input, att_img_feats], dim=-1)
            prior_hidden = F.relu(self.prior_logit_hidden(prior_concat_summary))
            prior_hidden = self.dropout_layer(prior_hidden)
            prior_logit = self.prior_logit_layer(prior_hidden).squeeze()


            #fifth step:
            #build recognition network: given gold answer, question and
            expand_gold_answer_embedding = gold_answer_embedding.unsqueeze(1).expand(batch_size, 16, 512)
            concat_recog_input = torch.cat([expand_gold_answer_embedding, question_embeddings, rationale_embedding], dim=-1)
            text_image_similarity = self.recog_summary_image_attention(
                concat_recog_input,
                trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
            )
            v_att = F.softmax(text_image_similarity, dim=2)
            flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
            att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
            att_img_feats = self.image_mlp(att_img_feats)

            recog_concat_summary = torch.cat([concat_recog_input, att_img_feats], dim=-1)
            recog_hidden = F.relu(self.recog_logit_hidden(recog_concat_summary)).squeeze()
            recog_hidden = self.dropout_layer(recog_hidden)
            recog_logit = self.recog_logit_layer(recog_hidden).squeeze()
            recog_discrete_z = gumble_softmax(recog_logit, config.gumble_temperature)

            #sixth step:
            #build perform network based on sampled recognition discrete z
            transpose_rationale_embedding = torch.transpose(rationale_embedding, 1, 2)
            expand_recog_discrete_z = recog_discrete_z.unsqueeze(2)
            aggregate_rationale_embedding = torch.bmm(transpose_rationale_embedding, expand_recog_discrete_z).squeeze()


            expand_agg_ral_emb = aggregate_rationale_embedding.unsqueeze(1).expand(batch_size, 4, 512)
            qa_question_emb = self.evidence_embedding(qa_question, qa_question_tags, qa_question_mask, obj_reps)
            combined_answer_rationale_emb = torch.cat([qa_question_emb, expand_agg_ral_emb, answer_embedding], dim=-1)
            answer_hidden = F.relu(self.perform_logit_hidden(combined_answer_rationale_emb))
            answer_hidden = self.dropout_layer(answer_hidden)
            answer_logit = self.perform_logit_layer(answer_hidden).squeeze()
            answer_class_probabilities = F.softmax(answer_logit, dim=-1)

            output_dict = {
                           'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                           }
            self._accuracy(answer_logit, answer_label.long().view(-1))
            answer_loss = self._loss(answer_logit, answer_label.long().view(-1))
            rationale_loss = self._loss(recog_logit, rationale_label.long().view(-1))
            kl_loss = kl_divergence_loss(prior_logit, recog_logit)
            output_dict['answer_loss'] = answer_loss
            output_dict['rationale_loss'] = rationale_loss
            output_dict['kl_loss'] = kl_loss
            return output_dict
        else:
            return self.predict(images, objects, segms, boxes, box_mask, qa_question, qa_question_tags, qa_question_mask,
                         qr_question, qr_question_tags, qr_question_mask, qa_answers, qa_answer_tags, qa_answer_mask,
                         qr_rationales, qr_rationale_tags, qr_rationale_mask, ar_answers, ar_answer_tags, ar_answer_mask,
                         ar_rationales, ar_rationale_tags, ar_rationale_mask, rationale_mask, ind, rationale_label, answer_label, label)


    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        # image_feats = object_reps['new_2d_objs'][:, 0, :,:]
        image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, object_reps['img_feats'])
        # image_attention_feats, text_image_similarity = self._image_text_co_attention_feats(span, image_feats)
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps['obj_reps'], image_attention_feats)
        # retrieved_feats = self._collect_obj_reps(span_tags, object_reps['new_1d_obj_reps'], image_attention_feats)

        if config.double_flag:
            span['bert'] = span['bert'].double()
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        return F.relu(self.input_mlp(span_rep)), retrieved_feats, text_image_similarity
    def _collect_obj_reps(self, span_tags, object_reps, image_attention_feats):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        #image not as a box
        span_mask = (span_tags>=0).float()
        #image as a box
        # span_mask = (span_tags>0).float()
        image_mask = 1.0-span_mask

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        span_visual_feats =  object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

        if config.double_flag:
            span_mask = span_mask.double()
            image_mask = image_mask.double()
        final_fests = span_mask.unsqueeze(-1).expand_as(span_visual_feats)*span_visual_feats + image_mask.unsqueeze(-1).expand_as(image_attention_feats)*image_attention_feats
        return final_fests

    def make_one_hot(self, answer_label, c):
        batch_size = len(answer_label)
        gold_answer_one_hot = torch.cuda.FloatTensor(batch_size, c)
        gold_answer_one_hot.zero_()
        answer_label = answer_label.unsqueeze(1)
        gold_answer_one_hot.scatter_(1, answer_label, 1)
        return gold_answer_one_hot

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
    def _image_text_co_attention_feats(self, span, img_feats):
        text = span['bert']
        if config.double_flag:
            text = text.double()
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.text_image_attention(
            text.view(text.shape[0],text.shape[1]*text.shape[2], text.shape[3]),
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att = F.softmax(text_image_similarity,dim=-1)
        att_text = torch.bmm(att, flat_img_feats)
        # print(text_image_similarity.size())
        att_text = att_text.view(text.shape[0], text.shape[1], text.shape[2], -1)
        att_text = F.relu(self.image_mlp(att_text))
        return att_text, text_image_similarity
    def predict(self,
                images: torch.Tensor,
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                qa_question: Dict[str, torch.Tensor],
                qa_question_tags: torch.LongTensor,
                qa_question_mask: torch.LongTensor,
                qr_question: Dict[str, torch.Tensor],
                qr_question_tags: torch.LongTensor,
                qr_question_mask: torch.LongTensor,
                qa_answers: Dict[str, torch.Tensor],
                qa_answer_tags: torch.LongTensor,
                qa_answer_mask: torch.LongTensor,
                qr_rationales: Dict[str, torch.Tensor],
                qr_rationale_tags: torch.LongTensor,
                qr_rationale_mask:torch.LongTensor,
                ar_answers: Dict[str, torch.Tensor],
                ar_answer_tags: torch.LongTensor,
                ar_answer_mask: torch.LongTensor,
                ar_rationales: Dict[str, torch.Tensor],
                ar_rationale_tags: torch.LongTensor,
                ar_rationale_mask:torch.LongTensor,
                rationale_mask: torch.Tensor = None,
                ind: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                label: torch.LongTensor = None,
                ) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param ind: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        if config.double_flag:
            images = images.double()
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        # segms = segms[:, :max_len]


        for tag_type, the_tags in (('question', qa_question_tags), ('answer', qa_answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        img_feats = obj_reps['img_feats']

        #first step: get all rationales embedding
        #return size: batch_size x 16 x emb_dim
        rationale_embedding = self.evidence_embedding(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)

        #second step: get gold answer embedding
        #return size: batch_size x 4 x emb_dim
        answer_embedding = self.evidence_embedding(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)

        #third step: get question embedding
        question_embeddings = self.evidence_embedding(qr_question, qr_question_tags, qr_question_mask, obj_reps)


        #fourth step: build prior network: given question, and rationale, prdict 1 of K distribution
        concat_prior_input = torch.cat([question_embeddings, rationale_embedding], dim=-1)
        trans_img_fests = img_feats.permute(0, 2, 3, 1)
        text_image_similarity = self.summary_image_attention(
            concat_prior_input,
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        v_att = F.softmax(text_image_similarity, dim=2)
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
        att_img_feats = self.image_mlp(att_img_feats)

        prior_concat_summary = torch.cat([concat_prior_input, att_img_feats], dim=-1)
        prior_hidden = F.relu(self.prior_logit_hidden(prior_concat_summary))
        prior_hidden = self.dropout_layer(prior_hidden)
        prior_logit = self.prior_logit_layer(prior_hidden).squeeze()
        # concat_prior_input = torch.cat([question_embeddings, rationale_embedding], dim=-1)
        # prior_hidden = F.relu(self.prior_logit_hidden(concat_prior_input))
        # prior_logit = self.prior_logit_layer(prior_hidden).squeeze()

        batch_size = prior_logit.size(0)
        #sample prior z
        prior_prob = F.softmax(prior_logit, dim=-1)
        answer_probs = []
        m = Categorical(prior_prob)
        for i in range(config.vae_inference_sample_ct):
            prior_z_sample = m.sample()
            prior_z_sample = self.make_one_hot(prior_z_sample, 16)
            transpose_rationale_embedding = torch.transpose(rationale_embedding, 1, 2)
            expand_recog_discrete_z = prior_z_sample.unsqueeze(2)
            aggregate_rationale_embedding = torch.bmm(transpose_rationale_embedding, expand_recog_discrete_z).squeeze()


            expand_agg_ral_emb = aggregate_rationale_embedding.unsqueeze(1).expand(batch_size, 4, 512)
            qa_question_emb = self.evidence_embedding(qa_question, qa_question_tags, qa_question_mask, obj_reps)
            combined_answer_rationale_emb = torch.cat([qa_question_emb, expand_agg_ral_emb, answer_embedding], dim=-1)
            answer_hidden = F.relu(self.perform_logit_hidden(combined_answer_rationale_emb))
            answer_logit = self.perform_logit_layer(answer_hidden).squeeze()
            answer_class_probabilities = F.softmax(answer_logit, dim=-1)
            answer_probs.append(answer_class_probabilities)
        all_action_probs = torch.stack(answer_probs)
        average_action_probs = torch.mean(all_action_probs, dim=0)
        max_value, max_index = torch.max(average_action_probs,1)
        correct = (max_index == answer_label).sum()

        batch_size = answer_embedding.size(0)
        predict_answer_one_hot = torch.cuda.FloatTensor(batch_size, 4)
        predict_answer_one_hot.zero_()
        predict_answer_label = max_index.unsqueeze(1)
        predict_answer_one_hot.scatter_(1, predict_answer_label, 1)
        predict_answer_one_hot_expand = predict_answer_one_hot.unsqueeze(-1)
        transposed_answer_embedding = torch.transpose(answer_embedding, 1, 2)
        predict_answer_embedding = torch.bmm(transposed_answer_embedding, predict_answer_one_hot_expand).squeeze()

        #fifth step:
        #build recognition network: given gold answer, question and
        expand_gold_answer_embedding = predict_answer_embedding.unsqueeze(1).expand(batch_size, 16, 512)
        concat_recog_input = torch.cat([expand_gold_answer_embedding, question_embeddings, rationale_embedding], dim=-1)
        text_image_similarity = self.recog_summary_image_attention(
            concat_recog_input,
            trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        )
        v_att = F.softmax(text_image_similarity, dim=2)
        flat_img_feats = trans_img_fests.view(trans_img_fests.shape[0], trans_img_fests.shape[1]*trans_img_fests.shape[2], trans_img_fests.shape[3])
        att_img_feats = torch.einsum('bnm,bmd->bnd', (v_att, flat_img_feats))
        att_img_feats = self.image_mlp(att_img_feats)

        recog_concat_summary = torch.cat([concat_recog_input, att_img_feats], dim=-1)
        recog_hidden = F.relu(self.recog_logit_hidden(recog_concat_summary)).squeeze()
        recog_hidden = self.dropout_layer(recog_hidden)
        recog_logit = self.recog_logit_layer(recog_hidden).squeeze()

        # concat_recog_input = torch.cat([expand_gold_answer_embedding, question_embeddings, rationale_embedding], dim=-1)
        # recog_hidden = F.relu(self.recog_logit_hidden(concat_recog_input)).squeeze()
        # recog_logit = self.recog_logit_layer(recog_hidden).squeeze()
        predict_rationale_prob = F.softmax(recog_logit, dim=-1)
        _, predict_rationale_label = torch.max(predict_rationale_prob, 1)

        final_label = answer_label*16+rationale_label
        final_predict_label = max_index*16 + predict_rationale_label
        final_predict_logit = self.make_one_hot(final_predict_label, 64)
        self._accuracy(final_predict_logit, final_label)
        output_dict = {
            'pred_label':  final_predict_label
        }
        return output_dict



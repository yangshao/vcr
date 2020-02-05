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
from models.multiatt.Mutan import *
from models.multiatt.Mutan_OPT import *
from block import fusions

class Sentence_Attention(torch.nn.Module):
    def __init__(self, input_dim):
        super(Sentence_Attention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = 256
        self.hidden_layer1 = torch.nn.Linear(512, self.hidden_dim)
        self.hidden_layer2 = torch.nn.Linear(self.hidden_dim, 1)

    def forward(self, sentence, sentence_mask):
        hidden  = F.tanh(self.hidden_layer1(sentence))
        output = F.tanh(self.hidden_layer2(hidden)).squeeze()
        att = output.masked_fill(sentence_mask==0, -1e9)
        att = F.softmax(att, dim=-1)
        res = torch.bmm(att.unsqueeze(1), sentence)
        return res



@Model.register("CRF_Weighted")
class CRFQA_Weighted(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 seq_encoder: Seq2SeqEncoder,
                 text_seq_encoder: Seq2SeqEncoder,
                 span_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.5,
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
        super(CRFQA_Weighted, self).__init__(vocab)
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
        self.text_seq_encoder = text_seq_encoder
        self.span_encoder = TimeDistributed(span_encoder)
        if self.adaptive_flag:
            self.factor_layer = torch.nn.Linear(2048, 3)
            if self.wo_qa or self.wo_qr or self.wo_ar:
                self.partial_factor_layer = torch.nn.Linear(512*4, 2)
            # self.partial_factor_layer = torch.nn.Linear(512*4, 2)
        self.co_attention = DotProductMatrixAttention()
        # self.co_attention = BilinearMatrixAttention(
        #     matrix_1_dim=512,
        #     matrix_2_dim=512,
        # )
        # self.co_co_attention = BilinearMatrixAttention(
        #     matrix_1_dim=768,
        #     matrix_2_dim=768,
        # )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=1024,
            matrix_2_dim=512
        )
        # self.vqa_sentence_att = Sentence_Attention(512)
        # self.co_sentence_att = Sentence_Attention(512)
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

        self.vqa_logit_layer = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout),
            torch.nn.Linear(1024,1),
            )

        self.cnn_layer = torch.nn.Conv2d(1024, 1, 1)
        self.cnn_linear = torch.nn.Linear(24*48, 512)
        self.bbox_linear_layer = torch.nn.Linear(512, 512)


        self.image_mlp = torch.nn.Linear(1024, 512)

        self.text_linear = torch.nn.Linear(768*2, 768)

        self.input_mlp = torch.nn.Linear(1280, 512)

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(5*512, 1),
        )

        self.vqa_visual_mutan = fusions.Block(input_dims=[1024, 1536], output_dim=1024, chunks=18, dropout_input=0.1)

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
            qa_logits, qa_att, qa_weight_logit, qa_visual_f = self.get_vqa_logits(qa_question, qa_question_tags, qa_question_mask, qa_answers, qa_answer_tags, qa_answer_mask, obj_reps, box_mask)
        if not self.wo_qr:
            qr_logits, qr_att, qr_weight_logit, qr_visual_f = self.get_vqa_logits(qr_question, qr_question_tags, qr_question_mask, qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps, box_mask)
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
                       'qa_att': qa_att,
                       'qr_att': qr_att,
                       'ar_att': ar_att,
                       'inv_ar_att': inv_ar_att
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

    def get_vqa_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps, box_mask):
        img_feats = obj_reps['img_feats']
        # img_feats = obj_reps['new_2d_objs'][:,0,:,:]
        q_rep, q_obj_reps, qi_sim = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        a_rep, a_obj_reps, ai_sim = self.embed_span(qa_answer, qa_answer_tags, qa_answer_mask, obj_reps)

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_similarity = qa_similarity/np.sqrt(512)

        qa_attention_weights = masked_softmax(qa_similarity, qa_question_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))
        # attended_q = F.relu(self.text_linear(torch.cat([a_rep, attended_q], dim=-1)))+a_rep
        attended_q = torch.cat([a_rep, attended_q], dim=-1)

        inverse_qa_attention_weights = masked_softmax(torch.transpose(qa_similarity, -1, -2), qa_answer_mask[..., None], dim=2, double_flag = config.double_flag)
        attended_a = torch.einsum('bnaq,bnad->bnqd', (inverse_qa_attention_weights, a_rep))
        # attended_a = F.relu(self.text_linear(torch.cat([q_rep, attended_a], dim=-1)))+q_rep
        attended_a = torch.cat([q_rep, attended_a], dim=-1)

        batch_size, ct, sent_len, emb_size = attended_q.size()
        if self.rnn_input_dropout and self.training:
            attended_q = self.rnn_input_dropout(attended_q)
            attended_a = self.rnn_input_dropout(attended_a)
        self_attended_q = self.seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        new_emb_size = 512
        self_attended_q = self_attended_q.view(batch_size, ct, new_emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, new_emb_size)
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

            bs, ct, sz = text_summary.size()
            #attend on the bounding boxes
            bbox_similarity = self.obj_attention(text_summary,
                                                 obj_reps['obj_reps'])
            atoo_attention_weights = masked_softmax(bbox_similarity, box_mask.unsqueeze(1).expand_as(bbox_similarity))
            attended_o = torch.einsum('bno,bod->bnd', (atoo_attention_weights, obj_reps['obj_reps']))
            attended_o = F.relu(self.bbox_linear_layer(attended_o))

            v_rep = torch.cat([att_img_feats, attended_o], dim=-1)

            concat_summary = self.vqa_visual_mutan(text_summary.view(bs*ct, sz), v_rep.view(bs*ct, 1536))
            concat_summary = concat_summary.view(bs, ct, sz)

            logits = self.vqa_logit_layer(concat_summary).squeeze()
        else:
            concat_summary = F.relu(self.wo_im_vqa_hidden_layer(text_summary))
            logits = self.wo_im_vqa_logit_layer(concat_summary).squeeze()


        return logits, v_att, final_weight, att_img_feats
    def get_co_logits(self, qa_question, qa_question_tags, qa_question_mask, qa_answer, qa_answer_tags, qa_answer_mask, obj_reps):
        q_rep = qa_question['bert']
        a_rep = qa_answer['bert']

        qa_similarity = self.co_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])
        qa_similarity = qa_similarity/np.sqrt(768)

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
        self_attended_q = self.text_seq_encoder(attended_q.view(batch_size*ct, sent_len, emb_size), qa_answer_mask.view(batch_size*ct, sent_len))
        self_attended_q = get_final_encoder_states(self_attended_q, qa_answer_mask.view(batch_size*ct, sent_len), bidirectional=True)
        new_emb_size = 512
        self_attended_q = self_attended_q.view(batch_size, ct, new_emb_size)
        batch_size, ct, sent_len, emb_size = attended_a.size()
        self_attended_a = self.text_seq_encoder(attended_a.view(batch_size*ct, sent_len, emb_size), qa_question_mask.view(batch_size*ct, sent_len))
        self_attended_a = get_final_encoder_states(self_attended_a, qa_question_mask.view(batch_size*ct, sent_len), bidirectional=True)
        self_attended_a = self_attended_a.view(batch_size, ct, new_emb_size)

        final_rep = torch.cat([self_attended_q, self_attended_a, self_attended_q-self_attended_a, torch.abs(self_attended_q-self_attended_a), self_attended_q*self_attended_a], dim=-1)
        logits = self.final_mlp(final_rep).squeeze()
        return logits, None, qa_attention_weights, inverse_qa_attention_weights
    def get_factor_weights(self, obj_reps, qa_question, qa_question_tags, qa_question_mask,
                           qa_answers, qa_answer_tags, qa_answer_mask,
                           qr_rationales, qr_rationale_tags, qr_rationale_mask,
                           ):
        q_rep, _, _ = self.embed_span(qa_question, qa_question_tags, qa_question_mask, obj_reps)
        a_rep, _, _ = self.embed_span(qa_answers, qa_answer_tags, qa_answer_mask, obj_reps)
        r_rep, _, _ = self.embed_span(qr_rationales, qr_rationale_tags, qr_rationale_mask, obj_reps)
        bs, ct, sent_len, dim = q_rep.size()
        q_rep = self.span_encoder(q_rep, qa_question_mask)
        q_final = get_final_encoder_states(q_rep.view(bs*ct, sent_len, -1), qa_question_mask.view(bs*ct, sent_len), bidirectional=True)
        q_final = q_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = a_rep.size()
        a_rep = self.span_encoder(a_rep, qa_answer_mask)
        a_final = get_final_encoder_states(a_rep.view(bs*ct, sent_len, -1), qa_answer_mask.view(bs*ct, sent_len), bidirectional=True)
        a_final = a_final.view(bs, ct, -1)
        bs, ct, sent_len, _ = r_rep.size()
        r_rep = self.span_encoder(r_rep, qr_rationale_mask)
        r_final = get_final_encoder_states(r_rep.view(bs*ct, sent_len, -1), qr_rationale_mask.view(bs*ct, sent_len), bidirectional=True)
        r_final = r_final.view(bs, ct, -1)
        q_final =  torch.mean(q_final, dim=1, keepdim=True)
        a_ct = 4
        r_ct = 16
        q_final = q_final.expand(bs, a_ct*r_ct, dim)
        a_final = a_final.unsqueeze(2).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)
        r_final = r_final.unsqueeze(1).expand(bs, a_ct, r_ct, dim).contiguous().view(bs, a_ct*r_ct, dim)

        img_final = self.cnn_layer(obj_reps['img_feats'])
        img_final = img_final.squeeze().view(-1, 1152)
        img_final = F.relu(self.cnn_linear(img_final)).unsqueeze(1).expand_as(q_final)

        final_rep = torch.cat([img_final, q_final, a_final, r_final], dim=-1)

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


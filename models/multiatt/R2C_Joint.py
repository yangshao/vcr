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
from block import fusions

@Model.register("R2C_Joint")
class R2C_Joint(Model):
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
        super(R2C_Joint, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################

        self.emb_fusion = fusions.Block(input_dims=[1536, 1536], output_dim=1536, chunks=18, dropout_input=0.1)
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


        self.answer_final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        self.rationale_final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
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
                a_question: Dict[str, torch.Tensor],
                r_question: Dict[str, torch.Tensor],
                a_question_tags: torch.LongTensor,
                a_question_mask: torch.LongTensor,
                r_question_tags: torch.LongTensor,
                r_question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                rationales: Dict[str, torch.Tensor],
                rationale_tags: torch.LongTensor,
                rationale_mask: torch.LongTensor,
                ind: torch.LongTensor = None,
                label: torch.LongTensor = None,
                answer_label: torch.LongTensor = None,
                rationale_label: torch.LongTensor = None,
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
        segms = segms[:, :max_len]

        # for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
        #     if int(the_tags.max()) > max_len:
        #         raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
        #             tag_type, int(the_tags.max()), objects.shape, the_tags
        #         ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        # Now get the question representations
        a_q_rep, a_q_obj_reps = self.embed_span(a_question, a_question_tags, a_question_mask, obj_reps['obj_reps'])
        r_q_rep, r_q_obj_reps = self.embed_span(r_question, r_question_tags, r_question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])
        r_rep, r_obj_reps = self.embed_span(rationales, rationale_tags, rationale_mask, obj_reps['obj_reps'])

        ####################################
        # Perform Q by A attention
        # [batch_size, 4, question_length, answer_length]
        qa_similarity = self.span_attention(
            a_q_rep.view(a_q_rep.shape[0] * a_q_rep.shape[1], a_q_rep.shape[2], a_q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], a_q_rep.shape[2], a_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, a_question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, a_q_rep))

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

        a_pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]

        qr_similarity = self.span_attention(
            r_q_rep.view(r_q_rep.shape[0] * r_q_rep.shape[1], r_q_rep.shape[2], r_q_rep.shape[3]),
            r_rep.view(r_rep.shape[0] * r_rep.shape[1], r_rep.shape[2], r_rep.shape[3]),
        ).view(r_rep.shape[0], r_rep.shape[1], r_q_rep.shape[2], r_rep.shape[2])
        qr_attention_weights = masked_softmax(qr_similarity, r_question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qr_attention_weights, r_q_rep))

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        rtoo_similarity = self.obj_attention(r_rep.view(r_rep.shape[0], r_rep.shape[1] * r_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(r_rep.shape[0], r_rep.shape[1],
                                                                        r_rep.shape[2], obj_reps['obj_reps'].shape[1])
        rtoo_attention_weights = masked_softmax(rtoo_similarity, box_mask[:,None,None])
        attended_o = torch.einsum('bnao,bod->bnad', (rtoo_attention_weights, obj_reps['obj_reps']))


        reasoning_inp = torch.cat([x for x, to_pool in [(r_rep, self.reasoning_use_answer),
                                                        (attended_o, self.reasoning_use_obj),
                                                        (attended_q, self.reasoning_use_question)]
                                   if to_pool], -1)

        if self.rnn_input_dropout is not None and self.training:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)
        reasoning_output = self.reasoning_encoder(reasoning_inp, rationale_mask)


        ###########################################
        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (r_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)

        r_pooled_rep = replace_masked_values(things_to_pool,rationale_mask[...,None], -1e7).max(2)[0]

        answer_logits = self.answer_final_mlp(a_pooled_rep).squeeze(2)
        rationale_logits = self.rationale_final_mlp(r_pooled_rep).squeeze(2)

        a_bs, a_ct, a_dim = a_pooled_rep.size()
        a_pooled_rep = a_pooled_rep.unsqueeze(2).expand(a_bs, 4, 16, a_dim).contiguous().view(a_bs, 64, a_dim)
        r_bs, r_ct, r_dim = r_pooled_rep.size()
        r_pooled_rep = r_pooled_rep.unsqueeze(1).expand(r_bs, 4, 16, r_dim).contiguous().view(r_bs, 64, r_dim)

        # concat_summary = self.vqa_visual_mutan(text_summary.view(bs*ct, sz), v_rep.view(bs*ct, 1536))
        pooled_rep = self.emb_fusion(a_pooled_rep.view(-1, a_dim), r_pooled_rep.view(-1, r_dim)).view(a_bs, 64, a_dim)
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
            answer_loss = self._loss(answer_logits, answer_label.long().view(-1))
            rationale_loss = self._loss(rationale_logits, rationale_label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]
            output_dict['answer_loss'] = answer_loss[None]
            output_dict['rationale_loss'] = rationale_loss[None]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}


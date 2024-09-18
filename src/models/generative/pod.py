""" Code adapted from https://github.com/lileipisces/POD """

from transformers import (
    T5ForConditionalGeneration,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    HammingDiversityLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    BeamSearchScorer,
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutput
import torch.nn as nn
import torch
import random

from ..base import EncDecHFBaseModel
from src.utils.constants import Task, InputType, ModelType, SeqMode, PAD_TOK


class POD(EncDecHFBaseModel):
    TASKS = [Task.TOPN, Task.NEXT_ITEM, Task.EXPLANATION]
    SEQ_MODE = SeqMode.HIST_ITEM
    INPUT_TYPE = InputType.CUSTOM

    def __init__(self, data_info, cfg, **kwargs):
        super().__init__(data_info, cfg, **kwargs)

        self.beam_params = {
            Task.TOPN: {
                'num_beams': getattr(self.cfg, 'topn_num_beams', 20),
                'num_return_sequences': getattr(self.cfg, 'topn_top_n', 10)
            },
            Task.NEXT_ITEM: {
                'num_beams': getattr(self.cfg, 'nextitem_num_beams', 20),
                'num_return_sequences': getattr(self.cfg, 'nextitem_top_n', 10)
            },
            Task.EXPLANATION: {
                'num_beams': getattr(self.cfg, 'exp_num_beams', 21),
                'num_beam_groups': getattr(self.cfg, 'exp_num_beam_groups', 3),
                'min_length': getattr(self.cfg, 'exp_min_length', 10),
                'max_length': self.cfg.txt_len,
                'num_return_sequences': getattr(self.cfg, 'exp_top_n', 1)
            }
        }
        self.top_n = getattr(self.cfg, 'top_n', 10)
        self.fields = ['id', 'encoder', 'whole_word_ids', 'encoder_mask', 'decoder', 'decoder_mask']

        self.model = Solomon.from_pretrained('t5-small')
        self.model.init_prompt(len(self.TASKS), getattr(self.cfg, 'prompt_num', 3), self.cfg.device)

    def prepare_xy(self, batch, generation=False):
        inputs = {t: {f: batch[f'{t.value}_{f}'] for f in self.fields} for t in self.TASKS if f'{t.value}_id' in batch}
        labels = {t: batch[f'{t.value}_decoder'].detach().cpu().numpy() for t in self.TASKS if f'{t.value}_id' in batch}

        return inputs, labels

    def unpack_batch(self, batch):
        return [batch[f] for f in self.fields]

    def forward(self, batch):
        task_id, enc_input, enc_whole_ixs, enc_mask, dec_input, dec_mask = self.unpack_batch(batch)
        return self.model(task_id, enc_input, enc_whole_ixs, enc_mask, labels=dec_input)

    def _gen_random_items(self):
        selected_items = []
        for _ in range(self.top_n):
            item = self.data_info.special_items[PAD_TOK]
            while item in list(self.data_info.special_items.values()) + selected_items:
                item = random.randint(0, self.data_info.n_items - 1)
            selected_items.append(item)
        return selected_items

    def generate(self, batch):
        inputs, labels = self.prepare_xy(batch, generation=True)

        res = {}
        for t in self.TASKS:
            task_id, enc_input, enc_whole_ixs, enc_mask, _, _ = self.unpack_batch(inputs[t])
            beam_outputs = self.model.beam_search(task_id, enc_input, enc_whole_ixs, enc_mask, **self.beam_params[t])

            if t in [Task.TOPN, Task.NEXT_ITEM]:
                idss_predict = []
                output_tensor = beam_outputs.view(task_id.size(0), self.top_n, -1)
                for i in range(task_id.size(0)):
                    results = self.tok.batch_decode(output_tensor[i], skip_special_tokens=True)
                    try:
                        idss_predict.append(list(map(lambda s: int(s.split()[0]), results)))
                    except:
                        idss_predict.append(self._gen_random_items())
            else:
                idss_predict = beam_outputs.tolist()
            res[t] = idss_predict
        return res, labels


class Solomon(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def init_prompt(self, task_num, prompts_per_task, device):
        emsize = self.shared.weight.size(1)
        self.prompts_per_task = prompts_per_task
        self.model_device = device
        self.prompt_embeddings = nn.Embedding(task_num * prompts_per_task, emsize)
        self.whole_word_embeddings = nn.Embedding(self.config.n_positions, emsize)  # sequence length
        initrange = 0.1
        self.prompt_embeddings.weight.data.uniform_(-initrange, initrange)
        self.prompt_offset = torch.arange(prompts_per_task).to(self.model_device)

    def input_plus_whole_word(self, input_ids, whole_word_ids):
        text_emb = self.shared(input_ids)  # (batch_size, src_len, emsize)
        whole_word_emb = self.whole_word_embeddings(whole_word_ids)
        text_emb_plus = text_emb + whole_word_emb

        return text_emb_plus

    def append_prompt(self, task_id, input_ids, whole_word_ids, attention_mask):
        # prompt
        batch_size = task_id.size(0)
        task_ids = (task_id * self.prompts_per_task) + self.prompt_offset.repeat(batch_size, 1)  # (batch_size, prompts_per_task)
        prompt = self.prompt_embeddings(task_ids)  # (batch_size, prompts_per_task, input_size)

        # text
        text_emb_plus = self.input_plus_whole_word(input_ids, whole_word_ids)
        input_emb = torch.cat([prompt, text_emb_plus], 1)  # (batch_size, src_total_len, emsize)

        # mask
        prompt_pad = torch.ones((batch_size, self.prompts_per_task), dtype=torch.int64).to(self.model_device)
        input_mask = torch.cat([prompt_pad, attention_mask], 1)  # (batch_size, src_total_len)

        return input_emb, input_mask

    def forward(
        self,
        task_id=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if encoder_outputs is None:
            if task_id is None:
                input_emb = self.input_plus_whole_word(input_ids, whole_word_ids)
            else:
                input_emb, attention_mask = self.append_prompt(task_id, input_ids, whole_word_ids, attention_mask)
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                #input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=input_emb,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return super().forward(
            #input_ids=input_ids,
            #attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            #inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def beam_search(
        self,
        task_id=None,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        max_length=50,
        num_beams=20,
        num_beam_groups=1,
        early_stopping=True,
        min_length=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        num_return_sequences=20,
        bad_words_ids=None,
    ):
        # define decoder start token ids
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.ones((num_beams * batch_size, 1), dtype=torch.int64).to(self.model_device)
        decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        if task_id is None:
            input_emb = self.input_plus_whole_word(input_ids, whole_word_ids)
        else:
            input_emb, attention_mask = self.append_prompt(task_id, input_ids, whole_word_ids, attention_mask)
        model_kwargs = {
            "encoder_outputs": self.encoder(
                attention_mask=attention_mask.repeat_interleave(num_beams, dim=0),
                inputs_embeds=input_emb.repeat_interleave(num_beams, dim=0),
                return_dict=True,
            )
        }

        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=self.model_device,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences,
            do_early_stopping=early_stopping,
        )

        criteria = StoppingCriteriaList()
        criteria.append(MaxLengthCriteria(max_length=max_length))

        # instantiate logits processors
        logits_processor = LogitsProcessorList()
        logits_processor.append(MinLengthLogitsProcessor(min_length, eos_token_id=self.config.eos_token_id))
        if bad_words_ids is not None:
            logits_processor.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id=self.config.eos_token_id))

        if num_beam_groups == 1:
            return super().beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs)
        else:
            if diversity_penalty > 0.0:
                logits_processor.append(
                    HammingDiversityLogitsProcessor(
                        diversity_penalty,
                        num_beams=num_beams,
                        num_beam_groups=num_beam_groups,
                    )
                )
            if repetition_penalty != 1.0:
                logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(
                        penalty=repetition_penalty,
                    )
                )

            return super().group_beam_search(
                decoder_input_ids,
                beam_scorer,
                stopping_criteria=criteria,
                logits_processor=logits_processor,
                **model_kwargs)

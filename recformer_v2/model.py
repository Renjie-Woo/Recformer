import torch
import torch.nn as nn
from transformers.models.longformer.modeling_longformer import (
    LongformerPreTrainedModel,
    LongformerEncoder,
    LongformerBaseModelOutputWithPooling
)
from recformer_v2.config import RecformerConfig, DEFAULT_CONFIG
from recformer_v2.embedding import RecformerEmbedding
from recformer_v2.pooler import RecformerPooler
from typing import List, Union, Optional, Tuple
from torch.nn import CrossEntropyLoss

class RecformerModel(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig = DEFAULT_CONFIG):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = RecformerEmbedding(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = RecformerPooler()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            item_position_ids: torch.Tensor,
            position_ids: torch.Tensor,
            inputs_embeds: torch.Tensor,
            pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # this path should be recorded in the ONNX export, it is fine with padding_len == 0 as well
        if padding_len > 0:
            # logger.warning_once(
            #     f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
            #     f"`config.attention_window`: {attention_window}"
            # )
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if item_position_ids is not None:
                item_position_ids = nn.functional.pad(item_position_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, item_position_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    # @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @replace_return_docstrings(output_type=LongformerBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            item_position_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerBaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, item_position_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            item_position_ids=item_position_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
                                                :, 0, 0, :
                                                ]

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, item_position_ids=item_position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LongformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
        )


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config: RecformerConfig):
        super().__init__()
        self.temp = config.temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



class RecformerForSeqRec(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)

        self.longformer = RecformerModel(config)
        self.sim = Similarity(config)
        # Initialize weights and apply final processing
        self.post_init()

    def init_item_embedding(self, embeddings: Optional[torch.Tensor] = None):
        self.item_embedding = nn.Embedding(num_embeddings=self.config.item_num, embedding_dim=self.config.hidden_size)
        if embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
            print('Initalize item embeddings from vectors.')

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is None:
            candidate_embeddings = self.item_embedding.weight.unsqueeze(0) # (1, num_items, hidden_size)
        else:
            candidate_embeddings = self.item_embedding(candidates) # (batch_size, candidates, hidden_size)
        pooler_output = pooler_output.unsqueeze(1) # (batch_size, 1, hidden_size)
        return self.sim(pooler_output, candidate_embeddings)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                global_attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                item_position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                candidates: Optional[torch.Tensor] = None, # candidate item ids
                labels: Optional[torch.Tensor] = None, # target item ids
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooler_output = outputs.pooler_output # (bs, hidden_size)

        if labels is None:
            return self.similarity_score(pooler_output, candidates)

        loss_fct = CrossEntropyLoss()

        if self.config.finetune_negative_sample_size<=0: ## using full softmax
            logits = self.similarity_score(pooler_output)
            loss = loss_fct(logits, labels)

        else:  ## using sampled softmax
            candidates = torch.cat((labels.unsqueeze(-1), torch.randint(0, self.config.item_num, size=(batch_size, self.config.finetune_negative_sample_size)).to(labels.device)), dim=-1)
            logits = self.similarity_score(pooler_output, candidates)
            target = torch.zeros_like(labels, device=labels.device)
            loss = loss_fct(logits, target)

        return loss
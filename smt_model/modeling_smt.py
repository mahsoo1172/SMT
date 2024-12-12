import torch
from torch.distributions import continuous_bernoulli
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_
from transformers import ConvNextConfig, ConvNextModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import time

from .configuration_smt import SMTConfig


class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max),
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                              requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start + x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i] + x.size(2)]
            return x


class MHA(nn.Module):
    def __init__(self, embedding_dim, num_heads=None, dropout=0, proj_value=True) -> None:
        super().__init__()

        self.proj_value = proj_value
        self.lq = nn.Linear(embedding_dim, embedding_dim)
        self.lk = nn.Linear(embedding_dim, embedding_dim)
        if proj_value:
            self.lv = nn.Linear(embedding_dim, embedding_dim)

        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, key_pad_mask=None, attn_mask=None, get_weights=True):

        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value) if self.proj_value else value

        q = torch.reshape(q, (target_len, b * self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b * self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b * self.num_heads, self.head_dim)).transpose(0, 1)

        attn_output_weigths = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_output_weigths.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weigths += attn_mask

        if key_pad_mask is not None:
            attn_output_weigths = attn_output_weigths.view(b, self.num_heads, target_len, source_len)
            attn_output_weigths = attn_output_weigths.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_output_weigths = attn_output_weigths.view(b * self.num_heads, target_len, source_len)

        attn_output_weigths_raw = self.softmax(attn_output_weigths)
        attn_output_weigths = self.dropout(attn_output_weigths_raw)
        attn_output = torch.bmm(attn_output_weigths, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if get_weights:
            attn_output_weigths_raw = attn_output_weigths_raw.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_output_weigths_raw.sum(dim=1) / self.num_heads

        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, dim_ff) -> None:
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.ff = dim_ff

        self.input_attention = MHA(embedding_dim=self.d_model,
                                   num_heads=4,
                                   proj_value=True,
                                   dropout=0.1)

        self.norm1 = nn.LayerNorm(self.d_model)

        self.cross_attention = MHA(embedding_dim=self.d_model,
                                   num_heads=4,
                                   proj_value=True,
                                   dropout=0.1)

        self.ffNet = nn.Sequential(
            nn.Linear(self.d_model, self.ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.ff, self.d_model)
        )

        self.dropout = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                predict_n_last_only=None):
        if memory_value is None:
            memory_value = memory_key

        mha_q = tgt[-predict_n_last_only:] if predict_n_last_only else tgt

        tgt2, weights_input = self.input_attention(mha_q, tgt, tgt, attn_mask=tgt_mask,
                                                   key_pad_mask=tgt_key_padding_mask, get_weights=True)
        tgt = mha_q + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        att_query = tgt

        tgt2, weights_cross = self.cross_attention(att_query, memory_key, memory_value, attn_mask=memory_mask,
                                                   key_pad_mask=memory_key_padding_mask, get_weights=True)

        tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffNet(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, weights_input, weights_cross


class DecoderStack(nn.Module):

    def __init__(self, num_dec_layers, d_model, dim_ff) -> None:
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, dim_ff=dim_ff) for _ in range(num_dec_layers)])

    def set_lm_mode(self):
        for layer in self.layers:
            layer.set_lm_mode()

    def set_transcription_mode(self):
        for layer in self.layers:
            layer.set_transcription_mode()

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask,
                memory_key_padding_mask, use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=True):

        output = tgt
        cache_t = list()
        all_weights = {
            "self": list(),
            "mix": list()
        }

        for i, dec_layer in enumerate(self.layers):
            output, weights_self, weights_cross = dec_layer(output, memory_key=memory_key,
                                                            memory_value=memory_value,
                                                            tgt_mask=tgt_mask,
                                                            memory_mask=memory_mask,
                                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                                            memory_key_padding_mask=memory_key_padding_mask,
                                                            predict_n_last_only=predict_last_n_only)

            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)

            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights_cross)

        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(
                cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights_cross, cache


class Decoder(nn.Module):
    def __init__(self, d_model, dim_ff, attn_heads, n_layers, maxlen, out_categories, attention_window=100) -> None:
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.dec_attn_win = attention_window
        self.positional_1D = PositionalEncoding1D(d_model, maxlen)

        self.decoder = DecoderStack(num_dec_layers=n_layers, d_model=d_model, dim_ff=dim_ff)

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        self.end_relu = nn.ReLU()

        self.out_layer = nn.Conv1d(d_model, out_categories, kernel_size=1)

    def set_lm_mode(self):
        self.decoder.set_lm_mode()

    def set_transcription_mode(self):
        self.decoder.set_transcription_mode()

    def forward(self, raw_features_1D, enhanced_features_1D, tokens,
                reduced_size, token_len, features_size, hidden_predict=None, num_pred=None, cache=None,
                keep_all_weights=True):

        device = raw_features_1D.device

        pos_tokens = self.embedding(tokens).permute(0, 2, 1)

        pos_tokens = self.positional_1D(pos_tokens, start=0)
        pos_tokens = pos_tokens.permute(2, 0, 1).contiguous()

        if num_pred is None:
            num_pred = tokens.size(1)

        if self.dec_attn_win > 1 and cache is not None:
            cache = cache[:, -self.dec_attn_win - 1]
        else:
            cache = None

        num_tokens_to_keep = num_pred if self.dec_attn_win is None else min(
            [num_pred + self.dec_attn_win - 1, pos_tokens.size(0), token_len[0]])
        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        target_mask = self.generate_target_mask(tokens.size(1), device=device)
        memory_mask = None

        key_target_mask = self.generate_token_mask(token_len, tokens.size(), device)
        key_memory_mask = None

        target_mask = target_mask[-num_pred:, -num_tokens_to_keep:]
        key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, weights, cache = self.decoder(pos_tokens, memory_key=enhanced_features_1D, memory_value=raw_features_1D,
                                              tgt_mask=target_mask, memory_mask=memory_mask,
                                              tgt_key_padding_mask=key_target_mask,
                                              memory_key_padding_mask=key_memory_mask, use_cache=True, cache=cache,
                                              predict_last_n_only=num_pred, keep_all_weights=keep_all_weights)

        dpoutput = self.dropout(self.end_relu(output))

        predictions = self.out_layer(dpoutput.permute(1, 2, 0).contiguous())

        if not keep_all_weights:
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3])

        return output, predictions, hidden_predict, cache, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = False

        return mask

    def generate_target_mask(self, target_len, device):
        if self.dec_attn_win == 1:
            return torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=1)
        else:
            return torch.logical_not(
                torch.logical_and(
                    torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                    torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device),
                               diagonal=-self.dec_attn_win + 1)))


class SMTOutput(CausalLMOutputWithCrossAttentions):
    """This is a nice output wrapper"""


class SMTModelForCausalLM(PreTrainedModel):
    config_class = SMTConfig

    def __init__(self, config: SMTConfig):
        super().__init__(config)
        # self.encoder = ConvNextEncoder(config.in_channels, stem_features=64, depths=[4,6], widths=[128, 256])
        next_config = ConvNextConfig(num_channels=config.in_channels, num_stages=3, hidden_sizes=[32, 64, 128],
                                     depths=[3, 3, 9])
        self.encoder = ConvNextModel(next_config)
        self.decoder = Decoder(d_model=config.d_model, dim_ff=config.dim_ff, attn_heads=config.num_attn_heads,
                               n_layers=config.num_dec_layers,
                               maxlen=config.maxlen, out_categories=config.out_categories,
                               attention_window=config.maxlen + 1)

        self.positional_2D = PositionalEncoding2D(config.d_model, config.maxh, config.maxw)

        self.padding_token = config.padding_token
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token)  # , reduction='none')

        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)

        # Creating vocab mask:
        # print(self.i2w)
        print('vocab mask creation started')
        note_vals_dict = {
            'C': -7, 'D': -6, 'E': -5, 'F': -4, 'G': -3, 'A': -2, 'B': -1,
            'c': 0,
            'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6
        }
        # print(self.i2w)
        # print(self.i2w.values())
        self.vocab_note_pos_mask = []
        sorted_keys = sorted(self.i2w.keys(), key=int)
        i = 0
        for char_key in sorted_keys:
            note_token = self.i2w[char_key]
            # print(char_key, note_token)

            # filtering only to valid note names
            valid_characters = ['c', 'd', 'e', 'f', 'g', 'a', 'b']

            # remove the extra characters, we only what the notes are - i.e. ccc, BBBB, gg
            filtered_note_token = ''.join([char for char in note_token if char.lower() in valid_characters])

            # Special cases - eos, ekern, etc are examples of non-note tokens
            if ((filtered_note_token == '') or ('ekern' in note_token) or ('eos' in note_token)
                    or ('<' in note_token) or ('>' in note_token) or ('*' in note_token)
                    or ('staff' in note_token) or ('met' in note_token) or ('clef' in note_token)
            ):
                self.vocab_note_pos_mask.append(-999)
                # print(self.vocab_note_pos_mask)
                continue

            # Empty string in position 1 of the vocab. Also non-note token
            if len(note_token) == 0:
                self.vocab_note_pos_mask.append(-999)
                # print(self.vocab_note_pos_mask)
                continue

            note = filtered_note_token[0]
            n = len(filtered_note_token)
            if note_vals_dict[note] < 0:
                note_position = note_vals_dict[note] - (n - 1) * 7
            else:
                note_position = note_vals_dict[note] + (n - 1) * 7

            self.vocab_note_pos_mask.append(note_position)

        # for i in range(len(self.i2w.values())):
        #     print(i, 'token:', list(self.i2w.values())[i], 'value', self.vocab_note_pos_mask[i])
        # with open("vocab_info.txt", "w") as f:  # Open file in write mode
        #     for i in range(len(self.i2w.values())):
        #         line = f"{i} token: {self.i2w[i]} value: {self.vocab_note_pos_mask[i]}\n"
        #         f.write(line)  # Write the line to the file

        # with open('vocab_key_value_info.txt', "w") as f:
        #   for key, value in self.i2w.items():
        #       line = f"Key: {key}, Value: {value}\n"
        #       f.write(line)
        print('vocab mask made')

        self.vocab_tensor_mask = torch.tensor(self.vocab_note_pos_mask, dtype=torch.float32,
                                              requires_grad=True,
                                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                              ).unsqueeze(0).unsqueeze(2)

    def forward_encoder(self, x):
        return self.encoder(pixel_values=x).last_hidden_state

    def forward_decoder(self, encoder_output, y_pred):
        b, _, _, _ = encoder_output.size()
        reduced_size = [s.shape[:2] for s in encoder_output]
        ylens = [len(sample) for sample in y_pred]

        pos_features = self.positional_2D(encoder_output)
        features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(2, 0, 1)
        enhanced_features = features
        enhanced_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
        output, predictions, _, _, weights = self.decoder(features, enhanced_features, y_pred[:, :], reduced_size,
                                                          [max(ylens) for _ in range(b)], encoder_output.size(),
                                                          cache=None, keep_all_weights=True)
        return SMTOutput(

            logits=predictions,
            hidden_states=output,
            attentions=weights["self"],
            cross_attentions=weights["mix"]
        )

    # Note Loss - First Implementation
    def note_position(self, idx, i2w_dict):
        note_token = i2w_dict[idx]
        # print('token', note_token)
        note_vals_dict = {
            'C': -7, 'D': -6, 'E': -5, 'F': -4, 'G': -3, 'A': -2, 'B': -1,
            'c': 0,
            'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6
        }
        valid_characters = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        filtered_note_token = ''.join([char for char in note_token if char.lower() in valid_characters])
        if ((filtered_note_token == '') or ('ekern' in note_token) or ('eos' in note_token)
                or ('<' in note_token) or ('>' in note_token) or ('*' in note_token)):
            # print(-999)
            return -999
        note = filtered_note_token[0]
        n = len(filtered_note_token)
        if note_vals_dict[note] < 0:
            note_position = note_vals_dict[note] - (n - 1) * 7
        else:
            note_position = note_vals_dict[note] + (n - 1) * 7

        return note_position

    def forward(self, x, y_pred, labels=None):
        # print('start forward')
        start_time = time.time()
        x = self.forward_encoder(x)
        output = self.forward_decoder(x, y_pred)

        if labels is not None:
            # output.loss = self.loss(output.logits, labels[:, :-1])

            # for local debugging, send to cpu
            # output.logits= output.logits.to('cpu') # line for debugging
            # labels= labels.to('cpu') # line for debugging

            # Note Loss - Second Implementation:

            logits_shape = output.logits.shape  # Get batch size shape, [batch_size x 20578 x seq_length]
            # print('logit shape', logits_shape)

            # Initialize the vocab note position tensor from the self.vocab_note_pos_mask list
            # vocab_tensor_mask shape is [1 x 20578 x 1]

            # vocab_tensor_mask = torch.tensor(self.vocab_note_pos_mask, dtype=torch.float32,
            #                                  requires_grad=True).unsqueeze(0).unsqueeze(2)

            # Use torch.repeat to make the vocab_tensor_mask match the logits shape
            # batch tensor mask shape is [batch_size x 20578 x seq_length]
            cloned_vocab_tensor_mask = self.vocab_tensor_mask.clone()
            batch_vocab_tensor_mask = cloned_vocab_tensor_mask.repeat(logits_shape[0], 1, logits_shape[2])
            # print('batch_vocab_tensor_mask shape', batch_vocab_tensor_mask.shape)

            # DIFFERENTIABLE DISCREPENCY WITH ARGMAX
            # torch.argmax does not flow gradients backwards to output.logits
            logits_argmax = torch.argmax(output.logits, dim=1)
            # Use gumbel softmax instead!!!! Gumbel softmax samples from a softmax distribution.
            logits_gumbel = torch.nn.functional.gumbel_softmax(logits=output.logits, dim=1, hard=True)

            # Create equivalent 'prediction positions' with gumbel_prediction
            gumbel_prediction_pos_mask = batch_vocab_tensor_mask * logits_gumbel
            gumbel_prediction_pos = torch.sum(gumbel_prediction_pos_mask, dim=1)

            # prediction_pos = torch.gather(batch_vocab_tensor_mask, dim=1, index=logits_argmax.unsqueeze(1))
            # prediction_pos = prediction_pos.squeeze(1)
            # print('prediction pos', prediction_pos.shape)

            # print('spot check')
            # print('vocabtens', batch_vocab_tensor_mask[0,:,:])
            # print('lg',logits_argmax[0,0:90])
            # print('prediction pos v2',prediction_pos[0,0:90])

            labels_pos = torch.gather(batch_vocab_tensor_mask, dim=1, index=labels[:, :-1].unsqueeze(1))
            # print('all before', labels_pos)
            # print('labels_pos before', labels_pos[0,0,0:90])
            # print('labels_pos shape before', labels_pos.shape)
            labels_pos = labels_pos.squeeze(1)
            # print('labels_pos after', labels_pos[0,0:90])
            # print('labels_pos_shape_after', labels_pos.shape)

            # print('all after', labels_pos)

            # when ground truth is non-note token, we want 'distance' loss to be equal to 0.
            # create mask for elements = 0 when non-note
            gt_mask = (labels_pos == -999)

            # when prediction is non-note token AND gt is a note token, we need to set distance to max (1).

            prediction_mask = (gumbel_prediction_pos == -999)

            # Raw delta without mask checking

            # normalize note distance by the maximum possible distance of 49 for the notes in the vocab.
            # (Checked by doing max - min distance in vocab). Spot checking the vocab, went from lowest: AAAA
            # to highest: aaaa

            max_note_distance = 49

            ### NOTE LOSS THAT FLOWS GRADIENTS BACK IS GUMBEL_PREDICTION_POS####
            ####################################################################
            # labels_pos = labels_pos* 100 # debugging line
            # delta = torch.abs((prediction_pos - labels_pos))
            # print('gumbel pred pos', gumbel_prediction_pos)
            # print('labels pos', labels_pos)
            # print('labels id', labels[0,0:90])
            # token_list = [self.i2w[int(i.item())] for i in labels[0,0:90]]
            # print(token_list)

            delta = torch.abs((gumbel_prediction_pos - labels_pos))
            ####################################################################
            # print('delta no div', delta)
            delta = delta / max_note_distance
            # print('delta no mask', delta)

            # when gt is a non-note token, set delta value to 0.
            # There is no note loss b/c the gt token is not a note.
            delta[gt_mask] = 0
            # print('delta 1st mask', delta)

            # when gt IS a note token AND prediction is not a note token, set loss to max.
            # Prediction is not even a note, set loss to max.
            delta[(~gt_mask & prediction_mask)] = 1

            for i in range(delta.shape[0]):
                row = delta[i]
                non_zero_row = row[row != 0]
                # print(f"Non-zero elements in row {i}: {non_zero_row}")
                # print(len(non_zero_row))
                # print('delta row[i]', delta[i][0:90])
                # print('gumbel pos [i]', gumbel_prediction_pos[i][0:90])
                # print('labels pos[i]', labels_pos[i][0:90])
            # print('deltas', delta)
            # max_len = output.logits.shape[2]
            delta_sum = torch.sum(delta, dim=1)
            # print('delta sum', delta_sum)
            # Normalization

            # # Find the number of musical note tokens per sequence in each row
            num_musical_note_tokens = (labels_pos != -999).sum(dim=1)

            # print('num musical tokens', num_musical_note_tokens)
            # print(num_musical_note_tokens)
            # If a row is NO musical note tokens, set its length to 1 to avoid issues
            num_musical_note_tokens[num_musical_note_tokens == 0] = 1

            # scale to number of musical tokens
            delta_sum_scaled_to_num_musical_tokens = (delta_sum / num_musical_note_tokens)

            # scale to number of samples in batch
            N = labels.shape[0]
            # print('N', N)
            note_loss = torch.sum(delta_sum_scaled_to_num_musical_tokens) / N

            output.loss = self.loss(output.logits, labels[:, :-1]) + note_loss
            # print('note loss', note_loss)
            # print('total loss', output.loss)
            # print('end forward', time.time() - start_time)
            return output, note_loss

            ############### FIRST ATTEMPT #####################################
            # if True:
            #   # Note Loss - First Implementation:
            #   logits_argmax = torch.argmax(output.logits,dim=1)

            #   prediction_pos= logits_argmax.cpu().apply_(lambda x: self.note_position(x, self.i2w))
            #   labels_pos = labels[:, :-1].cpu().apply_(lambda x: self.note_position(x, self.i2w))
            #   # print('prediction pos v1', prediction_posv1[0,0:90])
            #   # print('labels_pos v1', labels_posv1[0,0:90])
            #   # when ground truth is non-note token, we want 'distance' loss to be equal to 0.
            #   # create mask for elements = 0 when non-note
            #   gt_mask = (labels_pos == -999)

            #   # when prediction is non-note token AND gt is a note token, we need to set distance to max (1).

            #   prediction_mask = (prediction_pos == -999)

            #   # raw delta without mask checking
            #   max_note_distance = 49
            #   delta = torch.abs((prediction_pos - labels_pos))
            #   # print('delta no div', delta)
            #   delta = delta / max_note_distance
            #   # print('delta no mask', delta)
            #   # when gt is a non-note token, set delta value to 0.
            #   # There is no note loss b/c the gt token is not a note.
            #   delta[gt_mask] = 0
            #   # print('delta 1st mask', delta)

            #   # when gt IS a note token AND prediction is not a token, set loss to max.
            #   delta[(~gt_mask & prediction_mask)] = 1
            #   # max_len = output.logits.shape[2]
            #   delta_sum = torch.sum(delta, dim=1)
            #   print(delta_sum)
            #   # # Find the indices of the last non-zero elements in each row
            #   last_nonzero_indices = (labels[:, :-1] != 0).sum(dim=1)
            #   print(last_nonzero_indices)
            #   # If a row is all zeros, set its length to 0 to avoid issues
            #   last_nonzero_indices[last_nonzero_indices == 0] = 1
            #   # print('seq length without padding', last_nonzero_indices)
            #   N = labels.shape[0]
            #   delta_sum_scaled_to_length = (delta_sum / last_nonzero_indices.cpu())
            #   note_loss = torch.sum(delta_sum_scaled_to_length) / N
            #   # print('note_loss', note_loss)
            #   print('note loss v1', note_loss)
            #   # print('normal loss',output.loss)
            #   # print('normal loss plus note loss', output.loss + note_loss)

            # #   output.loss = output.loss + note_loss
            # #   return output, note_loss

        return output

    def predict(self, input, convert_to_str=False, y=None):
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(input.device).unsqueeze(0)
        encoder_output = self.forward_encoder(input)
        text_sequence = []
        for i in range(self.maxlen - predicted_sequence.shape[-1]):
            predictions = self.forward_decoder(encoder_output, predicted_sequence.long())
            predicted_token = torch.argmax(predictions.logits[:, :, -1]).item()
            predicted_sequence = torch.cat(
                [predicted_sequence, torch.argmax(predictions.logits[:, :, -1], dim=1, keepdim=True)], dim=1)
            if convert_to_str:
                predicted_token = f"{predicted_token}"
            if self.i2w[predicted_token] == '<eos>':
                break
            text_sequence.append(self.i2w[predicted_token])

        if y is not None:
            predictions.loss = self.loss(predictions.logits, y)

        return text_sequence, predictions

    # TODO: fix the  vectorized predict
    # def predict(self, input, convert_to_str=False):
    #     batch_size = input.shape[0]
    #     predicted_sequence = torch.full((batch_size, 1), self.w2i['<bos>'], device=input.device, dtype=torch.long)
    #     encoder_output = self.forward_encoder(input)
    #     active_samples = torch.arange(batch_size, device=input.device)  # Keep track of active samples
    #     text_sequences = [[] for _ in range(batch_size)]
    #     print('self maxlen', self.maxlen)
    #     for i in range(self.maxlen - 1):
    #         predictions = self.forward_decoder(encoder_output[active_samples], predicted_sequence[active_samples])
    #         predicted_tokens = torch.argmax(predictions.logits[:, :, -1],
    #                                         dim=1)  # Get predicted tokens for active samples
    #         # print(f"Shape of predicted_sequence[active_samples]: {predicted_sequence[active_samples].shape}")
    #         # print(f"Shape of predicted_tokens: {predicted_tokens.shape}")
    #         # print(f"Shape of predicted_tokens.unsqueeze(1): {predicted_tokens.unsqueeze(1).shape}")
    #         predicted_sequence = torch.cat([predicted_sequence[active_samples], predicted_tokens.unsqueeze(1)], dim=1)
    #
    #         # Check for <eos> token and update active samples
    #         eos_indices = (predicted_tokens == self.w2i['<eos>']).nonzero()
    #         # if eos_indices.nelement() > 0:
    #         #     print(eos_indices)
    #         #     print('eos indices nelement>0', (eos_indices.nelement() > 0))
    #         if eos_indices.nelement() > 0:
    #             #TODO: Active samples logic causes runtime error. Ignoring the time savings for removing individual batch samples from
    #             # processing when <eos> is reached.
    #             # print('active samples before', active_samples)
    #             # active_samples = active_samples[~torch.isin(torch.arange(active_samples.size(0), device=input.device), eos_indices.squeeze())]
    #             # print('active samples after', active_samples)
    #
    #             # This should still work. If ALL batch samples are done (no more empty samples), then should be okay to end early.
    #             if active_samples.nelement() == 0:  # All samples have finished
    #                 break
    #
    #         # Update text sequences for active samples
    #         for j, sample_idx in enumerate(active_samples):
    #             predicted_token = predicted_tokens[j].item()
    #             if convert_to_str:
    #                 predicted_token = f"{predicted_token}"
    #             text_sequences[sample_idx.item()].append(self.i2w[predicted_token])
    #
    #     return text_sequences, predictions
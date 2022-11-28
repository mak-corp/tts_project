from hw_tts.base import BaseModel
import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from dacite import from_dict
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [ (batch_size * n_heads) x seq_len x hidden_size ]
        
        attn = torch.bmm(q, k.transpose(-1, -2))
        attn /= self.temperature
        
        # attn: [ (batch_size * n_heads) x seq_len x seq_len ]
        
        if mask is not None:
            attn = torch.masked_fill(attn, mask, -torch.inf)
        
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        # output: [ (batch_size * n_heads) x seq_len x hidden_size ]
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5) 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
         # normal distribution initialization better than kaiming(default in pytorch)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v))) 
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # TODO change layer_norm

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 fft_conv1d_kernel, 
                 fft_conv1d_padding,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


def get_non_pad_mask(seq, pad):
    assert seq.dim() == 2
    return seq.ne(pad).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, pad):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        self.model_config = model_config
        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.fft_conv1d_kernel, 
            model_config.fft_conv1d_padding,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad=self.model_config.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, pad=self.model_config.PAD)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        self.model_config = model_config
        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.fft_conv1d_kernel, 
            model_config.fft_conv1d_padding,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad=self.model_config.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, pad=self.model_config.PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VariancePredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
        out = self.linear_layer(encoder_output)
        out = out.squeeze(-1)
        return out


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.max_seq_len = model_config.max_seq_len

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration = torch.clamp(torch.exp(duration_predictor_output), min=0, max=100)
            duration = ((torch.exp(duration) + 0.5) * alpha).long()
            duration = duration * (torch.cumsum(duration, dim=-1) <= self.max_seq_len)

            if duration.sum() < 10:
                duration[0][0] = duration[0][0] + 10

            output = self.LR(x, duration)
            
            mel_pos = torch.arange(1, output.size(1) + 1, dtype=torch.long, device=x.device).unsqueeze(0)
            return output, mel_pos


class QuantizationEmbedding(nn.Module):
    def __init__(self, n_bins, min_val, max_val, hidden, mode="linspace"):
        super().__init__()

        if mode == "linspace":
            boundaries = torch.linspace(min_val, max_val, n_bins + 1)[1:-1]
        elif mode == "logspace":
            boundaries = torch.logspace(min_val, max_val, n_bins + 1)[1:-1]
        else:
            raise ValueError()

        self.boundaries = nn.Parameter(boundaries, requires_grad=False)
        self.embedding = nn.Embedding(n_bins, hidden)

    def forward(self, x):
        return self.embedding(torch.bucketize(x, self.boundaries))


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config

        self.length_regulator = LengthRegulator(model_config)

        self.energy_predictor = VariancePredictor(model_config)
        self.energy_emb = QuantizationEmbedding(
            model_config.n_bins_energy,
            model_config.energy_min,
            model_config.energy_max,
            model_config.encoder_dim,
            mode="linspace",
        )

        self.pitch_predictor = VariancePredictor(model_config)
        self.pitch_emb = QuantizationEmbedding(
            model_config.n_bins_pitch,
            model_config.pitch_min,
            model_config.pitch_max,
            model_config.encoder_dim,
            mode="logspace",
        )

    def forward(
        self,
        x,
        alpha=1.0,
        energy_alpha=1.0,
        pitch_alpha=1.0,
        duration_target=None,
        max_len=None,
        energy_target=None,
        pitch_target=None):

        if duration_target is not None:
            x, duration_prediction = self.length_regulator(x, alpha, duration_target, max_len)

            energy_prediction = self.energy_predictor(x) * energy_alpha
            pitch_prediction = self.pitch_predictor(x) * pitch_alpha

            x = x + self.energy_emb(energy_target) + self.pitch_emb(pitch_target)

            return x, duration_prediction, energy_prediction, pitch_prediction
        else:
            x, mel_pos = self.length_regulator(x, alpha)

            energy_prediction = self.energy_predictor(x) * energy_alpha
            pitch_prediction = self.pitch_predictor(x) * pitch_alpha
            x = x + self.energy_emb(energy_prediction) + self.pitch_emb(pitch_prediction)

            return x, mel_pos


@dataclass
class FastSpeechConfig:
    num_mels: int = 80
    
    vocab_size: int = 300
    max_seq_len: int = 3000

    encoder_dim: int = 256
    encoder_n_layer: int = 4
    encoder_head: int = 2
    encoder_conv1d_filter_size: int = 1024

    n_bins_energy: int = 256
    energy_min: float = -1.136
    energy_max: float = 15.167

    n_bins_pitch: int = 256
    pitch_min: float = -1.186
    pitch_max: float = 6.822

    decoder_dim: int = 256
    decoder_n_layer: int = 4
    decoder_head: int = 2
    decoder_conv1d_filter_size: int = 1024

    fft_conv1d_kernel: list = (9, 1)
    fft_conv1d_padding: list = (4, 0)

    duration_predictor_filter_size: int = 256
    duration_predictor_kernel_size: int = 3
    dropout: float = 0.1
    
    PAD: int = 0
    UNK: int = 1
    BOS: int = 2
    EOS: int = 3

    PAD_WORD: str = '<blank>'
    UNK_WORD: str = '<unk>'
    BOS_WORD: str = '<s>'
    EOS_WORD: str = '</s>'


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2(BaseModel):
    """ FastSpeech """

    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()

        self.model_config = from_dict(data_class=FastSpeechConfig, data=model_config)

        self.encoder = Encoder(self.model_config)
        self.variance_adaptor = VarianceAdaptor(self.model_config)
        self.decoder = Decoder(self.model_config)

        self.mel_linear = nn.Linear(self.model_config.decoder_dim, self.model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, text, src_pos, mel_pos=None, mel_max_length=None, duration=None, energy=None, pitch=None,
                alpha=1.0, energy_alpha=1.0, pitch_alpha=1.0, **kwargs):

        x, non_pad_mask = self.encoder(text, src_pos)

        if self.training:
            output, log_duration_prediction, energy_prediction, pitch_prediction = self.variance_adaptor(
                x, alpha, energy_alpha, pitch_alpha, duration, mel_max_length, energy, pitch
            )
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {
                "mel_output": output,
                "log_duration_prediction": log_duration_prediction,
                "energy_prediction": energy_prediction,
                "pitch_prediction": pitch_prediction,
            }

        else:
            output, mel_pos = self.variance_adaptor(x, alpha, energy_alpha, pitch_alpha)
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output

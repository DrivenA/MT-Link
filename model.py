import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

def sequence_mask(X: torch.Tensor, valid_len: torch.Tensor, value: float = 0) -> torch.Tensor:
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def GlobalAveragePooling(emb: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    mask = sequence_mask(emb.new_ones(emb.size()[:2]), valid_len).unsqueeze(-1)
    emb = emb * mask  
    sum_emb = emb.sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1)  
    avg_emb = sum_emb / count
    return avg_emb

class BinaryClassifier(nn.Module):
    def __init__(self, embed_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(embed_size * 2, 128)  
        self.fc2 = nn.Linear(128, 1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, output_1, output_2, valid_len_1, valid_len_2):
        pooled_output_1 = GlobalAveragePooling(output_1, valid_len_1)
        pooled_output_2 = GlobalAveragePooling(output_2, valid_len_2)
        combined_output = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        x = torch.relu(self.fc1(combined_output))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.w = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div = math.sqrt(1. / embed_size)

    def forward(self, x, **kwargs):
        timestamp = kwargs['time_seq']  
        time_encode = torch.cos(timestamp.unsqueeze(-1) * self.w.reshape(1, 1, -1) + self.b.reshape(1, 1, -1))
        return self.div * time_encode
    
class SpatialTemporalEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, loc_nums, p_dropout):
        super().__init__()
        self.embed_size = embed_size
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)
        self.loc_embed = nn.Embedding(loc_nums+1, embed_size, padding_idx=0)
        self.hour_embed = nn.Embedding(31+1, int(embed_size/4), padding_idx=0)
        self.fc = nn.Linear(embed_size + int(embed_size/4) ,embed_size)
        self.dropout = nn.Dropout(p_dropout)
        self.tanh = nn.Tanh()
        
    def forward(self, token_seq, hour_seq, **kwargs):
        token_emb = self.loc_embed(token_seq)
        hour_emb = self.hour_embed(hour_seq)
        pos_embed = self.encoding_layer(token_seq, **kwargs)
        embed = self.dropout(self.tanh(self.fc(torch.cat([token_emb, hour_emb], dim=-1)) + pos_embed))
        return embed
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, num_heads, p_dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=p_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(self.embed_size, eps=1e-6))

    def forward(self, student_embed, src_key_padding_mask_1):
        output = self.encoder(student_embed, src_key_padding_mask=src_key_padding_mask_1)  
        return output
    
class CorrelationAttentionLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(CorrelationAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout,  batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        attn_output, attn_weights = self.attention(x, y, y, key_padding_mask=mask)
        x = self.norm1(attn_output + x)
        x = self.dropout(x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        out = self.dropout(out)
        return out, attn_weights

class CorrelationAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout, forward_expansion):
        super(CorrelationAttentionBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                CorrelationAttentionLayer(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, mask=None):
        attn_weights = None
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, y, mask)
        return x, attn_weights

class AttentionGuidedMaskStrategy(nn.Module):
    def __init__(self, threshold, embed_size, devices):
        super(AttentionGuidedMaskStrategy, self).__init__()
        self.embed_size = embed_size
        self.mask_embedding = nn.Parameter(torch.randn(1, self.embed_size)) 
        self.devices = devices
        self.mask_ratio = threshold
    
    def generate_mask(self, embedding, cross_attn_weights, query_padding_mask = None, key_padding_mask = None):
        B, L, E = embedding.shape
        attn_mask = cross_attn_weights.masked_fill(query_padding_mask.unsqueeze(-1), 0)
        attn_key = torch.sum(attn_mask, dim=1)
        non_padding_counts = (~key_padding_mask).sum(dim=1) 
        k_values = (self.mask_ratio * non_padding_counts).int()     
        mask_romove_topk = torch.zeros((B, L), device=embedding.device)   
        for i in range(B):
            k = k_values[i].item()
            if k > 0:
                non_zero_indices = attn_key[i].nonzero(as_tuple=True)[0]
                non_zero_values = attn_key[i][non_zero_indices]
                _, topk_indices = torch.topk(non_zero_values, k, largest=False)
                indices_to_mask = non_zero_indices[topk_indices]                   
                mask_romove_topk[i, indices_to_mask] = 1  
        mask = mask_romove_topk       
        return mask 

    def forward(self, attn_a, attn_b, embed_a, embed_b, a_padding_mask = None, b_padding_mask = None):
        attn_a = attn_a.to(self.devices)
        attn_b = attn_b.to(self.devices)
        embed_a = embed_a.to(self.devices)
        embed_b = embed_b.to(self.devices)    
        mask_b = self.generate_mask(embed_b, attn_a, a_padding_mask, b_padding_mask)
        mask_a = self.generate_mask(embed_a, attn_b, b_padding_mask, a_padding_mask)
        mask_b = mask_b.to(self.devices).unsqueeze(-1)
        mask_a = mask_a.to(self.devices).unsqueeze(-1)
        unmasked_b = (1 - mask_b) * embed_b 
        masked_b = mask_b * self.mask_embedding  
        masked_embed_b = unmasked_b + masked_b     
        unmasked_a = (1 - mask_a) * embed_a  
        masked_a = mask_a * self.mask_embedding 
        masked_embed_a = unmasked_a + masked_a
        return masked_embed_b, masked_embed_a

class MTLink(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, num_layers_mask, num_layers_attn, p_drop, num_heads, threshold, devices, loc_nums):
        super(MTLink, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.trans_num_layers = num_layers
        self.mask_num_layers = num_layers_mask
        self.attn_num_layer = num_layers_attn
        self.num_heads = num_heads
        self.threshold = threshold  
        self.devices = devices
        self.loc_nums = loc_nums
        self.dropout = p_drop

        self.position_module = TemporalEncoding(self.embed_size)
        self.embedding_module = SpatialTemporalEmbedding(self.position_module, self.embed_size, self.loc_nums, self.dropout)
        self.temporal_encoder = TransformerEncoder(self.embed_size, self.hidden_size, self.trans_num_layers, self.num_heads, self.dropout)
        self.masked_encoder = TransformerEncoder(self.embed_size, self.hidden_size, self.mask_num_layers, self.num_heads, self.dropout)
        self.attn_mask = AttentionGuidedMaskStrategy(self.threshold, self.embed_size, self.devices)
        self.correlation_attention = CorrelationAttentionBlock(self.embed_size, self.num_heads, self.attn_num_layer,self.dropout, 2 * self.embed_size)
        self.binary_classifier = BinaryClassifier(self.embed_size)



    def forward(self, a_loc_seq, a_tim_seq, a_tims_seq, a_seq_len, b_loc_seq, b_tim_seq, b_tims_seq, b_seq_len):
        src_a_padding_mask = (a_loc_seq == 0)
        src_b_padding_mask = (b_loc_seq == 0) 
        a_embed = self.embedding_module(a_loc_seq, a_tim_seq, time_seq=a_tims_seq)
        b_embed = self.embedding_module(b_loc_seq, b_tim_seq, time_seq=b_tims_seq)
        a_output = self.temporal_encoder(a_embed, src_a_padding_mask)
        b_output = self.temporal_encoder(b_embed, src_b_padding_mask)
        a_output, a_attn_weights = self.correlation_attention(a_output, b_output, src_b_padding_mask)
        b_output, b_attn_weights = self.correlation_attention(b_output, a_output, src_a_padding_mask)
        b_new_embed, a_new_embed = self.attn_mask(a_attn_weights, b_attn_weights, a_embed, b_embed, src_a_padding_mask, src_b_padding_mask)
        last_a_output = self.masked_encoder(a_new_embed, src_a_padding_mask)
        last_b_output = self.masked_encoder(b_new_embed, src_b_padding_mask)
        x = self.binary_classifier(last_a_output, last_b_output, a_seq_len, b_seq_len)
        return x
                   
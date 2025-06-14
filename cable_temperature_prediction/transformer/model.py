import math
import torch
from torch import nn


class _ScaledDotProductAttention(nn.Module):
  '''
  Scaled Dot-Product Attention function as described in section 3.2.1. Used as part of the Multi-Head Attention layer.
  '''
  def __init__(self):
    super(_ScaledDotProductAttention, self).__init__()
    # calculate attention weights
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, Q, K, V, mask=None):
    # transpose the final 2 dimensions of K to allow multiplication with Q
    K = K.permute(0, 1, 3, 2) # [b, h, sz_k, d_k] -> [b, h, d_k, sz_k]

    # calulate attention matrix between Q and K
    attn = Q.matmul(K) # [b, h, sz_q, d_q] @ [b, h, d_k, sz_k] -> [b, h, sz_q, sz_k]

    # scale attention matrix by factor sqrt(d_k)
    attn = attn / torch.tensor(K.shape[-2])

    # mask out illegal attention value connections
    if mask is not None:
      attn = attn.masked_fill_(mask, -math.inf)

    # convert attention values to weights
    attn = self.softmax(attn)
    # multiply weighted attention with V
    out = attn.matmul(V)

    return out, attn # attention weighted values, attention weights
  

class _MultiHeadAttention(nn.Module):
  '''
  Multi-Head Attention sub-layer as described in section 3.2.2. Used as part of the Encoder layer.
  '''
  def __init__(self, d_model, h):
    super(_MultiHeadAttention, self).__init__()
    # embedding size
    self.d_model = d_model
    # number of heads
    self.h = h
    # embedding projection size for query, keys and values vectors
    self.d_q = self.d_k = self.d_v = self.d_model // self.h
    # linear projection layers for embeddings
    self.fc_Q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
    self.fc_K = nn.Linear(in_features=self.d_model, out_features=self.d_model)
    self.fc_V = nn.Linear(in_features=self.d_model, out_features=self.d_model)
    # attention function
    self.attention = _ScaledDotProductAttention()
    # linear projection layer for attention
    self.fc_mh_out = nn.Linear(in_features=self.d_model, out_features=self.d_model)

  def forward(self, Q, K, V, mask=None):
    batch_size = Q.shape[0]
    # linear projection of Q, K and V
    p_Q = self.fc_Q(Q) # [b, sz_q, d_model] -> [b, sz_q, d_model]
    p_K = self.fc_K(K) # [b, sz_k, d_model] -> [b, sz_k, d_model]
    p_V = self.fc_V(V) # [b, sz_v, d_model] -> [b, sz_v, d_model]

    # divide embedding dimension into seperate heads for Q, K, V
    p_Q = p_Q.reshape((batch_size, -1, self.h, self.d_q)) # [b, sz_q, d_model] -> [b, sz_q, h, d_q]
    p_K = p_K.reshape((batch_size, -1, self.h, self.d_k)) # [b, sz_k, d_model] -> [b, sz_k, h, d_k]
    p_V = p_V.reshape((batch_size, -1, self.h, self.d_v)) # [b, sz_v, d_model] -> [b, sz_v, h, d_v]

    # move the head dimension of Q, K and V
    p_Q = p_Q.permute((0, 2, 1, 3)) # [b, sz_q, h, d_q] -> [b, h, sz_q, d_q]
    p_K = p_K.permute((0, 2, 1, 3)) # [b, sz_k, h, d_k] -> [b, h, sz_k, d_k]
    p_V = p_V.permute((0, 2, 1, 3)) # [b, sz_v, h, d_v] -> [b, h, sz_v, d_v]

    # calculate the scaled dot product attention for each head in parallel
    mh_out, mh_attn = self.attention(p_Q, p_K, p_V, mask)

    # move the head dimension of the attention weighted values
    mh_out = mh_out.permute((0, 2, 1, 3)) # [b, sz_v, h, d_v] -> [b, sz_v, h, d_v]

    # concatenate heads of attention weighted values
    mh_out = mh_out.reshape((batch_size, -1, self.d_model)) # [b, sz_v, h, d_v] -> [b, sz_v, h * d_v (d_model)]

    # linear projection of attention weighted values
    mh_out = self.fc_mh_out(mh_out) # [b, sz_v, d_model] -> [b, sz_v, d_model]

    return mh_out, mh_attn # multi-head output, multi-head attention weights
  

class _FeedForwardNetwork(nn.Module):
  '''
  Position-wise Feed Forward Network sub-layer as described in section 3.3. Used as part of the Encoder layer.
  '''
  def __init__(self, d_model, d_ff):
    super(_FeedForwardNetwork, self).__init__()
    # input size
    self.d_model = d_model
    # hidden units
    self.d_ff = d_ff
    # feed forward network layers
    self.fc_1 = nn.Linear(in_features=self.d_model, out_features=self.d_ff)
    self.fc_2 = nn.Linear(in_features=self.d_ff, out_features=self.d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc_2(self.relu(self.fc_1(x)))
  

class _PositionalEncoding(nn.Module):
  '''
  Positional Encoding as described in section 3.5.
  '''
  def __init__(self, embedding_dim):
    super(_PositionalEncoding, self).__init__()
    # embedding size
    self.embedding_dim = embedding_dim
    # 2i / d_model
    self.exp = torch.arange(start=0, end=self.embedding_dim, step=2, dtype=torch.float32) / self.embedding_dim
    # 10000
    self.base = torch.full(size=(self.exp.shape[-1],), fill_value=10000.0, dtype=torch.float32)
    # 10000 ^ (2i / d_model)
    self.denominator = torch.pow(self.base, self.exp)

  def forward(self, x):
    # input sequence size
    num_samples = x.shape[-2]
    # initialise positional encoding for each sequence position
    pe = torch.zeros(size=(num_samples, self.embedding_dim)).to(x.device)
    
    # calculate positional encoding for each position in the input sequence
    for pos in range(num_samples):
      # PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
      pe[pos, 0::2] = torch.sin(self.denominator)
      # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
      pe[pos, 1::2] = torch.cos(self.denominator)

    # combine input embedding and positional encoding
    x = x + pe
    return x
  

class _EncoderLayer(nn.Module):
  '''
  Encoder layer as described in section 3.1. Contains the multi-head attention and feed forward network sub-layers.
  '''
  def __init__(self, num_heads, d_model, d_ff):
    super(_EncoderLayer, self).__init__()
    # embedding size
    self.d_model = d_model
    # number of attention heads
    self.h = num_heads
    # feed foward network hidden units
    self.d_ff = d_ff
    # multi-head attention sub-layer
    self.mha = _MultiHeadAttention(self.d_model, self.h)
    # multi-head attention layer norm
    self.layer_norm_mha = nn.LayerNorm(normalized_shape=self.d_model)
    # feed forward network sub-layer
    self.ffn = _FeedForwardNetwork(self.d_model, self.d_ff)
    # feed foward network layer norm
    self.layer_norm_ffn = nn.LayerNorm(normalized_shape=self.d_model)

  def forward(self, x):
    # multihead attention
    query = keys = values = x
    mha_out, mha_attn = self.mha(query, keys, values)
    # residual connection and layer norm
    x = self.layer_norm_mha(x + mha_out)

    # feed forward network
    ffn_out = self.ffn(x)
    # residual connection and layer norm
    x = self.layer_norm_ffn(x + ffn_out)
    return x
  

class _Encoder(nn.Module):
  '''
  Encoder as described in section 3.1. Contains multiple encoder layers.
  '''
  def __init__(self, N, d_model, h, d_ff):
    super(_Encoder, self).__init__()
    # number of encoder layers
    self.N = N
    # embedding size
    self.d_model = d_model
    # number of attention heads
    self.h = h
    # feed forward network hidden units
    self.d_ff = d_ff
    # encoder of N encoder layers
    self.encoder = nn.ModuleList([_EncoderLayer(self.h, self.d_model, self.d_ff) for i in range(self.N)])

  def forward(self, x):
    # pass input through each layer of the encoder
    for encoder_layer in self.encoder:
      x = encoder_layer(x)
    return x
  

class _DecoderLayer(nn.Module):
  '''
  Decoder layer as described in section 3.1. Contains the multi-head attention and feed forward network sub-layers.
  '''
  def __init__(self, num_heads, d_model, d_ff):
    super(_DecoderLayer, self).__init__()
    # embedding size
    self.d_model = d_model
    # number of attention heads
    self.h = num_heads
    # feed foward network hidden units
    self.d_ff = d_ff

    # masked multi-head attention sub-layer
    self.masked_mha = _MultiHeadAttention(self.d_model, self.h)
    # masked multi-head attention layer norm
    self.layer_norm_masked_mha = nn.LayerNorm(normalized_shape=self.d_model)

    # multi-head attention sub-layer
    self.mha = _MultiHeadAttention(self.d_model, self.h)
    # multi-head attention layer norm
    self.layer_norm_mha = nn.LayerNorm(normalized_shape=self.d_model)

    # feed forward network sub-layer
    self.ffn = _FeedForwardNetwork(self.d_model, self.d_ff)
    # feed foward network layer norm
    self.layer_norm_ffn = nn.LayerNorm(normalized_shape=self.d_model)

  def forward(self, x, encoder_output, mask=None):
    # masked multi-head attention
    query = keys = values = x
    masked_mha_out, masked_mha_attn = self.masked_mha(query, keys, values, mask)
    # residual connection and layer norm
    x = self.layer_norm_masked_mha(x + masked_mha_out)

    # multi-head attention
    query = x
    keys = values = encoder_output
    mha_out, mha_attn = self.mha(query, keys, values)
    # residual connection and layer norm
    x = self.layer_norm_mha(x + mha_out)

    # feed forward network
    ffn_out = self.ffn(x)
    # residual connection and layer norm
    x = self.layer_norm_ffn(x + ffn_out)

    return x
  

class _Decoder(nn.Module):
  '''
  Decoder as described in section 3.1. Contains multiple decoder layers.
  '''
  def __init__(self, N,  d_model, h, d_ff):
    super(_Decoder, self).__init__()
    # number of decoder layers
    self.N = N
    # embedding size
    self.d_model = d_model
    # number of attention heads
    self.h = h
    # feed forward network hidden units
    self.d_ff = d_ff
    # decoder of N decoder layers
    self.decoder = nn.ModuleList([_DecoderLayer(self.h, self.d_model, self.d_ff) for i in range(self.N)])

  def forward(self, x, encoder_output, mask=None):
    # pass inputs through each layer of the decoder
    for decoder_layer in self.decoder:
      x = decoder_layer(x, encoder_output, mask)
    return x
  

class Transformer(nn.Module):
  '''
  Transformer architecture as described in section 3.
  '''
  def __init__(self, input_length, num_encoder_layers, num_decoder_layers, embedding_dim, num_heads, hidden_dims):
    super(Transformer, self).__init__()
    self.input_length = input_length
    self.num_encoder_layers = num_encoder_layers
    self.num_decoder_layers = num_decoder_layers
    self.embedding_dim = embedding_dim
    # number of attention heads
    self.num_heads = num_heads
    self.hidden_dims = hidden_dims

    self.input_embedding = torch.nn.LazyLinear(out_features=self.embedding_dim)
    self.target_embedding = torch.nn.LazyLinear(out_features=self.embedding_dim)

    self.positional_encoding = _PositionalEncoding(self.embedding_dim)
    self.encoder = _Encoder(self.num_encoder_layers, self.embedding_dim, self.num_heads, self.hidden_dims)
    self.decoder = _Decoder(self.num_decoder_layers, self.embedding_dim, self.num_heads, self.hidden_dims)
    self.fc = nn.LazyLinear(out_features=1)

  def forward(self, x_enc, stats, x_dec):
    # repeat window statistics to be tiled with decoder output embedding
    stats = stats.unsqueeze(dim=1).repeat(1, self.input_length, 1).float()

    # embeddings
    x_enc = self.input_embedding(x_enc)
    x_dec = self.target_embedding(x_dec)

    # positional encodings
    x_enc = self.positional_encoding(x_enc)
    x_dec = self.positional_encoding(x_dec)

    # mask decoder embedding
    mask = torch.triu(torch.ones(size=(x_dec.shape[1], x_dec.shape[1]), requires_grad=False), diagonal=1).bool().to(x_dec.device)

    y_enc = self.encoder(x_enc)
    y_dec = self.decoder(x_dec, y_enc, mask=mask)

    # combine embedding and summary statistics calculated from window
    y_dec = torch.concat([y_dec, stats], axis=-1)
    
    output = self.fc(y_dec)

    return output
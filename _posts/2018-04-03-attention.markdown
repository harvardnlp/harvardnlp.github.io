---
layout: post
title: "The Annotated Transformer"
excerpt: "The Annotated Transformer"
tags:
    - python
    - notebook
---

**----------------------**


**There is now a [new version](http://harvardnlp.github.io/annotated-transformer) of this blog post updated for modern PyTorch.** 

**----------------------**


{% highlight python %}
from IPython.display import Image
Image(filename='images/aiayn.png')
{% endhighlight %}


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_0_0.png) 


 
The Transformer from ["Attention is All You
Need"](https://arxiv.org/abs/1706.03762) has been on a lot of people's minds
over the last year. Besides producing major improvements in translation quality,
it provides a new architecture for many other NLP tasks. The paper itself is
very clearly written, but the conventional wisdom has been that it is quite
difficult to implement correctly.

In this post I present an "annotated" version of the paper in the form of a
line-by-line implementation. I have reordered and deleted some sections from the
original paper and added comments throughout. This document itself is a working
notebook, and should be a completely usable implementation. In total there are
400 lines of library code which can process 27,000 tokens per second on 4 GPUs.

To follow along you will first need to install
[PyTorch](http://pytorch.org/).  The complete notebook is also
available on
[github](https://github.com/harvardnlp/annotated-transformer) or on
Google
[Colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sh
aring) with free GPUs.

Note this is merely a starting point for researchers and interested developers.
The code here is based heavily on our [OpenNMT](http://opennmt.net) packages.
(If helpful feel free to [cite](#conclusion).) For other full-sevice
implementations of the model check-out
[Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) (tensorflow) and
[Sockeye](https://github.com/awslabs/sockeye) (mxnet).

- Alexander Rush ([@harvardnlp](https://twitter.com/harvardnlp) or
srush@seas.harvard.edu), with help from Vincent Nguyen and Guillaume Klein
 
# Prelims 


{% highlight python %}
# !pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl numpy matplotlib spacy torchtext seaborn 
{% endhighlight %}


{% highlight python %}
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline
{% endhighlight %}
 
Table of Contents


* Table of Contents
{:toc} 
 
> My comments are blockquoted. The main text is all from the paper itself. 
 
# Background 
 
The goal of reducing sequential computation also forms the foundation of the
Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural
networks as basic building block, computing hidden representations in parallel
for all input and output positions. In these models, the number of operations
required to relate signals from two arbitrary input or output positions grows in
the distance between positions, linearly for ConvS2S and logarithmically for
ByteNet. This makes it more difficult to learn dependencies between distant
positions. In the Transformer this is reduced to a constant number of
operations, albeit at the cost of reduced effective resolution due to averaging
attention-weighted positions, an effect we counteract with Multi-Head Attention.

Self-attention, sometimes called intra-attention is an attention mechanism
relating different positions of a single sequence in order to compute a
representation of the sequence. Self-attention has been used successfully in a
variety of tasks including reading comprehension, abstractive summarization,
textual entailment and learning task-independent sentence representations. End-
to-end memory networks are based on a recurrent attention mechanism instead of
sequencealigned recurrence and have been shown to perform well on simple-
language question answering and
language modeling tasks.

To the best of our knowledge, however, the Transformer is the first transduction
model relying entirely on self-attention to compute representations of its input
and output without using sequence aligned RNNs or convolution. 
 
# Model Architecture 
 
Most competitive neural sequence transduction models have an encoder-decoder
structure [(cite)](https://arxiv.org/abs/1409.0473). Here, the encoder maps an
input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence of
continuous representations $\mathbf{z} = (z_1, ..., z_n)$. Given $\mathbf{z}$,
the decoder then generates an output sequence $(y_1,...,y_m)$ of symbols one
element at a time. At each step the model is auto-regressive
[(cite)](https://arxiv.org/abs/1308.0850), consuming the previously generated
symbols as additional input when generating the next. 


{% highlight python %}
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
{% endhighlight %}


{% highlight python %}
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
{% endhighlight %}
 
The Transformer follows this overall architecture using stacked self-attention
and point-wise, fully connected layers for both the encoder and decoder, shown
in the left and right halves of Figure 1, respectively. 


{% highlight python %}
Image(filename='images/ModalNet-21.png')
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_14_0.png) 


 
## Encoder and Decoder Stacks

### Encoder

The encoder is composed of a stack of $N=6$ identical layers. 


{% highlight python %}
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
{% endhighlight %}


{% highlight python %}
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
{% endhighlight %}
 
We employ a residual connection [(cite)](https://arxiv.org/abs/1512.03385)
around each of the two sub-layers, followed by layer normalization
[(cite)](https://arxiv.org/abs/1607.06450). 


{% highlight python %}
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
{% endhighlight %}
 
That is, the output of each sub-layer is $\mathrm{LayerNorm}(x +
\mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented
by the sub-layer itself.  We apply dropout
[(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each
sub-layer, before it is added to the sub-layer input and normalized.

To facilitate these residual connections, all sub-layers in the model, as well
as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$. 


{% highlight python %}
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
{% endhighlight %}
 
Each layer has two sub-layers. The first is a multi-head self-attention
mechanism, and the second is a simple, position-wise fully connected feed-
forward network. 


{% highlight python %}
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
{% endhighlight %}
 
### Decoder

The decoder is also composed of a stack of $N=6$ identical layers.
 


{% highlight python %}
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
{% endhighlight %}
 
In addition to the two sub-layers in each encoder layer, the decoder inserts a
third sub-layer, which performs multi-head attention over the output of the
encoder stack.  Similar to the encoder, we employ residual connections around
each of the sub-layers, followed by layer normalization. 


{% highlight python %}
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
{% endhighlight %}
 
We also modify the self-attention sub-layer in the decoder stack to prevent
positions from attending to subsequent positions.  This masking, combined with
fact that the output embeddings are offset by one position, ensures that the
predictions for position $i$ can depend only on the known outputs at positions
less than $i$. 


{% highlight python %}
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
{% endhighlight %}
 
> Below the attention mask shows the position each tgt word (row) is allowed to
look at (column). Words are blocked for attending to future words during
training. 


{% highlight python %}

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
None
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_31_0.png) 

 
### Attention
An attention function can be described as mapping a query and a set of key-value
pairs to an output, where the query, keys, values, and output are all vectors.
The output is computed as a weighted sum of the values, where the weight
assigned to each value is computed by a compatibility function of the query with
the corresponding key.

We call our particular attention "Scaled Dot-Product Attention".   The input
consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.
We compute the dot products of the query with all keys, divide each by
$\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
 


{% highlight python %}
Image(filename='images/ModalNet-19.png')
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_33_0.png) 


 

In practice, we compute the attention function on a set of queries
simultaneously, packed together into a matrix $Q$.   The keys and values are
also packed together into matrices $K$ and $V$.  We compute the matrix of
outputs as:

$$
   \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$ 


{% highlight python %}
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
{% endhighlight %}
 
The two most commonly used attention functions are additive attention
[(cite)](https://arxiv.org/abs/1409.0473), and dot-product (multiplicative)
attention.  Dot-product attention is identical to our algorithm, except for the
scaling factor of $\frac{1}{\sqrt{d_k}}$. Additive attention computes the
compatibility function using a feed-forward network with a single hidden layer.
While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented
using highly optimized matrix multiplication code.


While for small values of $d_k$ the two mechanisms perform similarly, additive
attention outperforms dot product attention without scaling for larger values of
$d_k$ [(cite)](https://arxiv.org/abs/1703.03906). We suspect that for large
values of $d_k$, the dot products grow large in magnitude, pushing the softmax
function into regions where it has extremely small gradients  (To illustrate why
the dot products get large, assume that the components of $q$ and $k$ are
independent random variables with mean $0$ and variance $1$.  Then their dot
product, $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, has mean $0$ and variance
$d_k$.). To counteract this effect, we scale the dot products by
$\frac{1}{\sqrt{d_k}}$. 
 
 


{% highlight python %}
Image(filename='images/ModalNet-20.png')
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_38_0.png) 


 
Multi-head attention allows the model to jointly attend to information from
different representation subspaces at different positions. With a single
attention head, averaging inhibits this.
$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ...,
\mathrm{head_h})W^O    \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

Where the projections are parameter matrices $W^Q_i \in
\mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in
\mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in
\mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times
d_{\text{model}}}$.
In this work we employ $h=8$ parallel attention layers, or heads. For each of
these we use $d_k=d_v=d_{\text{model}}/h=64$. Due to the reduced dimension of
each head, the total computational cost is similar to that of single-head
attention with full dimensionality. 


{% highlight python %}
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
{% endhighlight %}
 
### Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
1) In "encoder-decoder attention" layers, the queries come from the previous
decoder layer, and the memory keys and values come from the output of the
encoder.   This allows every position in the decoder to attend over all
positions in the input sequence.  This mimics the typical encoder-decoder
attention mechanisms in sequence-to-sequence models such as
[(cite)](https://arxiv.org/abs/1609.08144).


2) The encoder contains self-attention layers.  In a self-attention layer all of
the keys, values and queries come from the same place, in this case, the output
of the previous layer in the encoder.   Each position in the encoder can attend
to all positions in the previous layer of the encoder.


3) Similarly, self-attention layers in the decoder allow each position in the
decoder to attend to all positions in the decoder up to and including that
position.  We need to prevent leftward information flow in the decoder to
preserve the auto-regressive property.  We implement this inside of scaled dot-
product attention by masking out (setting to $-\infty$) all values in the input
of the softmax which correspond to illegal connections. 
 
## Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and
decoder contains a fully connected feed-forward network, which is applied to
each position separately and identically.  This consists of two linear
transformations with a ReLU activation in between.

$$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$

While the linear transformations are the same across different positions, they
use different parameters from layer to layer. Another way of describing this is
as two convolutions with kernel size 1.  The dimensionality of input and output
is $d_{\text{model}}=512$, and the inner-layer has dimensionality $d_{ff}=2048$. 


{% highlight python %}
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
{% endhighlight %}
 
## Embeddings and Softmax
Similarly to other sequence transduction models, we use learned embeddings to
convert the input tokens and output tokens to vectors of dimension
$d_{\text{model}}$.  We also use the usual learned linear transformation and
softmax function to convert the decoder output to predicted next-token
probabilities.  In our model, we share the same weight matrix between the two
embedding layers and the pre-softmax linear transformation, similar to
[(cite)](https://arxiv.org/abs/1608.05859). In the embedding layers, we multiply
those weights by $\sqrt{d_{\text{model}}}$. 


{% highlight python %}
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
{% endhighlight %}
 
## Positional Encoding
Since our model contains no recurrence and no convolution, in order for the
model to make use of the order of the sequence, we must inject some information
about the relative or absolute position of the tokens in the sequence.  To this
end, we add "positional encodings" to the input embeddings at the bottoms of the
encoder and decoder stacks.  The positional encodings have the same dimension
$d_{\text{model}}$ as the embeddings, so that the two can be summed.   There are
many choices of positional encodings, learned and fixed
[(cite)](https://arxiv.org/pdf/1705.03122.pdf).

In this work, we use sine and cosine functions of different frequencies:
$$PE_{(pos,2i)} = sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i/d_{\text{model}}})$$
where $pos$ is the position and $i$ is the dimension.  That is, each dimension
of the positional encoding corresponds to a sinusoid.  The wavelengths form a
geometric progression from $2\pi$ to $10000 \cdot 2\pi$.  We chose this function
because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be
represented as a linear function of $PE_{pos}$.

In addition, we apply dropout to the sums of the embeddings and the positional
encodings in both the encoder and decoder stacks.  For the base model, we use a
rate of $P_{drop}=0.1$.

 


{% highlight python %}
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
{% endhighlight %}
 
> Below the positional encoding will add in a sine wave based on position. The
frequency and offset of the wave is different for each dimension. 


{% highlight python %}
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend(["dim %d"%p for p in [4,5,6,7]])
None
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_49_0.png) 

 
We also experimented with using learned positional embeddings
[(cite)](https://arxiv.org/pdf/1705.03122.pdf) instead, and found that the two
versions produced nearly identical results.  We chose the sinusoidal version
because it may allow the model to extrapolate to sequence lengths longer than
the ones encountered during training. 
 
## Full Model

> Here we define a function that takes in hyperparameters and produces a full model. 


{% highlight python %}
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
{% endhighlight %}


{% highlight python %}
# Small example model.
tmp_model = make_model(10, 10, 2)
None
{% endhighlight %}
 
# Training

This section describes the training regime for our models. 
 
> We stop for a quick interlude to introduce some of the tools
needed to train a standard encoder decoder model. First we define a batch object
that holds the src and target sentences for training, as well as constructing
the masks. 
 
## Batches and Masking 


{% highlight python %}
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
{% endhighlight %}
 
> Next we create a generic training and scoring function to keep track of loss.
We pass in a generic loss compute function that also handles parameter updates. 
 
## Training Loop 


{% highlight python %}
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
{% endhighlight %}
 
## Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about
4.5 million sentence pairs.  Sentences were encoded using byte-pair encoding,
which has a shared source-target vocabulary of about 37000 tokens. For English-
French, we used the significantly larger WMT 2014 English-French dataset
consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary.


Sentence pairs were batched together by approximate sequence length.  Each
training batch contained a set of sentence pairs containing approximately 25000
source tokens and 25000 target tokens. 
 
> We will use torch text for batching. This is discussed in more detail below.
Here we create batches in a torchtext function that ensures our batch size
padded to the maximum batchsize does not surpass a threshold (25000 if we have 8
gpus). 


{% highlight python %}
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
{% endhighlight %}
 
## Hardware and Schedule
We trained our models on one machine with 8 NVIDIA P100 GPUs.  For our base
models using the hyperparameters described throughout the paper, each training
step took about 0.4 seconds.  We trained the base models for a total of 100,000
steps or 12 hours. For our big models, step time was 1.0 seconds.  The big
models were trained for 300,000 steps (3.5 days). 
 
## Optimizer

We used the Adam optimizer [(cite)](https://arxiv.org/abs/1412.6980) with
$\beta_1=0.9$, $\beta_2=0.98$ and $\epsilon=10^{-9}$.  We varied the learning
rate over the course of training, according to the formula:
$$
lrate = d_{\text{model}}^{-0.5} \cdot
  \min({step\_num}^{-0.5},
    {step\_num} \cdot {warmup\_steps}^{-1.5})
$$
This corresponds to increasing the learning rate linearly for the first
$warmup\_steps$ training steps, and decreasing it thereafter proportionally to
the inverse square root of the step number.  We used $warmup\_steps=4000$. 
 
> Note: This part is very important. Need to train with this setup of the model. 


{% highlight python %}

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
{% endhighlight %}
 

> Example of the curves of this model for different model sizes and for
optimization hyperparameters. 


{% highlight python %}
# Three settings of the lrate hyperparameters.
opts = [NoamOpt(512, 1, 4000, None), 
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000"])
None
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_69_0.png) 

 
## Regularization

### Label Smoothing

During training, we employed label smoothing of value $\epsilon_{ls}=0.1$
[(cite)](https://arxiv.org/abs/1512.00567).  This hurts perplexity, as the model
learns to be more unsure, but improves accuracy and BLEU score. 
 
> We implement label smoothing using the KL div loss. Instead of using a one-hot
target distribution, we create a distribution that has `confidence` of the
correct word and the rest of the `smoothing` mass distributed throughout the
vocabulary. 


{% highlight python %}
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
{% endhighlight %}
 
> Here we can see an example of how the mass is distributed to the words based
on confidence. 


{% highlight python %}
# Example of label smoothing.
crit = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]])
v = crit(Variable(predict.log()), 
         Variable(torch.LongTensor([2, 1, 0])))

# Show the target distributions expected by the system.
plt.imshow(crit.true_dist)
None
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_74_0.png) 

 
> Label smoothing actually starts to penalize the model if it gets very
confident about a given choice. 


{% highlight python %}
crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
None
{% endhighlight %}

 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_76_0.png) 

 
# A First  Example

> We can begin by trying out a simple copy-task. Given a random set of input
symbols from a small vocabulary, the goal is to generate back those same
symbols. 
 
## Synthetic Data 


{% highlight python %}
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
{% endhighlight %}
 
## Loss Computation 


{% highlight python %}
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm
{% endhighlight %}
 
## Greedy Decoding 


{% highlight python %}
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))
{% endhighlight %}

    Epoch Step: 1 Loss: 3.023465 Tokens per Sec: 403.074173
    Epoch Step: 1 Loss: 1.920030 Tokens per Sec: 641.689380
    1.9274832487106324
    Epoch Step: 1 Loss: 1.940011 Tokens per Sec: 432.003378
    Epoch Step: 1 Loss: 1.699767 Tokens per Sec: 641.979665
    1.657595729827881
    Epoch Step: 1 Loss: 1.860276 Tokens per Sec: 433.320240
    Epoch Step: 1 Loss: 1.546011 Tokens per Sec: 640.537198
    1.4888023376464843
    Epoch Step: 1 Loss: 1.682198 Tokens per Sec: 432.092305
    Epoch Step: 1 Loss: 1.313169 Tokens per Sec: 639.441857
    1.3485562801361084
    Epoch Step: 1 Loss: 1.278768 Tokens per Sec: 433.568756
    Epoch Step: 1 Loss: 1.062384 Tokens per Sec: 642.542067
    0.9853351473808288
    Epoch Step: 1 Loss: 1.269471 Tokens per Sec: 433.388727
    Epoch Step: 1 Loss: 0.590709 Tokens per Sec: 642.862135
    0.5686767101287842
    Epoch Step: 1 Loss: 0.997076 Tokens per Sec: 433.009746
    Epoch Step: 1 Loss: 0.343118 Tokens per Sec: 642.288427
    0.34273059368133546
    Epoch Step: 1 Loss: 0.459483 Tokens per Sec: 434.594030
    Epoch Step: 1 Loss: 0.290385 Tokens per Sec: 642.519464
    0.2612409472465515
    Epoch Step: 1 Loss: 1.031042 Tokens per Sec: 434.557008
    Epoch Step: 1 Loss: 0.437069 Tokens per Sec: 643.630322
    0.4323212027549744
    Epoch Step: 1 Loss: 0.617165 Tokens per Sec: 436.652626
    Epoch Step: 1 Loss: 0.258793 Tokens per Sec: 644.372296
    0.27331129014492034

 
> This code predicts a translation using greedy decoding for simplicity. 


{% highlight python %}
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
{% endhighlight %}

    
        1     2     3     4     5     6     7     8     9    10
    [torch.LongTensor of size 1x10]
    

 
# A Real World Example

> Now we consider a real-world example using the IWSLT German-English
Translation task. This task is much smaller than the WMT task considered in the
paper, but it illustrates the whole system. We also show how to use multi-gpu
processing to make it really fast. 


{% highlight python %}
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de
{% endhighlight %}
 
## Data Loading
> We will load the dataset using torchtext and spacy for tokenization. 


{% highlight python %}
# For data loading.
from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
{% endhighlight %}
 
> Batching matters a ton for speed. We want to have very evenly divided batches,
with absolutely minimal padding. To do this we have to hack a bit around the
default torchtext batching. This code patches their default batching to make
sure we search over enough sentences to find tight batches. 
 
## Iterators 


{% highlight python %}
class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
{% endhighlight %}
 
## Multi-GPU Training

> Finally to really target fast training, we will use multi-gpu. This code
implements multi-gpu word generation. It is not specific to transformer so I
won't go into too much detail. The idea is to split up word generation at
training time into chunks to be processed in parallel across many different
gpus. We do this using pytorch parallel primitives:

* replicate - split modules onto different gpus.
* scatter - split batches onto different gpus
* parallel_apply - apply module to batches on different gpus
* gather - pull scattered data back onto one gpu.
* nn.DataParallel - a special module wrapper that calls these all before
evaluating.
 


{% highlight python %}
# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize
{% endhighlight %}
 
> Now we create our model, criterion, optimizer, data iterators, and
paralelization 


{% highlight python %}
# GPUs to use
devices = [0, 1, 2, 3]
if True:
    pad_idx = TGT.vocab.stoi["<blank>"]
    model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)
None
{% endhighlight %}
 
> Now we train the model. I will play with the warmup steps a bit, but
everything else uses the default parameters.  On an AWS p3.8xlarge with 4 Tesla
V100s, this runs at ~27,000 tokens per second with a batch size of 12,000 
 
## Training the System 


{% highlight python %}
#!wget https://s3.amazonaws.com/opennmt-models/iwslt.pt
{% endhighlight %}


{% highlight python %}
if False:
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model_par, 
                  MultiGPULossCompute(model.generator, criterion, 
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
    model = torch.load("iwslt.pt")
{% endhighlight %}
 
> Once trained we can decode the model to produce a set of translations. Here we
simply translate the first sentence in the validation set. This dataset is
pretty small so the translations with greedy search are reasonably accurate. 


{% highlight python %}
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break
{% endhighlight %}

    Translation:	<unk> <unk> . In my language , that means , thank you very much . 
    Gold:	<unk> <unk> . It means in my language , thank you very much . 

 
# Additional Components: BPE, Search, Averaging 
 
> So this mostly covers the transformer model itself. There are four aspects
that we didn't cover explicitly. We also have all these additional features
implemented in [OpenNMT-py](https://github.com/opennmt/opennmt-py).

 
 
> 1) BPE/ Word-piece: We can use a library to first preprocess the data into
subword units. See Rico Sennrich's [subword-
nmt](https://github.com/rsennrich/subword-nmt) implementation. These models will
transform the training data to look like this: 
 
▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen
▁bestimmte n ▁Empfänger ▁gesendet ▁werden . 
 
> 2) Shared Embeddings: When using BPE with shared vocabulary we can share the
same weight vectors between the source / target / generator. See the
[(cite)](https://arxiv.org/abs/1608.05859) for details. To add this to the model
simply do this: 


{% highlight python %}
if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight
{% endhighlight %}
 
> 3) Beam Search: This is a bit too complicated to cover here. See the [OpenNMT-
py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/Beam.py)
for a pytorch implementation.

 
 
> 4) Model Averaging: The paper averages the last k checkpoints to create an
ensembling effect. We can do this after the fact if we have a bunch of models: 


{% highlight python %}
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))
{% endhighlight %}
 
# Results

On the WMT 2014 English-to-German translation task, the big transformer model
(Transformer (big)
in Table 2) outperforms the best previously reported models (including
ensembles) by more than 2.0
BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration
of this model is
listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs.
Even our base model
surpasses all previously published models and ensembles, at a fraction of the
training cost of any of
the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a
BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4
the training cost of the
previous state-of-the-art model. The Transformer (big) model trained for
English-to-French used
dropout rate Pdrop = 0.1, instead of 0.3.

 


{% highlight python %}
Image(filename="images/results.png")
{% endhighlight %}



 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_113_0.png) 


 
> The code we have written here is a version of the base model. There are fully
trained version of this system available here  [(Example
Models)](http://opennmt.net/Models-py/).
>
> With the addtional extensions in the last section, the OpenNMT-py replication
gets to 26.9 on EN-DE WMT. Here I have loaded in those parameters to our
reimplemenation. 


{% highlight python %}
!wget https://s3.amazonaws.com/opennmt-models/en-de-model.pt
{% endhighlight %}


{% highlight python %}
model, SRC, TGT = torch.load("en-de-model.pt")
{% endhighlight %}


{% highlight python %}
model.eval()
sent = "▁The ▁log ▁file ▁can ▁be ▁sent ▁secret ly ▁with ▁email ▁or ▁FTP ▁to ▁a ▁specified ▁receiver".split()
src = torch.LongTensor([[SRC.stoi[w] for w in sent]])
src = Variable(src)
src_mask = (src != SRC.stoi["<blank>"]).unsqueeze(-2)
out = greedy_decode(model, src, src_mask, 
                    max_len=60, start_symbol=TGT.stoi["<s>"])
print("Translation:", end="\t")
trans = "<s> "
for i in range(1, out.size(1)):
    sym = TGT.itos[out[0, i]]
    if sym == "</s>": break
    trans += sym + " "
print(trans)
{% endhighlight %}

    Translation:	<s> ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden . 

 
## Attention Visualization

> Even with a greedy decoder the translation looks pretty good. We can further
visualize it to see what is happening at each layer of the attention 


{% highlight python %}
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()

{% endhighlight %}

    Encoder Layer 2


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_1.png) 


    Encoder Layer 4


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_3.png) 


    Encoder Layer 6


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_5.png) 


    Decoder Self Layer 2


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_7.png) 


    Decoder Src Layer 2


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_9.png) 


    Decoder Self Layer 4


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_11.png) 


    Decoder Src Layer 4


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_13.png) 


    Decoder Self Layer 6


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_15.png) 


    Decoder Src Layer 6


 
![png]({{ BASE_PATH }}/images/the-annotated-transformer_119_17.png) 

 
# Conclusion

> Hopefully this code is useful for future research. Please reach out if you
have any issues. If you find this code helpful, also check out our other OpenNMT
tools.

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```

> Cheers,
> srush 
 
{::options parse_block_html="true" /}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO
INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT:
https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL
variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your
page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://harvard-nlp.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a
href="https://disqus.com/?ref_noscript">comments powered by
Disqus.</a></noscript>
 
 
<div id="disqus_thread"></div>
<script>
    /**
     *  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION
BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
     *  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT:
https://disqus.com/admin/universalcode/#configuration-variables
     */
    /*
    var disqus_config = function () {
        this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's
canonical URL variable
        this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with
your page's unique identifier variable
    };
    */
    (function() {  // REQUIRED CONFIGURATION VARIABLE: EDIT THE SHORTNAME BELOW
        var d = document, s = d.createElement('script');

        s.src = 'https://EXAMPLE.disqus.com/embed.js';  // IMPORTANT: Replace
EXAMPLE with your forum shortname!

        s.setAttribute('data-timestamp', +new Date());
        (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a
href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by
Disqus.</a></noscript> 

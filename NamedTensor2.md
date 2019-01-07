---
layout: page
title: "Tensor Considered Harmful"
excerpt: "Named tensors for better deep learning code."
---
<a href="https://colab.research.google.com/github/harvardnlp/namedtensor/blob/ma
ster/notebooks/NamedTensor2.ipynb" target="_parent"><img
src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In
Colab"/></a>

*-Alexander Rush*

* Table of Contents

{:toc}

*TL;DR: My previous post [Tensor Considered
Harmful](http://nlp.seas.harvard.edu/NamedTensor) proposed a method for using
**named tensors** to remedy some of the usability issues present in the
ubiquitous Tensor object. The post sparked significant conversation about
whether this was a useful technique or would simply litter code unnecessary
annotations. This follow-up post considers the use of these objects in real-
world deep learning examples. This entry is significantly more pragmatic than
the previous one, and particularly addresses the question of how named tensors
can coexist within the current deep learning ecosystem. The prototype **PyTorch
library** accompanying this blog post is available as
[namedtensor](https://github.com/harvardnlp/NamedTensor).*

# The Challenge

The post  [Tensor Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor)
proposes that many of the core usability issues of Tensor-based programming for
deep learning came down to difficulties in manipulating and aligning tensor
objects. The post targets inherent issues in the use of these objects in
practical experiments. Many of the examples consider direct linear algebra
manipulation experiments in a constrained environment.

In practice, though, this was cheating. To actually use *named tensors* we need
to interact with the world at large. The entire of libraries like PyTorch is
written with a tuple-based calling convention. If we want to write real software
in real environments, it is not sufficient to simply show that naming is useful,
it has be usable with the current functionality.

**The Challenge**: How can we *lift* deep learning systems in a pragmatic manner
so that they preserve the semantics of *named tensors*?




{% highlight python %}
#@title Setup
!rm -fr NamedTensor/; git clone -q https://github.com/harvardnlp/NamedTensor.git
!cd NamedTensor; pip install -q .; pip install -q torch numpy opt_einsum
!cp NamedTensor/notebooks/test* . 

{% endhighlight %}


{% highlight python %}
import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F
from namedtensor import NamedTensor, ntorch
from namedtensor import _im_init
_im_init()
{% endhighlight %}

## Method 1: Explicit Annotations

In PyTorch, the standard deep learning library lives in the `torch.nn` module
which is commonly abbreviated to `nn`. This library contains a collection of
modules that manipulate tensor objects.


{% highlight python %}
relu = nn.ReLU()
linear = nn.Linear(3, 1)
linear 
{% endhighlight %}




{:.output_data_text}
```
Linear(in_features=3, out_features=1, bias=True)
```



The API for these objects is specified through their tuple dimensions. For
instance for "relu" we see that this keeps the size the same as the original.



{% highlight python %}
print("\n".join(relu.__doc__.split("\n")[:13]))
{% endhighlight %}

    Applies the rectified linear unit function element-wise
        :math:`\text{ReLU}(x)= \max(0, x)`
    
        .. image:: scripts/activation_images/ReLU.png
    
        Args:
            inplace: can optionally do the operation in-place. Default: ``False``
    
        Shape:
            - Input: :math:`(N, *)` where `*` means, any number of additional
              dimensions
            - Output: :math:`(N, *)`, same shape as the input
    


On the other hand if we look at the linear object we can see that it takes as
input something of the form "(N, \*, in_features)" and outputs something of the
form  "(N, \*, out_features)"


{% highlight python %}
print("\n".join(linear.__doc__.split("\n")[:14]))
{% endhighlight %}

    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
    
        Shape:
            - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
              additional dimensions
            - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
              are the same shape as the input.
    


Now let's try this out with our trusty images.


{% highlight python %}
ims = torch.tensor(numpy.load('test_images.npy'), requires_grad=False).float()
ims = NamedTensor(ims, ("b", "h", "w", "c"))
first = ims.get("b", 0)
first
{% endhighlight %}




![png]({{ BASE_PATH }}/images/namedtensor2_16_0.png)




{% highlight python %}
# Standard method
relu(first.values.sub(0.5)).add(0.5)
{% endhighlight %}




![png]({{ BASE_PATH }}/images/namedtensor2_17_0.png)



Our approach is going to instead explicitly chain the operation through the `op`
method. This method takes in an op function that takes a single raw tensor as an
argument. In this case it is pretty boring it just applies the function
directly.


{% highlight python %}
first.sub(0.5).op(lambda x: relu(x)).add(0.5)

# or 

first.sub(0.5).op(relu).add(0.5)
{% endhighlight %}




![png]({{ BASE_PATH }}/images/namedtensor2_19_0.png)



Consider instead what happens when we apply the linear operation which changes
the dimension. Here the linear is modifying the last dimension of the tensor.


{% highlight python %}
try:
    first.op(linear)
except AssertionError:
    print("Failed to apply linear op.")
{% endhighlight %}

    Failed to apply linear op.



{% highlight python %}
first.op(linear, c2="c").get("c2", 0)
{% endhighlight %}




![png]({{ BASE_PATH }}/images/namedtensor2_22_0.png)



The main trade-off of this approach is that we need to still know the ordering
of the changed dimensions. For instance if we look at the shape of the Conv 2d
class we see that it expects channel first.


{% highlight python %}
conv = nn.Conv2d(3, 3, 2)
print("\n".join(conv.__doc__.split("\n")[75:85]))
{% endhighlight %}

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
    
              .. math::
                  H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                            \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
    
              .. math::
                  W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]



{% highlight python %}
ims.transpose("c", "h", "w").op(conv, c2="c", h2="h", w2="w").get("b", 0).transpose("h2", "w2", "c2")
{% endhighlight %}




![png]({{ BASE_PATH }}/images/namedtensor2_25_0.png)



In addition to `op` we consider two related methods. The method  `augment` wraps
operations that add a new dimension and `reduce` wraps operations that drop a
dimension,


{% highlight python %}
loss = nn.NLLLoss(reduce=False)
output = ntorch.randn(dict(batch=10, classes=20)).softmax("classes")
targets = ntorch.ones(dict(batch= 10)).long()
l = lambda x: loss(x, targets.values)
output.reduce(l, ("classes")).shape
{% endhighlight %}

    /usr/local/lib/python3.6/dist-packages/torch/nn/_reduction.py:49: UserWarning: size_average and reduce args will be deprecated, please use reduction='none' instead.
      warnings.warn(warning.format(ret))





{:.output_data_text}
```
OrderedDict([('batch', 10)])
```




{% highlight python %}
embedding = nn.Embedding(10, 20)
x = ntorch.ones(dict(batch= 10, slen=20)).long()
x.augment(embedding, "embeddingsize").shape
{% endhighlight %}




{:.output_data_text}
```
OrderedDict([('batch', 10), ('slen', 20), ('embeddingsize', 20)])
```



These two methods are really just syntactic sugar on top of the `op` method
above, but they make it a bit easier to tell what is happening when you read the
code.

## Method 2: Named Everything

The above approach is relatively pragmatic. We want to use the "unsafe" pytorch
library so we need to type it on input and output so that we can maintain our
labels. However ideally we can know exactly the names of the dimensions that are
being used so that we can propagate them through. Interestingly the PyTorch
distributions library is written in such a way to make this possible, so it is
fun to see what it looks like as a named library.


{% highlight python %}
import torch.distributions as distributions
import seaborn
import matplotlib.pyplot as plt
from namedtensor import ndistributions 
{% endhighlight %}

First let's make some parameters for a multivariate normal. and make a
distribution.


{% highlight python %}
mu = torch.randn(10, 2)
Sigma = torch.zeros(10, 2, 2)
Sigma[:] = torch.tensor([[3., 0], [0, 1.]])
{% endhighlight %}


{% highlight python %}
dist = distributions.MultivariateNormal(mu, Sigma)
{% endhighlight %}

Okay, so what happened here. We made a distribution object that has a bunch of
different distributions all combined together. This object has two important
properties, its batch shape and its event shape. In particular this is a batch
of 10 distributions each over with 2D outputs.


{% highlight python %}
dist.batch_shape, dist.event_shape
{% endhighlight %}




{:.output_data_text}
```
(torch.Size([10]), torch.Size([2]))
```



Now let's say we want to sample from all of these distributions simultaneously.
We can do that with this call.


{% highlight python %}
samples = dist.sample(torch.Size([20, 30]))
samples.shape
{% endhighlight %}




{:.output_data_text}
```
torch.Size([20, 30, 10, 2])
```



So now we have an object that is 20x30 samples of a 10 batches each of dim 2. Oh
man...



Okay, let's rewind and try it in named world now.


{% highlight python %}
mu = ntorch.randn(dict(dist=10, out=2)).mul(5)
Sigma = ntorch.zeros(dict(batch=10, a=2, b=2))
Sigma.values[:] = torch.tensor([[1., 0], [0, 1.]])
{% endhighlight %}


{% highlight python %}
dist = ndistributions.MultivariateNormal(mu, Sigma)
{% endhighlight %}

I've overridden the shape calls to give us named output now and sample takes in
a dict. Should be a bit more clear.


{% highlight python %}
dist.batch_shape, dist.event_shape
{% endhighlight %}




{:.output_data_text}
```
(OrderedDict([('dist', 10)]), OrderedDict([('out', 2)]))
```




{% highlight python %}
samples = dist.sample(sample1=20, sample2=30)
samples.shape
{% endhighlight %}




{:.output_data_text}
```
OrderedDict([('sample1', 20), ('sample2', 30), ('dist', 10), ('out', 2)])
```



So everything was the same as before, except that the distribution propagates
our dimension labels through to the end. This really comes in handy when you
want to do some plots. Here was enumerate over the samples from each
distribution and plot the samples.


{% highlight python %}
for i in range(10):
    b = samples.get("dist", i)
    plt.scatter(x=b.get("out", 0).values.numpy(),
                y=b.get("out", 1).values.numpy())
{% endhighlight %}


![png]({{ BASE_PATH }}/images/namedtensor2_48_0.png)


# Onward

The next question becomes whether this approach can actually be applied to real
deep learning problems, and do we like the result. To test this, I went through
several of the key deep learning micro-benchmarks to see what the code actually
looks like. Honestly, I am not sure I am completely convinced, I think it looks
like a start, but maybe not completely finished.

## MNist

The first example comes from a simple MNist network that is shipped with
PyTorch. The full original example is available at [MNist
example](https://github.com/pytorch/examples/blob/master/mnist/main.py) and our
named example is available [named MNist example](https://github.com/harvardnlp/n
amedtensor/blob/master/examples/mnist.py). Here we compare two variants, one
with standard tensor and the other with named tensor.

<img src="http://deeplearning.net/tutorial/_images/mylenet.png">


{% highlight python %}
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
{% endhighlight %}

The standard implementation is here (where I have added size comments). The code
is pretty clean, it is a strain line of applying modules one after the other.
The dimensions mostly line up along the way.


{% highlight python %}
class StandardCNN(BaseNet):
    def forward(self, x):
        # batch x c x h x w
        x = F.relu(self.conv1(x))
        # batch x c1 x h1 x w1
        x = F.max_pool2d(x, 2, 2)
        # batch x c1 x h1a x w1a
        x = F.relu(self.conv2(x))
        # batch x c2 x h2 x w2
        x = F.max_pool2d(x, 2, 2)
        # batch x c2a x h2a x w2a
        x = x.view(-1, 4*4*50)
        # batch x fc1
        x = F.relu(self.fc1(x))
        # batch x fc2
        x = self.fc2(x)
        # batch x classes
        return F.log_softmax(x, dim=1)
{% endhighlight %}

Contrast this to our named variant. The code is not necessarily more concise and
requires a bit more tinkering. However it does have some useful differences.

1) The `op` will check that changed dimensions get updated along the way, so it
is harder to screw up transposes.

2) Names are in the code so early failures have better errors. Furthermore we
can add checks like the `assert_size`.

3) The view and softmax become nicer in the process.



{% highlight python %}
class NamedNet(nn.Module):
    def forward(self, x):
        pool = lambda x: F.max_pool2d(x, 2, 2)

        return (
            x.transpose("c", "h", "w")
            .op(self.conv1, F.relu, c1="c", h1="h", w1="w")
            .op(pool, h1a="h1", w1a="w1")
            .op(self.conv2, F.relu, c2="c1", h2="h1a", w2="w1a")
            .assert_size(c2=50) # Just for fun.
            .op(pool, h2a="h2", w2a="w2")
            .stack(fc=("c2", "h2a", "w2a"))
            .op(self.fc1, F.relu, fc2="fc")
            .op(self.fc2, classes="fc2")
            .log_softmax("classes")
        )
{% endhighlight %}

The full implementation uses other aspects of the reduction for the loss and
backpropagation.

## Text Classification

The next example is a standard text classification problem. This is based model
from Yoon Kim in [Convolutional Neural Networks for Sentence Classification
](https://arxiv.org/abs/1408.5882).


<img src="http://www.wildml.com/wp-content/uploads/2015/11/Screen-
Shot-2015-11-06-at-8.03.47-AM.png">

This is a fun model because it is quite simple, but was surprisingly annoying to
implement in 2015. Nowadays it is just a couple of lines of code. I borrowed the
implementation of [Jun Wang](https://github.com/junwang4/CNN-sentence-
classification-pytorch-2018) and updated it for a [named tensor version](https:/
/github.com/harvardnlp/namedtensor/blob/master/examples/cnn_kim.py). Both use
the same set of parameters.


{% highlight python %}
class BaseCNN(nn.Module):
    def __init__(
        self,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        embedding_dim=300,
        pretrained_embeddings=None,
    ):
        super(BaseCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings)
        )

        conv_blocks = []
        for kernel_size in kernel_sizes:
            conv1d = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
            )

            conv_blocks.append(conv1d)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
{% endhighlight %}

Here is Jun Wang's implementation (with the original comments). Even though this
code looks simple, it has some traps including a transpose, cat, view and
softmax.


{% highlight python %}
class StandardCNN(BaseCNN):
    def forward(self, x):       # x: (batch, sentence_len)
        x = self.embedding(x)   # embedded x: (batch, sentence_len, embedding_dim)


        #    input:  (batch, in_channel=1, in_length=sentence_len*embedding_dim),
        #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        x = x.transpose(1,2)  # needs to convert x to (batch, embedding_dim, sentence_len)

        x_list= [F.relu(conv_block(x)).max(-1)[0]
                 for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        out = F.dropout(out, p=0.5, training=self.training)
        return F.softmax(self.fc(out), dim=1), feature_extracted
{% endhighlight %}

Here is the named version. We need to use `augment` to handle the extra
embedding dimension and
add several ops. However as a benefit we get to use names for the transpose,
cat, view and softmax.


{% highlight python %}
class NamedCNN(BaseCNN):
    def forward(self, x):  # x: (batch, slen)
        x = x.augment(self.embedding, "embedding") \
             .transpose("embedding", "slen")

        x_list = [x.op(conv_block, F.relu, time="slen", filters="embedding")
                  .max("time")[0]
                 for conv_block in self.conv_blocks]
        out = ntorch.cat(x_list, "filters")
        
        feature_extracted = out
        drop = lambda x: F.dropout(x, p=0.5, training=self.training)
        out = out.op(drop, self.fc, classes="filters") \
                 .softmax("classes") 
   
        return out, feature_extracted
{% endhighlight %}

## VAE

Finally let's consider a variational autoencoder (VAE). This is perhaps the
simplest version of the model with a feed-forward encoder and decoder. This
example is taken from the  [torch
examples](https://github.com/pytorch/examples/blob/master/vae/main.py) VAE and
updated to a [named
vae](https://github.com/harvardnlp/namedtensor/blob/master/examples/vae.py).

This example is to mainly show off the use of named distributions as a way
propagating forward dimensions.


<img src="http://ijdykeman.github.io/assets/cvae_figures/vae_diagram.svg">


(diagram from <a href="http://ijdykeman.github.io/ml/2016/12/21/cvae.html">Isaac
Dykeman's VAE explainer</a>)



{% highlight python %}
class BaseVAE(nn.Module):
    def __init__(self, num_samples):
        super(BaseVAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.num_samples = num_samples
{% endhighlight %}

Original code, using distributions for the latent space.


{% highlight python %}
class StandardVAE(V)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        normal = distributions.Normal(mu, logvar.mul(0.5).exp())
        return normal.rsample(torch.Size([self.num_samples])), normal
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z, normal = self.reparameterize(mu, logvar)
        return self.decode(z), normal
{% endhighlight %}

Named version. Relatively similar except using named distributions as above to
propagate named dimensions through sampling.


{% highlight python %}
class NamedVAE(V)
    def encode(self, x):
        h1 = x.op(self.fc1, F.relu, h="x")
        return h1.op(self.fc21, z="h"), h1.op(self.fc22, z="h")

    def reparameterize(self, mu, logvar):
        normal = ndistributions.Normal(mu, logvar.mul(0.5).exp())
        return normal.rsample(samples=self.num_samples), normal

    def decode(self, z):
        return z.op(self.fc3, F.relu, h="z").op(self.fc4, x="h").sigmoid()

    def forward(self, x):
        mu, logvar = self.encode(x.stack(x=("ch", "height", "width")))
        z, normal = self.reparameterize(mu, logvar)
        return self.decode(z), normal
{% endhighlight %}

# Next Steps

There was a lot of excellent feedback from the previous post, happy to hear
other ideas or pointers to other approaches. I feel like personally this is
getting close to a syntax that I would feel comfortable using. However, the
chain function call / pseudo-monad style can be a bit off-putting to people, so
it is possibly a non-statrter. Would be curious to know what you think.


*Sorry if there are tacky ads down here :(. Disqus seems to do it automatically.*

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
href="https://disqus.com/?ref_noscript" rel="nofollow">comments powered by Disqus.</a></noscript>

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lm.utils import count_params

#import line_profiler
"""
Dimension symbols:
    B - batch size
    S - sequence length
    D - hidden dimension (n_embd)
    H - number of attention heads (n_head)
    HD - hidden dimension of a single attention head (d // n_head)
    V - size of the vocabulary
"""


class MultiHeadAttention(nn.Module):
    """The multi-head attention module in a decoder block."""

    def __init__(self, n_embd: int, n_head: int, p_dropout: float = 0.1):
        super().__init__()
        """Initialize the modules used by multi-head attention."""

        self.n_head = n_head
        attn_hidden_dim = n_embd // n_head

        self.q_attn = nn.Linear(n_embd, n_embd)
        self.k_attn = nn.Linear(n_embd, n_embd)
        self.v_attn = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p_dropout)

        scale_factor = 1 / torch.sqrt(torch.tensor(attn_hidden_dim))
        self.register_buffer("scale_factor", scale_factor)

    def q_kT_v(
        self, x: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Project the hidden states to q, kT, v prior to computing attention.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
        """
        #q = rearrange(self.q_attn(x), 'b s (h hd) -> b h s hd', h=self.n_head) # ...
        #kT = rearrange(self.k_attn(x), 'b s (h hd) -> b h hd s', h=self.n_head) # ...
        #v = rearrange(self.v_attn(x), 'b s (h hd) -> b h s hd', h=self.n_head) # ...
        b = x.size(0)
        s = x.size(1)
        hd = x.size(2) // self.n_head

        q = x.reshape(b, self.n_head, s, hd)
        kT = x.reshape(b, self.n_head, hd, s)
        v = x.reshape(b, self.n_head, s, hd)

        return q, kT, v


    def self_attention(
        self,
        q: torch.FloatTensor,
        kT: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Compute multi-head attention over the inputs.

        Args:
            q: The query vector used by multi-head attention (B x H x S x HD)
            kT: The transpose of the key vector used by multi-head attention (B x H x HD x S)
            v: The value vector used by multi-head attention (B x H x S x HD)
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.

        Returns:
            attn: Outputs of applying multi-head attention to the inputs (B x S x D)
        """

        # compute the attention weights using q and kT
        # print(q.shape)
        qkT =  torch.matmul(q, kT) # ...
        unmasked_attn_logits = qkT * self.scale_factor

        """
        In decoder models, attention logits are masked such that computation at
        each position does not involve embeddings / hidden states of future
        positions.

        This boolean mask should have shape (S x S) and has value True iff
        position i is allowed to attend to position j (i.e., j <= i).

        Example (S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])
        
        Note that `causal mask` needs to be on the same device as the input
        tensors (q, kT, v). You can move a tensor to the right device by calling
        `tensor.to(q.device)`.

        Hint: torch.triu or torch.tril
        """
        causal_mask = torch.tril(torch.full((q.size(2), q.size(2)), True)).to(q.device) #...
        #print(causal_mask)
        #causal_mask.to(q.device)
        """
        Sometimes, we want to pad the input sequences so that they have the same
        length and can fit into the same batch. These padding tokens should not
        have any effect on the output of self-attention. To achieve this, we
        need to mask out the logits that correspond to those tokens.

        Example (B = 2, S = 5):
        causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])

        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        mask = tensor([
        [[[False, False, False, False, False],
          [False, False, False, False, False],
          [False, False,  True, False, False],
          [False, False,  True,  True, False],
          [False, False,  True,  True,  True]]],

        [[[ True, False, False, False, False],
          [ True,  True, False, False, False],
          [ True,  True,  True, False, False],
          [ True,  True,  True,  True, False],
          [ True,  True,  True,  True,  True]]]
        ])

        Note that `mask` needs to be on the same device as the input tensors
        q, kT and v.
        """

        if attention_mask is None:
            mask = causal_mask
            mask = mask.to(unmasked_attn_logits.device)
        else:
            #print(causal_mask.shape)
            #print(attention_mask.shape)
            #mask = torch.permute(casual_mask.float() @ attention_mask.t(), (2, 0, 1)).to(unmasked_attn_logits.device)
            mask = torch.einsum('ij,kj->kij', causal_mask.float(), attention_mask).to(unmasked_attn_logits.device)
            #mask = mask.unsqueeze(1).repeat(1, list(unmasked_attn_logits.shape)[1], 1)
            mask = repeat(mask, 'a c d -> a b c d', b=unmasked_attn_logits.size(1)).to(unmasked_attn_logits.device)
            #mask.to(q.device)
        #print(mask.shape)
        #mask.to(q.device)
        """
        Fill unmasked_attn_logits with float_min wherever causal mask has value False.

        Hint: torch.masked_fill
        """
        float_min = torch.finfo(q.dtype).min
        #print(unmasked_attn_logits.shape)
        attn_logits = unmasked_attn_logits.masked_fill(torch.logical_not(mask), float_min)
        #print(attn_logits)
        attn_weights = attn_logits.softmax(dim=-1) # ...
        #print(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights.to(v.device)
        #print(attn_weights)

        # scale value by the attention weights.
        attn = torch.matmul(attn_weights, v)
        #attn = attn.reshape(attn.size(0), attn.size(2), attn.size(1)*attn.size(3))
        attn = rearrange(attn, 'b h s hd -> b s (h hd)')
        #print(attn.shape)
        
        return attn

    def projection(self, attn: torch.FloatTensor) -> torch.FloatTensor:
        """Apply a dropout and a linear projection to outputs of attention"""
        return self.dropout(self.proj(attn))

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None = None
    ) -> torch.FloatTensor:
        """A full forward pass of the multi-head attention module.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block

        Returns:
            y: outputs (B x S x D) of the multi-head attention module
        """
        q, kT, v = self.q_kT_v(x)
        #if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        #    attn = rearrange(F.scaled_dot_product_attention(q, kT.rearrange(-2, -1), v, dropout_p=self.dropout.p, is_causal=True), 'b h s hd -> b s (h hd)')
        #else:
        attn = self.self_attention(q, kT, v, attention_mask)
        y = self.projection(attn)
        return y


class FeedForward(nn.Module):
    """The feedforward attention module in a decoder block."""

    def __init__(self, n_embd: int, p_dropout: float = 0.1):
        """Initialize the modules used by feedforward."""
        super().__init__()

        middle_dim = 4 * n_embd  # stick to what GPT-2 does
        self.linear_in = nn.Linear(n_embd, middle_dim)
        self.linear_out = nn.Linear(middle_dim, n_embd)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """A full forward pass of the feedforward module.

        Args:
            x: outputs (B x S x D) of the first Add & Norm operation

        Returns:
            z: outputs (B x S x D) of the feedforward module

        Different from what you saw in class which uses ReLU as the activation,
        we are going to follow GPT-2 which uses GeLU. You should also apply
        self.dropout to the output.
        """
        y = F.gelu(self.linear_in(x)) # y = F.gelu(...)
        z = self.dropout(self.linear_out(y)) # z = self.dropout(...)
        return z


class DecoderBlock(nn.Module):
    """A single decoder block in a decoder language model."""

    def __init__(self, n_embd: int, n_head: int):
        """Initialize the modules used in a decoder block."""
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.mha = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        #self.rms_1 = nn.RMSNorm(n_embd)
        #self.rms_2 = nn.RMSNorm(n_embd)

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.FloatTensor | None
    ) -> torch.FloatTensor:
        """A full forward pass of the decoder block.

        Args:
            x: embeddings or hidden states (B x S x D) from the previous decoder block
            attention_mask (optional): Mask indicating tokens that shouldn't
              be included in self-attention (B x S). 1 stands for a token that is
              included, and 0 stands for a token that isn't.
        Returns:
            y: outputs of the current decoder block

        A note on where to place layer normalization (LN): in the lecture, you
        saw "post-LN", which applies LN to the outputs of MHA / FF modules after
        the residual is added. Another approach to do this is "pre-LN", which
        appiles LN to the inputs of the attention and feedforward modules. Both
        implementations should pass the tests. See explanations here:
        https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab
        """
        #x = x + self.mha.forward(self.ln_1(x), attention_mask)
        #return x + self.ff.forward(self.ln_2(x))
        x = self.ln_1(x + self.mha.forward(x, attention_mask))
        return self.ln_2(x + self.ff.forward(x))


class DecoderLM(nn.Module):
    """The decoder language model."""
    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.p_dropout = p_dropout

        self.token_embeddings = nn.Embedding(n_vocab, n_embd)
        self.position_embeddings = nn.Embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(self.p_dropout)

        # initialize weights according to nanoGPT
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / torch.sqrt(2 * n_layer))

        # count flops per token according to nanoGPT
        self.flops_per_token = (
            6 * count_params(self) + 12 * n_layer * n_embd * n_positions
        )


    def embed(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """Convert input_ids to embeddings (token_embeddings + positional_embeddings).

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            embeddings: token representations with shape (B x S x D)
        """

        """# https://paperswithcode.com/method/weight-tying
        Position ids are indices of tokens in the sequence. When attention_mask
        isn't provided, they are simply [0, 1, 2, ...] for every sequence in the
        batch. When they are provided, you should ignore tokens with attention_mask
        equal to 0.
        
        Example (B = 2, S = 5):
        
        attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

        position_ids = tensor([
         [0, 0, 0, 1, 2],
         [0, 1, 2, 3, 4]
        ])

        Note that the position ids for masked out tokens do not matter, as long
        as they don't trigger out-of-bounds errors when fed into the embedding
        layer. I.e., they should be within [0, n_positions).

        Hint: torch.cumsum
        """

        assert input_ids.shape[1] <= self.n_positions
        token_embeddings = self.token_embeddings(input_ids)
        token_embeddings = token_embeddings.to(input_ids.device) # ...

        if attention_mask is not None:
            #print(attention_mask)
            attention_mask = attention_mask.to(input_ids.device)
            position_ids = torch.sub(torch.cumsum(attention_mask, dim=1), 1)
            #print(position_ids)
        else:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).repeat(input_ids.size(0), 1)
            #print(position_ids)
        #print(self.position_embeddings.weight)
        positional_embeddings = self.position_embeddings(position_ids.int()) #F.embedding(position_ids.int().to(input_ids.device), self.position_embeddings.weight.to(input_ids.device)) # ...
        positional_embeddings = positional_embeddings.to(token_embeddings.device)
        #print(positional_embeddings)

        return token_embeddings + positional_embeddings #self.dropout(token_embeddings + positional_embeddings)

    def token_logits(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project the final hidden states of the model to token logits.

        Args:
            x: hidden states produced by the final decoder block (B x S x D)

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)

        Hint: Use the weight tying technique discussed in Q1.2
        """

        logits = torch.einsum('b s d, v d -> b s v', x, self.token_embeddings.weight) #/ torch.sqrt(torch.tensor(self.n_embd)) # self.ln(x) # ...
        return self.dropout(logits)


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """A forward pass of the decoder LM, converting input_ids to token logits.

        Args:
            input_ids: tokens ids with shape (B x S)
            attention_mask (optional): Mask indicating whether tokens should be
              ignored.

        Returns:
            logits: logits corresponding to the predicted next token likelihoods (B x S x V)
        """
        x = self.embed(input_ids, attention_mask)

        # Idea from nanoGPT
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.ln(x)

        logits = self.token_logits(x) #self.embed(input_ids, attention_mask)) # ...
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: #...:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

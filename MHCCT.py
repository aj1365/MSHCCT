import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from models.MHA import MultiHeadAttention
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = default(stride, max(1, (kernel_size // 2) - 1))
    padding = default(padding, max(1, (kernel_size // 2)))

    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)

# positional

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

# modules



class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.1, drop_path_rate=0.1):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        #self.self_attn = MultiHeadAttention(in_features=d_model, head_num=nhead)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1    = nn.LayerNorm(d_model)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate)

        self.activation = F.gelu

    def forward(self, src, *args, **kwargs):
        
        #src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src1, _ = self.self_attn(self.pre_norm(src),self.pre_norm(src),self.pre_norm(src))
        src= src + self.drop_path(src1)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        batch, drop_prob, device, dtype = x.shape[0], self.drop_prob, x.device, x.dtype

        if drop_prob <= 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (batch, *((1,) * (x.ndim - 1)))

        keep_mask = torch.zeros(shape, device = device).float().uniform_(0, 1) < keep_prob
        output = x.div(keep_prob) * keep_mask.float()
        return output

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=1,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(chan_in, chan_out,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=1, height=16, width=16):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return rearrange(self.conv_layers(x), 'b c h w -> b (h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=64,
                 num_layers=2,
                 num_heads=4,
                 mlp_ratio=4.0,
                 num_out=100,         ############################# 30 f0r scene 20
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        assert positional_embedding in {'sine', 'learnable', 'none'}

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert exists(sequence_length) or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding == 'none':
            self.positional_emb = None
        elif positional_embedding == 'learnable':
            self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                               requires_grad=True)
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        else:
            self.positional_emb = nn.Parameter(sinusoidal_embedding(sequence_length, embedding_dim),
                                               requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    drop_path_rate=layer_dpr)
            for layer_dpr in dpr])

        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_out)
        self.apply(self.init_weight)

    def forward(self, x):
        b = x.shape[0]

        if not exists(self.positional_emb) and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = repeat(self.class_emb, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_token, x), dim=1)

        if exists(self.positional_emb):
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.seq_pool:
            attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n')
            x = einsum('b n, b n d -> b d', attn_weights.softmax(dim = 1), x)
        else:
            x = x[:, 0]

        return self.fc(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ClassifierHead(nn.Module):

    def __init__(self, num_classes):
        super(ClassifierHead, self).__init__()
        #self.dropout = SoftDropout(p=0.1, n=5)
        #self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.linear = nn.Linear(100, num_classes)        ####################### number of classes

    def forward(self, x):
        x = self.gelu(x)
        #x = self.dropout(x)
        pred = self.linear(x)

        return pred
        
class SoftDropout(nn.Module):

    def __init__(self, p:float=0.5, n:int=5):
        super(SoftDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(n)])

    def forward(self, x):
        outputs = torch.stack([dropout(x) for dropout in self.dropouts])
        return torch.sum(outputs, 0) / torch.numel(outputs)
# CCT Main model

class CCT(nn.Module):
    def __init__(
        self,
        img_size=256,  # Ensure this is an integer, or properly unpack if tuple
        embedding_dim=64,
        n_input_channels=3,
        n_conv_layers=3,
        kernel_size=3,
        stride=2,
        num_classes=45,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args, **kwargs
    ):
        super().__init__()
        
        # Ensure img_size is unpacked correctly
        img_height, img_width = pair(img_size)  # Correctly handles both int and tuple cases

        # Tokenizers
        self.tokenizers = nn.ModuleList([
            Tokenizer(n_input_channels=n_input_channels,
                      n_output_channels=embedding_dim,
                      kernel_size=k,
                      stride=stride,
                      padding=padding,
                      pooling_kernel_size=pooling_kernel_size,
                      pooling_stride=pooling_stride,
                      pooling_padding=pooling_padding,
                      max_pool=True,
                      activation=nn.GELU,
                      n_conv_layers=n_conv_layers,
                      conv_bias=False) for k in [1, 3, 5]
        ])

        # Ensure sequence_length receives valid integers
        self.classifiers = nn.ModuleList([
            TransformerClassifier(
                sequence_length=self.tokenizers[i].sequence_length(n_channels=n_input_channels,
                                                                   height=int(img_height),  
                                                                   width=int(img_width)),   
                embedding_dim=embedding_dim,
                seq_pool=True,
                dropout_rate=0.,
                stochastic_depth=0.1,
                *args, **kwargs)
            for i in range(3)
        ])

        # Final Classification Head
        self.classifier_head = ClassifierHead(num_classes)

    def forward(self, x):
        # Tokenize input using different kernels
        tokenized_outputs = [tokenizer(x) for tokenizer in self.tokenizers]

        # Pass through Transformer classifiers
        classified_outputs = [classifier(t) for classifier, t in zip(self.classifiers, tokenized_outputs)]

        # Aggregate multi-scale features
        x = sum(classified_outputs)  # Element-wise sum

        return self.classifier_head(x)

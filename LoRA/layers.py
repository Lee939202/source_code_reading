#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    '''
    这段代码实现了一个嵌入层（embedding layer），并加入了 LoRA（Learned Optimized Relative Attention）的机制。
    这个嵌入层继承了 PyTorch 中的 nn.Embedding 和自定义的 LoRALayer 类，实现了一个嵌入层与 LoRA 层的组合。

    在 **init**() 方法中，首先调用了 nn.Embedding 的构造方法，初始化了嵌入矩阵。
    然后调用了 LoRALayer 的构造方法，传入了 LoRA 相关的参数，用于初始化 LoRA 层的变量。
    接着，如果 LoRA 中的 r 大于 0，就初始化 LoRA 的参数 A、B，对参数进行了归一化和初始化。
    最后调用了 reset_parameters() 方法，对嵌入矩阵和 LoRA 层的参数进行了初始化。

    reset_parameters() 方法中，首先调用了 nn.Embedding 的 reset_parameters() 方法，对嵌入矩阵进行了初始化。
    然后判断是否存在 LoRA 层，如果存在，则对参数 A 进行零初始化，对参数 B 进行正态分布初始化。

    train() 方法中，调用了 nn.Embedding 的 train() 方法，并判断是否需要合并权重。
    如果需要合并权重，并且权重没有被合并过，则将权重合并，并将合并状态标记为 True。

    eval() 方法中，调用了 nn.Embedding 的 eval() 方法，并判断是否需要合并权重。
    如果需要合并权重，并且权重已经被合并过，则将权重拆分，并将合并状态标记为 False。

    forward() 方法中，首先调用了 nn.Embedding 的 forward() 方法，获取嵌入矩阵的结果。
    如果 LoRA 中的 r 大于 0，并且权重没有被合并过，则对结果进行 LoRA 层的计算，并返回结果；否则直接返回嵌入矩阵的结果。
    在 LoRA 层的计算中，先通过 F.embedding() 方法获取参数 A 对应的矩阵，然后与参数 B 相乘，再乘以归一化因子，得到最终的结果。
    '''
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            '''
            self.scaling是一个缩放因子，是一个浮点数，其值由LoRA的超参数lora_alpha和r计算得出。
            具体地，self.scaling是用lora_alpha除以r计算得出的，表示将LoRA输出的值缩放到合适的大小。
            在实现中，self.scaling用于缩放lora_A和lora_B的矩阵乘积，以便在后续训练和评估中正确合并或分离这些权重。
            '''
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged 确保权重未合并
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Embedding.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it 合并权重并进行标记
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    '''
    这是一个继承自 nn.Linear 和 LoRALayer 的类，用于实现 Linear LoRA 模型。
    其中 nn.Linear 是 PyTorch 提供的线性层类，LoRALayer 是我们自己定义的一个实现了 Linear LoRA 模型的类。
    这个类将这两个类的特点结合在一起，用来构建 Linear LoRA 模型。

    下面是这个类的主要方法的解释和注释：

    __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: int = 1, 
    lora_dropout: float = 0., fan_in_fan_out: bool = False, merge_weights: bool = True, **kwargs)：
    初始化方法，用来定义该层的输入和输出特征数、LoRA 模型的参数等等。
    其中 in_features 是输入特征的数量，
    out_features 是输出特征的数量，
    r 是 LoRA 模型的参数之一，控制模型的稀疏性。
    lora_alpha 和 lora_dropout 是控制模型训练的参数。
    fan_in_fan_out 和 merge_weights 用于指示输入和输出特征是否需要交换，并指示是否应该合并预先训练的权重矩阵和 LoRA 参数。
    fan_in和fan_out是指一个全连接层（或者卷积层）的输入和输出通道数量。对于全连接层，fan_in指的是输入特征的数量，而fan_out指的是输出特征的数量。
    对于卷积层，fan_in指的是输入通道的数量，fan_out指的是输出通道的数量。

    fan_in_fan_out是一个bool类型的参数，当它被设置为True时，表示该层的权重参数将按照(fan_in, fan_out)的形状存储。
    否则，权重参数将按照(fan_out, fan_in)的形状存储。
    在PyTorch中，默认情况下，全连接层的权重参数按照(fan_out, fan_in)的形状存储，
    而卷积层的权重参数按照(out_channels, in_channels, kernel_size)的形状存储。

    reset_parameters(self)：重新初始化模型参数，对于 LoRA 参数 A 使用 nn.init.kaiming_uniform_ 方法进行初始化，
    对于 LoRA 参数 B 使用 nn.init.zeros_ 方法进行初始化。

    train(self, mode: bool = True)：设置训练模式。如果已经合并过权重，则通过减去 LoRA 参数的乘积来取消合并权重，否则保持不变。

    eval(self)：设置评估模式。如果没有合并过权重，则通过加上 LoRA 参数的乘积来合并权重，否则保持不变。

    forward(self, x: torch.Tensor)：前向传播方法。
    如果已经合并过权重，则使用预先训练的权重矩阵进行线性变换，否则使用预先训练的权重矩阵和 LoRA 参数进行线性变换。
    最终返回变换结果。
    '''
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out) 如果要替换的层存储类似（fan_in，fan_out）的权重，则将此设置为True
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    '''
    这是一个继承自nn.Linear的神经网络层类，同时也实现了LoRA（Low-Rank Approximation）算法，是一种用于神经网络压缩和加速的技术。

    在__init__函数中，通过传入in_features和out_features来确定该层的输入和输出特征数，同时也接收一些其他参数，如LoRA算法的参数，dropout率等等。

    在reset_parameters函数中，该层的权重参数会按照kaiming_uniform_的方法进行初始化，而lora_A会初始化为全零，lora_B会初始化为0。

    在train函数中，如果merge_weights为True，且该层权重未被合并，就会先将它们合并，以减少神经网络的参数数量，从而提高训练速度。
    合并权重的过程中，会使用卷积函数F.conv1d来计算lora_A和lora_B的乘积，然后使用self.zero_pad函数将结果插入到weight的相应位置，
    最终将合并后的权重更新到该层的weight参数中。

    在eval函数中，也是类似的过程，只不过是将合并后的权重恢复为原来的形式。这样做的原因是在训练和测试的过程中，合并权重会导致结果不准确。

    在forward函数中，先使用F.linear函数计算出输入x的结果result，然后如果该层已经合并了权重，直接返回结果；
    否则，使用Lora算法计算lora_A和lora_B的乘积，然后使用self.zero_pad函数将结果插入到result的相应位置，最终将结果返回。
    '''
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, 'The length of enable_lora must divide out_features'
        '''
        这行代码是将传入的enable_lora列表保存为self.enable_lora属性，该属性用于指定哪些输出通道应该使用LoRA，哪些不应该使用LoRA。
        在MergedLinear层中，每个输出通道都可以使用LoRA（如果对应的enable_lora列表项为True），也可以不使用（如果对应的enable_lora列表项为False）。
        如果enable_lora列表的长度不能整除输出特征数，会抛出一个错误。
        '''
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        '''
        这段代码是一个PyTorch模型的自定义层的初始化方法。这个自定义层实现了一种叫做LoRa的卷积神经网络结构。

        如果输入特征数量大于零（in_features）且enable_lora列表中有任何一个值为True，就会执行下列操作：

        创建LoRa层的权重矩阵lora_A和lora_B，它们的大小是(r * sum(enable_lora), in_features)和(out_features // len(enable_lora) * sum(enable_lora), r)，
        其中r是LoRa层的超参数，out_features是输出特征数量。这里用了PyTorch的nn.Parameter将它们包装成可优化的变量。

        计算一个缩放参数scaling，用于调整lora_A和lora_B的初始值。这个参数等于超参数lora_alpha除以r。

        将该层的weight参数设置为不需要梯度计算（requires_grad=False），以防止它被更新。

        计算一个掩码lora_ind，用于限制后续的卷积操作只对lora_A和lora_B的非零元素进行计算。
        掩码的大小为(out_features, )，在执行lora_ind[enable_lora, :] = True这行代码时，
        只有enable_lora列表中值为True的位置对应的行会被赋为True。

        调用reset_parameters()方法，这个方法会根据LoRa层的初始化方式初始化权重矩阵。

        如果fan_in_fan_out参数为True，则将权重矩阵进行转置（即做一个转置操作），这是为了保持输入特征数与输出特征数之间的对称性。

        总体来说，这段代码就是在初始化LoRa层的权重矩阵，并且根据超参数和输入输出特征数进行调整。
        '''
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            '''
            这段代码的作用是为了计算LORA（Locally Recurrent Autoencoder）模型的权重矩阵中需要保留哪些权重，哪些需要被置零。
            具体来说，它会根据是否启用LORA以及LORA模型的参数来计算一个布尔类型的矩阵lora_ind，用于在之后的前向计算中实现只对一部分权重进行循环。

            假设我们有一个输出特征数量为10，输入特征数量为5的全连接层，现在要开启LORA，LORA的参数是[True, False, True, False, True]。
            这意味着我们将使用LORA对第1、3、5个输出特征进行循环计算。
            那么对于这个例子，enable_lora列表的值就是[True, False, True, False, True]。

            然后，这段代码会创建一个形状为(out_features, )的全零张量，并将其转换为形状为(len(enable_lora), -1)的矩阵。
            在本例中，out_features为10，因此我们将创建一个形状为(5, 2)的矩阵。
            接下来，它将enable_lora中值为True的行设置为True。在本例中，这意味着第1、3、5行会被设置为True。
            最后，它将矩阵重新变形为形状为(out_features, )的一维张量，并将其存储在lora_ind变量中。
            因此，lora_ind将是一个形状为(10, )的张量，其中第1、3、5个元素为True，其余为False，表示LORA模型需要循环计算这三个输出特征的权重。
            '''
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        '''
        在 Python 中，星号 * 被用作解包符号，它可以将可迭代对象拆开成独立的参数。
        在这个函数中，*x.shape[:-1] 表示将 x.shape 中除了最后一个元素以外的所有元素拆开，作为函数的参数传递。

        举个例子，假设 x.shape 为 (3, 4, 5)，那么 x.shape[:-1] 就是 (3, 4)，
        *x.shape[:-1] 就会将这两个数字作为两个独立的参数传递给函数，即 zero_pad(self, x) 中的第一个参数。
        在这个例子中，(*x.shape[:-1], self.out_features) 就相当于 (3, 4, self.out_features)。
        '''
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        '''
        这段代码中的train方法包括了以下步骤：

        1.调用nn.Linear.train()，将当前线性层的training mode设置为传入的参数mode（默认为True）。

        2.如果当前线性层开启了参数合并（self.merge_weights=True）且参数已经被合并过（self.merged=True），则需要执行以下步骤：

        如果当前线性层开启了LoRA，并且其中至少一个LoRA分支被启用（self.r > 0 and any(self.enable_lora)），
        则需要计算出LoRA的更新量delta_w，并将其乘以self.scaling进行缩放。
        将缩放后的delta_w进行零填充（zero padding），使其形状与线性层的权重self.weight相同，并将其与self.weight相减，从而更新线性层的权重。
        将参数的合并状态self.merged设置为False，表示下一次需要重新合并参数。
        '''
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result
            

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    '''
    这是一个在卷积层上实现的 LoRA 算法，以下是算法的具体流程：

    初始化：初始化一个普通卷积层，并在上面添加 LoRA 相关的参数和属性。
    其中参数包括 r、lora_alpha 和 lora_dropout，属性包括 lora_A、lora_B、scaling、merged 等。
    如果 r > 0，则还需要为 lora_A 和 lora_B 生成新的张量。

    训练和验证：在训练时，调用 nn.Conv2d.train()，在评估时调用 nn.Conv2d.eval()。
    如果 merge_weights 为 True，则在训练时和评估时将卷积层的权重与 lora_A 和 lora_B 相乘并相加，从而实现权重融合。
    注意，这个过程只在 merged 为 False 时执行。

    前向传播：在前向传播过程中，如果 merged 为 False，则先将卷积层的权重与 lora_A 和 lora_B 相乘并相加，
    然后再调用 nn.Conv2d.forward() 函数。
    这个过程相当于是将 LoRA 的输出加到卷积层的输出上。
    如果 merged 为 True，则直接调用 nn.Conv2d.forward() 函数。

    总体来说，这个算法就是在卷积层的基础上增加了一些可学习的参数，并使用这些参数来计算 LoRA 的输出，然后将这个输出加到卷积层的输出上。
    这样可以增加模型的表达能力，提高模型的性能。
    '''
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)


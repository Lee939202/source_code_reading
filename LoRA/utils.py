#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer


'''
这是一个使用PyTorch实现的LoRA（Learned Optimizer with Relevance Aware Layer）模型的辅助函数，
包括两个函数：mark_only_lora_as_trainable和lora_state_dict。

mark_only_lora_as_trainable函数用于将模型中除了LoRA层以外的参数设置为不可训练（requires_grad=False），
以便在训练过程中只更新LoRA层的参数。可以通过传递bias参数来控制是否将偏置参数也设置为可训练。
如果bias为'none'，则只有权重参数是可训练的；
如果bias为'all'，则所有偏置参数都是可训练的；
如果bias为'lora_only'，则只有LoRA层的偏置参数是可训练的。

lora_state_dict函数用于返回模型中LoRA层的状态字典，可以通过传递bias参数来控制是否返回偏置参数。
如果bias为'none'，则只返回权重参数；
如果bias为'all'，则返回所有权重参数和偏置参数；
如果bias为'lora_only'，则返回LoRA层的权重参数和偏置参数。

LoRA模型是一种优化器模型，其目的是学习如何优化神经网络的参数。
具体来说，LoRA模型会在每个训练步骤中动态调整优化器的超参数，以适应当前的优化问题。
该模型是通过在神经网络中增加一个LoRA层来实现的，LoRA层的输出用于动态调整优化器的超参数。
'''
def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    

"""
Date: 2021-06-02 00:33:09
LastEditors: GodK
"""
import sys

sys.path.append("../")
from common.utils import Preprocessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


class MyDataset(Dataset):
    """
    这是一个自定义数据集类 MyDataset，继承自 torch.utils.data.Dataset 类。
    这个类的作用是将数据包装成可供 PyTorch 使用的数据集格式，以便进行数据处理、训练等操作。

    类 MyDataset 中有三个方法，分别是 __init__、__getitem__ 和 __len__。

    __init__ 方法是类的初始化方法，在创建 MyDataset 对象时调用，它接收一个参数 data，表示数据集。
    在这个方法中，将 data 赋值给类的成员变量 self.data，并计算数据集的长度 self.length。

    __getitem__ 方法用于获取数据集中指定位置的数据。
    它接收一个参数 index，表示要获取的数据的位置。
    在这个方法中，直接返回 self.data[index] 即可。

    __len__ 方法用于获取数据集的长度。
    在这个方法中，直接返回 self.length 即可。

    这个类的作用是将数据集包装成 PyTorch 中的 Dataset 类型，方便后续的数据处理和训练操作。
    在使用时，可以将原始数据传入 MyDataset 类中进行包装，然后使用 torch.utils.data.DataLoader 将数据加载进来，
    方便进行数据的迭代访问。
    """
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataMaker(object):
    """
    这是一个名为 DataMaker 的类，主要是用于对数据进行预处理，将原始数据转换成模型输入所需的格式。

    在 __init__ 方法中，该类接收一个 tokenizer 参数和一个 add_special_tokens 参数。
    tokenizer 表示使用的分词器，
    add_special_tokens 表示是否在序列的开始和结尾添加特殊的 token，如 [CLS] 和 [SEP] 等。

    generate_inputs 方法接收三个参数：datas，max_seq_len 和 ent2id，用于生成输入模型的数据。
    datas 表示输入的数据列表，max_seq_len 表示序列的最大长度，ent2id 表示实体类型到 id 的映射。

    在这个方法中，首先使用 tokenizer 对 sample["text"] 进行分词，得到 tokenized 后的文本。
    然后使用 preprocessor 中的 get_ent2token_spans 方法，得到实体在 tokenized 后的文本中的位置，
    进而生成对应的标签，将其转化为 one-hot 编码，存储在 labels 变量中。
    最后将 input_ids、attention_mask、token_type_ids 和 labels 以字典的形式返回。

    generate_batch 方法接收四个参数：batch_data、max_seq_len、ent2id 和 data_type，
    用于将一个 batch 的数据转换成模型输入所需的格式。
    在这个方法中，首先调用 generate_inputs 方法，将 batch_data 转换成模型输入所需的格式，
    然后将样本、input_ids、attention_mask、token_type_ids 和 labels 分别存储在不同的列表中，
    并使用 torch.stack 将它们拼接成一个张量，最后返回拼接后的张量。

    decode_ent 方法目前为空，可能是用于将模型预测的结果转化成实体的方法，但是实现还未完整。
    """
    def __init__(self, tokenizer, add_special_tokens=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """生成喂入模型的数据

        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """

        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in datas:
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length'
            )

            labels = None
            if data_type != "predict":
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"]
                )
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    labels[ent2id[label], start, end] = 1
            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if labels is not None:
                labels = torch.tensor(inputs["labels"]).long()

            sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)

            all_inputs.append(sample_input)

        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "predict":
                labels_list.append(sample[4])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0) if data_type != "predict" else None

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels

    def decode_ent(self, pred_matrix):
        pass


class MetricsCalculator(object):
    """
    这是一个用于计算模型指标的类，包含了三个方法：

    get_sample_f1(y_pred, y_true)：计算样本的 F1 值。
    这里的 y_pred 和 y_true 是模型预测的结果和真实标签，都是二维张量（batch_size, max_seq_len）。

    get_sample_precision(y_pred, y_true)：计算样本的精确率（precision）。
    这里的 y_pred 和 y_true 是模型预测的结果和真实标签，都是二维张量（batch_size, max_seq_len）。

    get_evaluate_fpr(y_pred, y_true)：计算评估指标，包括 F1 值、精确率（precision）和召回率（recall）。
    这里的 y_pred 和 y_true 是模型预测的结果和真实标签，都是二维张量（batch_size, max_seq_len）。
    
    其中 get_sample_f1 和 get_sample_precision 的计算方式较为简单，直接使用公式计算即可。
    而 get_evaluate_fpr 的计算稍微复杂一些，它首先将 y_pred 和 y_true 转换成 Numpy 数组，
    然后根据阈值确定预测结果，将预测和真实标签中的实体位置提取出来，计算 F1 值、精确率和召回率。
    """
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()

        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()

        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        return f1, precision, recall


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        """
        这个函数是一个用于生成Sin-Cos位置Embedding的函数。
        输入参数包括batch_size（批次大小）、seq_len（序列长度）和output_dim（输出维度）。
        函数的作用是生成一个形状为(batch_size, seq_len, output_dim)的Sin-Cos位置Embedding张量。

        在函数内部，首先创建一个包含0到seq_len-1的位置id的张量position_ids。
        接下来，我们创建一个包含0到output_dim/2-1的索引张量indices，并计算出一个包含10000的指数的张量。
        这个指数被用于计算Sin-Cos函数的周期。

        然后，我们将position_ids与indices相乘，并将其输入到Sin-Cos函数中，
        得到一个形状为(batch_size, seq_len, output_dim/2, 2)的张量。
        这个张量被重复batch_size次，然后调整形状为(batch_size, seq_len, output_dim)。
        最后，将生成的Sin-Cos位置Embedding张量移动到指定设备中，并返回。
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)

        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        这段代码是一个PyTorch模型的前向计算过程，可以理解为输入一些数据，输出预测的结果。下面我将逐行进行解释：

        self.device = input_ids.device
        这一行代码将输入的input_ids所在的设备类型（CPU或GPU）赋值给了模型的device属性。
        这里的device属性是一个字符串，表示该模型在哪个设备上运行。
        这行代码的作用是为了确保后面所有的张量都在同一个设备上。

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        这一行代码调用了一个子模型（self.encoder）对输入数据（input_ids，attention_mask和token_type_ids）进行编码。
        这里的encoder可以是一个文本嵌入模型（如BERT），它会将输入的文本转换为向量表示。
        context_outputs是编码器的输出，它是一个包含多个张量的元组，
        其中context_outputs[0]表示编码器输出的最后一层的输出。

        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        这一行代码将context_outputs的第一个张量赋值给了last_hidden_state。
        last_hidden_state是编码器输出的最后一层的张量表示，它的形状为(batch_size, seq_len, hidden_size)。
        其中batch_size表示输入的数据样本数量，seq_len表示输入的文本序列长度，hidden_size表示每个单词被嵌入为多少维度的向量。

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        这两行代码分别获取last_hidden_state的第一维和第二维的大小，即batch_size和seq_len。这里的作用是为了后面的张量操作做准备。

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        这两行代码对last_hidden_state进行线性变换，生成一个新的张量outputs。
        outputs的形状为(batch_size, seq_len, ent_type_size*inner_dim*2)，
        其中ent_type_size表示每个实体类型的数量，inner_dim表示输出的每个实体的向量表示的维度。
        第二行代码使用torch.split函数按照维度-1（最后一维）进行分割，
        将outputs分割成两个形状为(batch_size, seq_len, ent_type_size, inner_dim)的张量。
        这里的作用是为了后面的张量操作做准备。
        """
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        """
        将模型最后一层输出last_hidden_state通过一个全连接层self.dense进行处理，得到形状为(batch_size, seq_len, ent_type_size * inner_dim * 2)的张量outputs。

        接着，torch.split函数将outputs按照最后一个维度分成两部分，即qw和kw，每部分形状为(batch_size, seq_len, ent_type_size, inner_dim)。

        最后，torch.stack函数将qw和kw沿着倒数第二个维度进行拼接，得到形状为(batch_size, seq_len, ent_type_size, inner_dim*2)的张量outputs，
        其中倒数第二个维度被拆分成了两个维度：ent_type_size和inner_dim，这样方便后面的计算。
        拆分和拼接这两个操作相当于把一个形状为(batch_size, seq_len, ent_type_size * inner_dim * 2)的张量outputs
        分成两个形状为(batch_size, seq_len, ent_type_size, inner_dim)的张量qw和kw。
        """
        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        """
        这段代码是实现RoPE（Relative Positional Encoding）的功能，用于更好地捕捉序列中位置信息。

        RoPE是通过将序列的位置嵌入到词嵌入向量中来表示位置信息。
        具体来说，这里使用了一种sinusoidal position encoding的方式，将每个位置用一个由sin和cos函数组成的向量表示。
        然后将这个向量加到词嵌入向量中，来表达每个词的位置信息。

        具体实现上，首先调用了一个叫做sinusoidal_position_embedding的函数，用于生成pos_emb，即位置嵌入向量。
        pos_emb的维度是(batch_size, seq_len, inner_dim)，其中batch_size是输入数据的批大小，seq_len是序列的长度，inner_dim是每个位置向量的维度。

        然后，将pos_emb拆分成两部分，分别是cos_pos和sin_pos。
        cos_pos和sin_pos的维度都是(batch_size, seq_len, 1, inner_dim/2)，其中inner_dim/2是每个sin和cos函数的维度，也是每个位置向量的一半。
        由于sin和cos函数的周期是2*pi，因此inner_dim必须是偶数，以便将其分成两部分。
        在这里，取的是pos_emb的偶数和奇数维度分别作为sin和cos函数的输入。

        接着，将qw和kw分别拆分成两部分，
        分别是qw1、qw2和kw1、kw2。qw1和kw1都是pos_emb中的cos部分，qw2和kw2是pos_emb中的sin部分，
        这样就可以将位置信息加到输入中了。

        具体来说，首先将qw2和kw2按照位置交叉组合起来，
        即将qw2的偶数维和kw2的奇数维组合在一起，qw2的奇数维和kw2的偶数维组合在一起，形成qw2'和kw2'。
        qw2'和kw2'的维度都是(batch_size, seq_len, ent_type_size, inner_dim/2)。

        然后，将qw1和kw1与cos_pos相乘，将qw2'和kw2'与sin_pos相乘，再将相乘的结果相加起来，即得到了新的qw和kw。
        这样，输入中就融入了位置信息。
        """
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        """
        这段代码是实现了一个矩阵乘法，也就是点积操作，即将qw和kw的转置相乘，最终输出一个四维的张量logits。
        具体的操作是通过einsum函数实现的，其中'b'代表batch_size，'m'和'n'代表seq_len，'h'代表inner_dim，可以理解为一个多维矩阵的索引。

        在进行点积之前，首先需要进行一个padding mask的操作，因为在句子中，有些部分是填充的，不能对它们进行点积，需要进行过滤。
        通过attention_mask计算得到一个四维张量pad_mask，然后将其与logits进行点积，
        并对pad_mask的补集进行运算，将这些填充位置对应的值全部设为一个非常小的值（这里设为1e-12），这样在softmax时，填充位置对应的概率值就会接近于0。

        最后，还有一个操作是排除下三角，即将下三角的值全部设置为一个非常小的值（同样设为1e-12），这样在softmax时，下三角对应的概率值就会接近于0。
        """
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

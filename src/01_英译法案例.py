import re

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# 起始标记
SOS_token = 0
# 结束标记
EOS_token = 1
# 序列最大长度
MAX_LENGTH = 10
# 数据文件的路径
data_path = "./data/eng-fra-v2.txt"


# 清洗函数
def normalizeString(s):
    """
    字符串标准化
    :param s: 待处理的字符串
    :return: 处理后的字符串
    """
    # 转成小写且去掉两边空格
    s = s.lower().strip()
    # 在 .!? 前面增加一个 空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 将 非标准字符 替换成 空格
    s = re.sub(r"[^a-zA-Z.!?]+", " ", s)
    # 返回数据
    return s


# 获取数据
def get_data(file_path):
    # 将数据加载到内存中
    with open(file_path, "r", encoding="utf-8") as f:
        # 一次性获取所有行
        lines = f.readlines()

    # 清洗文本并构建句子对
    pairs = [[normalizeString(s) for s in line.split("\t")] for line in lines]

    # 构建英语词典
    english_word2index = {"SOS": SOS_token, "EOS": EOS_token}
    # 构建法语词典
    french_word2index = {"SOS": SOS_token, "EOS": EOS_token}

    # 定义集合记录去重后的单词
    english_set = set()
    french_set = set()

    # 获取全部单词
    for pair in pairs:
        # 英语单词
        for word in pair[0].split(" "):
            english_set.add(word)
        # 法语单词
        for word in pair[1].split(" "):
            french_set.add(word)

    # 给集合排序，确保每次生成的词典顺序一致
    english_set = sorted(english_set)
    french_set = sorted(french_set)

    # 将单词录入词典中
    # 英语
    for word in english_set:
        english_word2index[word] = len(english_word2index)
    # 法语
    for word in french_set:
        french_word2index[word] = len(french_word2index)

    # 构建反向映射词典  索引->单词
    english_index2word = {index: word for word, index in english_word2index.items()}
    french_index2word = {index: word for word, index in french_word2index.items()}

    # 返回数据
    return (
        pairs,
        english_word2index,
        english_index2word,
        french_word2index,
        french_index2word,
    )


# 定义全局数据
(
    pairs,
    english_word2index,
    english_index2word,
    french_word2index,
    french_index2word,
) = get_data(data_path)


# 数据集对象
class PairsDataset(Dataset):
    # 初始化
    def __init__(self, pairs, english_word2index, french_word2index):
        self.pairs = pairs
        self.samples_len = len(self.pairs)
        self.english_word2index = english_word2index
        self.french_word2index = french_word2index

    def __len__(self):
        return self.samples_len

    # 填充或截断至指定长度
    def _pad_or_truncate(self, tensor, max_length, pad_value=EOS_token):
        length = tensor.size(0)
        if length < max_length:
            # 用 0 补齐
            pad = torch.full(
                (max_length - length,),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,  # 跟随原 tensor 的设备，避免 CPU/GPU 不匹配
            )
            return torch.cat([tensor, pad], dim=0)
        else:
            return tensor[:max_length]  # 截断

    def __getitem__(self, index):
        # 确保索引在合法范围内
        index = min(max(index, 0), self.samples_len - 1)
        # 根据索引获取对应句子
        x = self.pairs[index][0]
        y = self.pairs[index][1]
        # 按空格切割句子，将单词转为对应索引
        x = [self.english_word2index[word] for word in x.split(" ")]
        y = [self.french_word2index[word] for word in y.split(" ")]
        # 添加结束标记
        x.append(EOS_token)
        y.append(EOS_token)
        # 将数据转化为张量
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        # 填充/截断序列长度为 MAX_LENGTH
        tensor_x = self._pad_or_truncate(tensor_x, MAX_LENGTH)
        tensor_y = self._pad_or_truncate(tensor_y, MAX_LENGTH)

        # 返回数据 (英语张量, 法语张量)
        return tensor_x, tensor_y


# 获取数据加载器对象
def get_dataloader(batch_size=1):
    dataset = PairsDataset(pairs, english_word2index, french_word2index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 构建基于GRU的编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        初始化对象
        :param input_size: 词嵌入层的输入维度，即：词典的大小
        :param hidden_size: 词嵌入层的输入维度，即：词向量的大小
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 词嵌入层
        self.embedding = nn.Embedding(input_size, hidden_size)
        # GRU层
        # 参1：输入的特征维度，即 词向量的大小
        # 参2：隐藏层维度
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # 前向传播
    def forward(self, input, hidden):
        output = self.embedding(input)  # 词嵌入层
        output, hidden = self.gru(output, hidden)  # GRU层
        return output, hidden

    def init_hidden(self,batch_size=1):
        """
        初始化隐藏层
        :return:
        """
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# 构建基于GRU的解码器
class DecoderRNN(nn.Module):
    # 参数初始化
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.output_size = output_size  # 解码器的输出维度，即 法语词典的大小
        self.hidden_size = hidden_size  # 解码器的隐藏层维度
        self.dropout_p = dropout_p  # 随机失活的概率
        self.max_length = max_length  # 句子最大长度

        # 解码器的词嵌入层
        # 输入: [batch_size, seq_len]     输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(output_size, hidden_size)

        # 注意力权重计算层
        # 隐藏层与词嵌入层拼接后映射成注意力权重    [1, 1, 256] + [1, 1, 256] = [1, 1, 512] -> [1, 1, 10]
        self.attn = nn.Linear(hidden_size * 2, max_length)

        #  注意力融合层
        #  词嵌入层与注意力权重融合 [1, 1, 512] -> [1, 1, 256]
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        # GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # dropout层
        self.dropout = nn.Dropout(p=dropout_p)
        # 输出层
        self.output = nn.Linear(hidden_size, output_size)

    # 前向传播
    def forward(self, input, hidden, encoder_output):
        embedded = self.embedding(input)  # 词嵌入层
        embedded = self.dropout(embedded)  # 随机失活

        # 将 hidden 维度从 [1, batch, 256] 转置为 [batch, 1, 256]，才能和 embedded 拼接
        hidden_attn = hidden.transpose(0, 1)

        # 计算注意力权重
        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded, hidden_attn), dim=-1)), dim=-1
        )

        # 计算注意力上下文
        # [1, 1, 10] @ [1, 10, 256] -> [1, 1, 256]
        attn_applied = torch.bmm(attn_weights, encoder_output)

        # 词嵌入层与注意力权重融合
        # [1, 1, 256] + [1, 1, 256] = [1, 1, 512] -> [1, 1, 256]
        output = self.attn_combine(torch.cat((attn_applied, embedded), dim=-1))

        # 激活函数
        output = torch.relu(output)
        # GRU层
        output, hidden = self.gru(output, hidden)
        # 输出层
        output = self.output(output)
        # 返回数据
        return output, hidden, attn_weights

    # 初始化隐藏层
    def init_hidden(self,batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# 编码器测试
def encoder_test():
    # 获取数据加载器
    dataloader = get_dataloader(batch_size=1)

    # 定义参数
    vocab_size = len(english_word2index)
    hidden_size = 256

    # 创建编码器对象
    encoder_rnn = EncoderRNN(vocab_size, hidden_size)

    # 将编码器移入GPU中
    encoder_rnn.to(device)

    # 遍历数据加载器
    for i, (x, y) in enumerate(dataloader):
        print(f"x: {x}, shape: {x.shape}")
        print(f"y: {y}, shape: {y.shape}")

        hidden = encoder_rnn.init_hidden()
        output, hidden = encoder_rnn(x, hidden)

        print(f"output: {output.shape}")
        print(f"hidden: {hidden.shape}")

        break


# 解码器测试
def decoder_test():
    # 获取数据加载器
    dataloader = get_dataloader(batch_size=1)

    # 定义参数
    english_vocab_size = len(english_word2index)
    french_vocab_size = len(french_word2index)
    hidden_size = 256

    # 创建编码器对象,并移入GPU中
    encoder_rnn = EncoderRNN(input_size=english_vocab_size, hidden_size=hidden_size).to(
        device
    )
    # 创建解码器对象,并移入GPU中
    decoder_rnn = DecoderRNN(output_size=french_vocab_size, hidden_size=hidden_size).to(
        device
    )

    # 遍历数据加载器
    for i, (x, y) in enumerate(dataloader):
        print(f"x: {x}, shape: {x.shape}")
        print(f"y: {y}, shape: {y.shape}")

        # 编码器的前向传播
        encoder_hidden = encoder_rnn.init_hidden()
        encoder_output, encoder_hidden = encoder_rnn(x, encoder_hidden)

        print(f"encoder_output: {encoder_output.shape}")
        print(f"encoder_hidden: {encoder_hidden.shape}")
        print()

        for j in range(y.shape[1]):
            # 解码器的前向传播
            decoder_input = y[0][j].view(1, -1)
            decoder_output, decoder_hidden, attn_weights = decoder_rnn(
                decoder_input, encoder_hidden, encoder_output
            )

            print(f"第{j+1}个法语单词:")
            print(f"decoder_output: {decoder_output.shape}")
            print(f"decoder_hidden: {decoder_hidden.shape}")

        break


# 训练参数
epochs = 100  # 训练轮数
lr = 1e-4  # 学习率
teacher_forcing_ratio = 0.5  # 教师强制比例
print_interval_num = 1000  # 打印间隔，每1000批打印一次
plot_interval_num = 1000  # 绘图间隔，每1000批绘制一次


# 批次训练迭代器
def train_iters(
    x,
    y,
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    optimizer_encoder,
    optimizer_decoder,
    criterion,
):
    """
    每批次训练流程
    :param x: 特征
    :param y: 标签
    :param encoder: 编码器
    :param decoder: 解码器
    :param optimizer_encoder: 编码器优化器
    :param optimizer_decoder: 解码器优化器
    :param criterion: 损失函数
    :return: 损失值
    """

    # 获取当前的 batch_size (如果是最后一点数据，可能不足 32)
    batch_size = x.size(0)

    # 1.编码阶段
    # 初始化编码器隐藏层
    encoder_hidden = encoder.init_hidden(batch_size)
    # 编码器前向传播
    encoder_output, encoder_hidden = encoder(x, encoder_hidden)

    # 2.解码阶段
    # 解码器的初始隐藏层
    decoder_hidden = encoder_hidden
    # 初始化输入，形状为 [batch_size, 1]
    y_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

    loss = 0.0  # 初始化损失值
    y_len = y.shape[1]  # 目标序列长度
    n_steps = 0  # 实际的时间步数

    # 根据概率决定是否使用 teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for i in range(y_len):
            # 解码器的前向传播
            output_y, decoder_hidden, attn_weights = decoder(
                y_input, decoder_hidden, encoder_output
            )
            # 目标标签：取所有句子的第 i 个词，形状 [batch_size]
            target_y = y[:, i]
            # 模型输出降维：从 [batch_size, 1, vocab_size] 变成 [batch_size, vocab_size]
            loss += criterion(output_y.squeeze(1), target_y)
            # 时间步+1
            n_steps += 1
            # 下一个时间步的输入：真实标签，形状 [batch_size, 1]
            y_input = y[:, i].unsqueeze(1)
    else:
        for i in range(y_len):
            # 解码器的前向传播
            output_y, decoder_hidden, attn_weights = decoder(
                y_input, decoder_hidden, encoder_output
            )
            # 获取目标值
            target_y = y[:, i]
            # 损失计算
            loss += criterion(output_y.squeeze(1), target_y)
            # 获取概率最大的词的索引和概率
            topv, topi = output_y.topk(1)
            # 时间步+1
            n_steps += 1

            # 批量训练时不能用 if 判断 EOS 提前退出，必须让所有句子都跑完 MAX_LENGTH
            # 批量提取下一个时间步的输入：从 [batch_size, 1, 1] 变成 [batch_size, 1]
            y_input = topi.squeeze(-1).detach()

    # 梯度清零 + 反向传播 + 参数更新
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()
    loss.backward()
    optimizer_encoder.step()
    optimizer_decoder.step()

    # 返回平均损失
    return loss.item() / max(n_steps, 1)


# 模型训练
def train_seq2seq(train_batch_size):
    # 1.获取数据加载器
    dataloader = get_dataloader(batch_size=train_batch_size)

    # 1.5.检查模型保存文件夹是否存在
    os.makedirs("./model", exist_ok=True)

    # 2.模型初始化
    encoder = EncoderRNN(len(english_word2index), 256).to(device)
    decoder = DecoderRNN(len(french_word2index), 256).to(device)

    # 3.优化器初始化
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=lr)

    # 4.损失函数初始化
    criterion = nn.CrossEntropyLoss()

    # 5. 定义统计信息变量
    loss_list = []  # 损失值列表
    start_time = time.time()  # 开始时间

    # 6.轮次训练
    for epoch in range(epochs):
        epoch_time = time.time()  # 本轮开始时间
        for i, (x, y) in enumerate(tqdm(dataloader), start=1):
            loss = train_iters(
                x, y, encoder, decoder, optimizer_encoder, optimizer_decoder, criterion
            )
            # # 每1000批打印一次
            # if i % print_interval_num == 0:
            #     print(
            #         f"epoch: {epoch + 1}, batch: {i}, loss: {loss}    本轮耗时: {time.time() - epoch_time:.2f}s"
            #     )
            # 每100批记录一次损失值
            if i % plot_interval_num == 0:
                loss_list.append(loss)

        # 每10轮保存模型参数
        if epoch+1 % 10 == 0:
            torch.save(encoder.state_dict(), f"./model/encoder_{epoch+1}.pth")
            torch.save(decoder.state_dict(), f"./model/decoder_{epoch+1}.pth")

        # 打印本轮信息
        print(
            f"第{epoch + 1}轮: loss: {loss_list[-1]}  本轮耗时: {time.time() - epoch_time:.2f}s  总耗时: {time.time() - start_time:.2f}s"
        )

    # 7.训练结束后绘制损失曲线
    # 自动生成 x 轴坐标（从 1 到 loss_list 的长度）
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker='o')
    plt.xlabel("记录点 (每 1000 个 batch)")
    plt.ylabel("Loss (平均损失)")
    plt.title("训练损失曲线")
    plt.grid(True)
    plt.show()


# 模型测试与评估
def evaluate_seq2seq(x, encoder, decoder):
    # 0.关闭梯度计算
    with torch.no_grad():
        # 1.编码阶段
        # 初始化编码器隐藏层
        encoder_hidden = encoder.init_hidden()
        # 编码器前向传播
        encoder_output, encoder_hidden = encoder(x, encoder_hidden)

        # 2.解码阶段参数准备
        # 解码器隐藏状态
        decoder_hidden = encoder_hidden
        # 解码器的初始输入
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3.自回归解码过程
        # 定义列表记录解码结果
        decoder_words = []
        # 初始化注意力矩阵
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH, device=device)
        # 开始逐词解码
        for i in range(MAX_LENGTH):
            # 解码器前向传播
            output_y, decoder_hidden, attn_weights = decoder(
                input_y, decoder_hidden, encoder_output
            )
            # 记录注意力矩阵
            decoder_attentions[i] = attn_weights.squeeze()
            # 预测的下一个词
            topv, topi = output_y.topk(1)
            # 检测到结束标记，停止生成
            if topi.squeeze().item() == EOS_token:
                break
            else:  # 否则，添加预测的词到预测词列表
                decoder_words.append(french_index2word[topi.squeeze().item()])
            # 将预测的词作为下一个时间步输入
            input_y = topi.squeeze(-1).detach()

        # 返回 解码结果 和 注意力矩阵
        return decoder_words, decoder_attentions[:i]


def dm_test_evaluate_seq2seq(
    encoder_dict_path, decoder_dict_path, batch_size=100, show_processing=False
):
    # 获取数据加载器对象
    dataloader = get_dataloader(batch_size=1)
    # 加载编码器模型
    encoder = EncoderRNN(len(english_word2index), 256).to(device)
    encoder.load_state_dict(
        torch.load(encoder_dict_path, map_location=device, weights_only=True)
    )
    # 加载解码器模型
    decoder = DecoderRNN(len(french_word2index), 256).to(device)
    decoder.load_state_dict(
        torch.load(decoder_dict_path, map_location=device, weights_only=True)
    )

    encoder.eval()
    decoder.eval()

    # 定义抽取数量
    batch_size_to_test = batch_size

    # 统计正确率
    total_words = 0
    correct_words = 0
    exact_match_count = 0

    print(f"\n开始评估模型 (共抽取 {batch_size_to_test} 条样本)...\n")

    # 抽取样本预测
    for i, (x, y) in enumerate(dataloader):
        # 抽取次数大于定义时中止预测
        if i >= batch_size_to_test:
            break

        # 预测过程
        decoder_words, decoder_attentions = evaluate_seq2seq(x, encoder, decoder)

        # 处理真实标签 (x 和 y 都是 Tensor，需要转回列表并映射成单词)
        # x[0] 是英文索引，y[0] 是法文真实索引
        input_indices = x[0].tolist()
        target_indices = y[0].tolist()

        # 将索引转回单词，并过滤掉 EOS 和补齐的 padding (这里用 EOS_token 作为填充，所以过滤它)
        input_words = [
            english_index2word[idx] for idx in input_indices if idx != EOS_token
        ]
        target_words = [
            french_index2word[idx] for idx in target_indices if idx != EOS_token
        ]

        # 打印直观结果
        if show_processing:
            print(f"样本 {i + 1}:")
            print(f"  英文原句: {' '.join(input_words)}")
            print(f"  真实法文: {' '.join(target_words)}")
            print(f"  模型预测: {' '.join(decoder_words)}")
            print("-" * 40)

        # --- 简单计算准确率 ---
        # 如果预测长度和真实长度一致，且每个词都一致，算作全对
        if decoder_words == target_words:
            exact_match_count += 1
        else:
            # 以真实的长度作为分母，对比预测出来的结果
            for pred_w, true_w in zip(decoder_words, target_words):
                if pred_w == true_w:
                    correct_words += 1
            total_words += len(target_words)  # 分母加上真实句子的长度

    # 打印最终的指标
    print(f"共抽取 {batch_size_to_test} 条样本")
    print("\n📊 评估结果:")
    print(f"  句子完全匹配率: {exact_match_count / batch_size_to_test * 100:.2f}%")
    if total_words > 0:
        print(f"  词级准确率 (Word Accuracy): {correct_words / total_words * 100:.2f}%")
    else:
        print("  词级准确率: 模型没能预测出任何对应单词。")


# 测试
if __name__ == "__main__":
    # 模型训练
    # train_seq2seq(train_batch_size=32)

    # 模型路径
    encoder_path = "./model/encoder_91.pth"
    decoder_path = "./model/decoder_91.pth"

    # 模型评估
    dm_test_evaluate_seq2seq(encoder_path, decoder_path, 100, show_processing=True)

# 第3讲Transformer 及大模型(1)

## Transformer 的核心思想

Transformer 使用自注意力，让序列中不同位置的信息相互作用；前馈网络则对每个位置的表示做进一步变换。与逐步处理序列的循环网络相比，许多计算可以并行完成。多头注意力从不同子空间提取关系，残差连接和归一化帮助训练深层模型。

图：输入 token 转成向量后，通过注意力计算位置之间的权重，再汇总上下文信息；多层堆叠形成可用于预训练和下游任务的表示。

## 为什么Transformers？
- 为什么Transformers模型在NLP自然语言处理任务中能
够力压群雄，变成SOTA模型的必备组件之一？
- Transformers结构解决了一个自然语言处理和计算机视觉领域
都困扰已久的问题——长期依赖

## 从例子看起
- Motivated example:
我昨天上学迟到了，老师批评了____。
- Should be 我, rather than 小明，吃饭…
- 语言模型
- 2-Gram: Search the most possible word after 了 in the
dictionary
- 3-Gram: Search the most possible word after 批评了 in the
dictionary
- 4-Gram,… 9-Gram----How about N for different sentences?
- So it seems that there should be ideally arbitrary grams

## 循环神经网络
Recurrent Neural Networks
- RNNs 指的是一系列处理序列信息的神经网络，
其输入和输出长度都是可变的
- 输入是向量x的序列 (x 指t时刻的向量) + 上一时刻
t
的历史信息 (HOW: 隐藏层上一时刻的状态)
- 输出是softmax层，预测接下来的激活

## Unrolling a Recurrent Neural Network
- 输入 (x): (example: word of a sentence)
- 隐状态 (s): function of previous
Memory
hidden state and new input
- 输出 (o): (example: predict next word
in the sentence)
Formulation
Recursively

## RNN Computational Graph
- 每个时刻重用相同的权重矩阵
- Parameters U, V, W are shared across time
- Similar to CNNs: this reduces the # of parameters we need to
optimize
- And it allow us to process arbitrary “temporal size” of input
- 适用于任意类型的输入/输出映射:

## Training in Recurrent NN
- 将时间上离散的RNN展开为多层前
馈网络 (FFNN)
- FFNN的每层对应一个时刻的序列信
息处理, 层数的下标为对应时刻
- 自顶向下 使用基本的反向传播训练
FFNN——Backpropagation though
time (BPTT)
- 将循环神经网络（RNN）看作一个在时
间上展开的多层前馈神经网络，其中
每一个前馈网络的“层”对应RNN中的
一个时间步
- 没有环：纵向为时间展开，横向是前
向传播
- 在每个时间步，同时沿横向和纵向传
播梯度

## Back propagation Through Time
- BPTT算法是针对循环层的训练算法，包含三个步骤：
1. 前向计算每个神经元的输出值；
2. 反向计算将整个序列数据的前向传播过程展开，然后对整个序列
的误差进行反向传播，从最后一个时间步的输出反向传递到第一
个时间步的输出
3. 计算每个权重的梯度：利用微积分中的链式法则，计算出损失函
数相对于网络中每个时间步的隐藏状态和权重（包括在不同时间
步共享的权重）的梯度。
4. 用随机梯度下降算法更新权重。

## Gradients Can Explode or Vanish
Error item through time in BP
So, we have
where β is the upper bound
of a matrix module
If t-k is very large, the value of this item increases rapidly or vanish quickly RNN 所谓梯度消失的真正含义是，梯度被近距离梯度主导，导致模型难以学到远距离的依赖关系

## Long-Term Dependency
- Short-term dependence:
Bob is eating an apple.
- Long-term dependence:
Context Bob likes apples. He is hungry and decided to
have a snack. So now he is eating an apple.
In theory, vanilla RNNs
can handle arbitrarily
long-termdependence.
In practice, it’sdifficult.

## LSTM
Pick What to Forget and What To
Remember
Bob and Alice are having lunch. Bob likes apples. Alice likes oranges.
She is eating an orange.
Conveyer belt for previous state and new data:
1. Decide what to forget (state)
2. Decide what to remember (state)
3. Decide what to output (if anything)

## LSTM: State, Gates
- Cell state(单元状态）
- 在整个链条上只经过一些细微
的线性变换
- 状态直接传到下一个
- Gates(门控）
- 门控的作用是选择性地使信息
通过.
- 3个sigmoid层的输出决定了信息
的通过(~1)与否 (~0)

## LSTM Summary
LSTM 单元维护记忆状态 $c_t$ 和输出状态 $h_t$。遗忘门 $f_t$ 决定保留多少旧记忆，输入门 $i_t$ 决定写入多少新候选信息，输出门 $o_t$ 决定暴露多少当前记忆：

$$
f_t=\sigma(W_f[h_{t-1};x_t]+b_f),\quad i_t=\sigma(W_i[h_{t-1};x_t]+b_i),\quad o_t=\sigma(W_o[h_{t-1};x_t]+b_o),
$$

$$
\tilde c_t=\tanh(W_c[h_{t-1};x_t]+b_c),\quad c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\quad h_t=o_t\odot\tanh(c_t).
$$

## GRU – A Variation on the LSTM
- A slightly more dramatic variation on the LSTM is the Gated

- GRU将遗忘和输入门限结合输入到单个“更新门限”中
- 同样还将单元状态和隐藏状态合并以降低复杂性
- The resulting model is simpler than standard LSTM models, and has
been growing increasingly popular.

## From RNN to Transformer
- RNN的问题
- 并行能力差：t时刻的计算依赖于t − 1时刻的计算结果，只能串行计
算，无法并行加速
- 长时间依赖：序列长度太长时，早期的信息很难被传递到后期
- Transformer的出现解决了上述问题
- 自Attention机制提出后，加入attention的Seq2seq模型在各个任务
上都有了提升，所以后来seq2seq模型指的都是结合RNN和
attention的模型
- 之后google又提出了解决Seq2Seq问题的Transformer模型，用全
attention的结构代替了LSTM，在翻译任务上取得了更好的成绩

## Transformer起源
Transformer 源自 Vaswani 等人的论文 *Attention Is All You Need*。其编码器和解码器使用多头注意力、位置编码、前馈网络、残差连接与层归一化；解码器在自注意力层使用掩码，并在编码器—解码器注意力层读取编码器输出。

## 内容提纲
01 为什么需要Transformer？
02 Transformer架构及原理
03 大模型与预训练
04 几个扩展主题

## Transformer架构
p使用Transformer实现机器翻译
- 输入一个句子，输出另一种语
言的句子
- Transformer的内部仍然是NLP
任务中最常用的编码器-解码
器结构

## 内部结构
p编码器和解码器都由六个小
的编/解码器堆叠而成
- 每个小的编/解码器的结构完全相同
，但参数并不共享
p编码器由注意力模块和前馈
神经网络两个子部分构成
p解码器同样有与编码器相同
的模块，但在中间加入了从编
码器到解码器的注意力模块，
Transformer的内部结构
有助于解码器关注输入序列中
的相关部分
- 没有RNN层
- 计算上没有时序上的依赖，可以并
行
编码器 解码器

## 编码器
Transformer 编码器先将输入 token 映射为词向量（示例使用 512 维），再依次经过自注意力层和逐位置前馈网络。自注意力让不同位置交换信息；前馈网络对每个位置独立计算，因此可并行处理。一个编码器层的输出作为下一层输入。

## Attention Mechanism
利用循环神经网络去处理 NLP 任务时，长距离“记忆”能力一直是个大难题，而引入“注意力机制”也能有效缓解这一问题。
传统的RNN中，y1、y2 和 y3 的计算都是基于同一个 C. 但这可能并不是最好的方案，因为 输入序列中的不同单词对 y1、y2 和 y3 的影响是不同的，因此很自然地
就有了如下思路：

注意力分两步：先按查询 $q$ 为输入 $X=\{x_1,\ldots,x_N\}$ 计算分布，再求加权平均：

$$
\alpha_n=p(z=n\mid X,q)=\mathrm{softmax}_n(s(x_n,q))=\frac{e^{s(x_n,q)}}{\sum_{j=1}^N e^{s(x_j,q)}},
$$

$$
\mathrm{att}(X,q)=\sum_{n=1}^N\alpha_nx_n=\mathbb E_{z\sim p(z\mid X,q)}[x_z].
$$

## Apply attention mechanism to model
language

图：自注意力用于句子 “The weather is nice today”：每个词对句中其他词计算注意力权重，并依据这些权重汇集上下文信息。

## Self-Attention
p示例：”The animal didn‘t cross
the street because it was too tired”
，其中it指代是The animal
pself-attention层通过将输入句子中
其他位置作为辅助信息，帮助网
络“理解”其中的指代（实际上
就是长时间的依赖关系），与
RNN的隐藏状态所起作用类似
p右图是训练好的网络，对self-
attention做可视化的结果，可以发
现 it 对 the animal的“注意力”最
集中

## Self-Attention：步骤(1)
1、在把各个词送入Self-Attention之前先将其乘以一个特征矩阵，以特征化的表示各个单词，然后将其送入Self-Attention中（词嵌入的方法，
Word embedding），即ai=Wxi
2、然后把不同的a乘上不同的矩阵W变为向量q（去匹配其他向量）、k
（被匹配）、v（要被抽取出的信息）

图中每个输入向量经线性变换生成查询、键和值：$a^i=W^x x^i$，$q^i=W^q a^i$，$k^i=W^k a^i$，$v^i=W^v a^i$。

## Self-Attention：步骤(2)
3、用每个向量q去对每个k做attention，这里公式除以根号d是为了平衡q和k乘的时候方差太大

第 $i$ 个查询与第 $j$ 个键的缩放点积得分为 $s_{ij}=(q^i\cdot k^j)/\sqrt{d_k}$；除以 $\sqrt{d_k}$ 有助于防止维度增大时点积方差过大。

## Self-Attention：步骤(3)
4、然后做Soft-max

对每个查询，将得分沿键维度做 softmax：$a_{ij}=e^{s_{ij}}/\sum_{j'}e^{s_{ij'}}$。

## Self-Attention：步骤(4)
5、得到输出的向量b1、b2、…

输出向量为值向量的加权和：$b^i=\sum_ja_{ij}v^j$。因此一个位置的输出能够综合整个输入序列的信息。

## Query-Key-Value (QKV) Model
查询—键—值（QKV）矩阵形式：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad H=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

## Self-Attention：矩阵实现
用矩阵运算来实现和加速

矩阵实现将所有 token 的线性投影、相似度计算和加权求和批量并行化，适合 GPU 加速：$Q=XW_Q$，$K=XW_K$，$V=XW_V$，$H=\mathrm{softmax}(QK^\top/\sqrt{d_k})V$。

## Self-Attention with Short-cut
- 采用了残差网络中的short-cut结构，目的是解决深度学
习中的退化问题

这里在自注意力模块外加残差连接与归一化，缓解深层网络的退化及训练不稳定：$Y=\mathrm{LayerNorm}(X+\mathrm{SelfAttention}(X))$。

## Multi-Head Attention
p使用多组𝑄, 𝐾, 𝑉
p扩展了模型专注于不同位置的能力
p赋予了注意力模块更多的子表达方式
Multi-Head
Attention相当于h个不
同的self-attention的
集成（ensemble）

多头注意力让不同头在不同的投影空间学习关系：

$$
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V),
$$

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.
$$

## Multi-Layer Self-Attention
多层自注意力将前一层输出作为下一层输入；较低层可捕捉局部关系，叠加后可形成更复杂的上下文表示。

## General attention layer
一般注意力层将输入投影为键和值，将查询与键对齐并计算上下文：

$$
k_j=x_jW_K,\quad v_j=x_jW_V,\quad e_{ij}=\frac{q_i\cdot k_j}{\sqrt{D}},\quad a_{ij}=\mathrm{softmax}_j(e_{ij}),\quad y_i=\sum_ja_{ij}v_j.
$$

## Self attention layer
**自注意力**将查询也从同一输入序列产生，即 $q_i=x_iW_Q$，随后按前述内容的对齐、softmax 和加权求和步骤得到每个位置的上下文表示。

## Masked self attention layer
**掩码自注意力**在 softmax 前将未来位置（$j>i$）的得分设为 $-\infty$，使位置 $i$ 只能读取自身及过去位置。这用于自回归生成，防止预测时看到未来 token。

## Multi-head self attention layer
多头自注意力将查询、键、值分别投影到多个子空间，每个头独立计算注意力；各头输出拼接后再线性变换。

## Self-Attention + Positional Encoding
由于在Self-Attention中没有考虑位置信息，所以在原始论文中加一个ei来表示位置信息，可理解为在xi向量上加了一个one-hot表示的pi，然后经过
计算发现ei并不影响原来的向量，也就是原来的信息不会被影响

位置编码与词嵌入逐元素相加：令 $a^i=W^Ix^i$，$e^i=W^Pp^i$（$p^i$ 表示第 $i$ 位的位置），输入注意力层的是 $a^i+e^i$。

## 时序信息编码(1)
RNN 通过按时间顺序输入数据并依赖前一隐藏状态，**隐式**地表示时序，但这种依赖限制并行计算。Transformer 将位置编码显式加入各 token 表示，使模型在并行处理整个序列时仍能区分词的位置和相对距离。

## 时序信息编码(2)
p位置编码方式：a. 根据数据学习；b. 自己设计编码规则
p规则式位置编码：
每个维度所对应的位置信息都是一个 每个维度所对应的位置信息都是基于某不变的常数：任意两个位置上的向量 种公式变换所得到：能够感知到位置信
都可以进行交换而模型却不能捕捉到 息变化（再也找不到与位置20处位置信这一差异（如位置20处的向量，无论你将 息相同的位置）
它换到哪个位置，都和原来一模一样）

## 时序信息编码(3)
p常数Positional Embedding：无效
原始输入在经过Token Embedding后，又加入了一个常数位置信息的的Positional Embedding。在经过一次线性变换后便得到了图右边所示的结果
序列在交换位置前和交换位置后计算得到的结果在本质上并没有任何区别，仅仅只是交换了对应的位置。因此需要加入Positional Embedding来刻画数据在时序上的特征

## 时序信息编码(4)
p非常数Positional Embedding：可以弥补自注意力机制不能捕捉序列时序信息的缺陷
在交换位置前与交换位置后，与同一个权重矩阵进行线性变换后的
结果截然不同

## 时序信息编码(5)
规则式正弦位置编码使用与词向量相同的维度 $d_{model}$：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

奇偶维分别使用正弦和余弦。利用和角公式，固定位置偏移 $k$ 对应的位置编码可由原位置编码作线性组合，有利于学习相对位置。

## 解码器
p与编码器类似，略有区别
p编码器的输出会被转换为一系列的注意力向量𝐾和𝑉，用于解码器中间的
encoder-decoder attention模块
p在self-attention中，计算第𝑝𝑜𝑠个输出时，仅用到[0, 𝑝𝑜𝑠)的词向量的信息：似乎是RNN的做派，但实际上通过带mask的self-attention来实现，计算第pos个
输出时，将其后面的数据置负无穷

## 完整的Transformer结构
Transformer
blocks

完整 Transformer 结构包括：输入嵌入与位置编码；编码器中的多头自注意力和逐位置前馈网络；解码器中的掩码自注意力、编码器—解码器注意力和前馈网络；各子层均配有残差连接与归一化，最后经线性层和 softmax 得到输出概率。

## Transformer计算复杂度分析
对于长度为 $T$、隐藏维度为 $D$ 的序列，完整自注意力的主要计算复杂度为 $O(T^2D)$，逐位置前馈网络约为 $O(TD^2)$。短序列时，较大的 $D$ 使前馈网络成为主要计算负担；长序列时，自注意力的 $T^2$ 项占主导。

## Position-based Sparse Attention
Atomic sparse patterns
Compound sparse patterns

## Transformer小结
- 优点：
- 其抛弃了在NLP中最根本RNN或CNN，取得了非常不错的效果
- 最大的带来性能提升的关键是将任意两个单词的距离变成1，
这对解决NLP中棘手的长期依赖问题是非常有效的。
- 不仅可以应用在NLP的机器翻译领域，甚至可不局限于NLP
- 算法的可扩展性、伸缩性和并行性非常好，符合目前的硬件（
主要指GPU）环境
- self-attention模型更可解释，attention结果的分布表明了该模
型学习到了一些语法和语义信息

- 缺点：
- 计算开销非常大
- 粗暴抛弃RNN和CNN虽然非常炫，但也使模型丧失了捕捉局部
特征的能力
- Transformer失去的位置信息其实在NLP中非常重要，而在特征
向量中加入Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷

## 内容提纲
01 为什么需要Transformer？
02 Transformer架构及原理
03 大模型与预训练
04 几个扩展主题

## 大模型的Scaling Law
训练时长 数据集规模 参数大小
模型越大、数据越多、训练越长，模型精度越好
- "Scaling Laws for Neural Language Models" (OpenAI, 2020)
首次系统研究了模型规模、计算量和性能的关系 发现了幂律缩放现象
"Training Compute-Optimal Large Language Models" (DeepMind, 2022) 提出了更优的参数量和训练数据量的配比关系 54

## 基于Transformer的大模型
时间线展示 2019—2023 年基于 Transformer 的大模型，例如 GPT-3、GShard、PanGu、LaMDA、Chinchilla、OPT、BLOOM、InstructGPT、ChatGPT、LLaMA 和 GPT-4；用于说明这一架构在语言模型中的持续扩展。

## 全球主要大语言模型比较
图中排行榜截面按 **Net Improvement** 排序，并同时列出 Confirmed Success、Praise vs Complaint、Steerability、Bash Recovery 五项变化率：

| 排名 | 模型 | Net Improvement | Confirmed Success | Praise vs Complaint | Steerability | Bash Recovery |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Claude Fable 5.1 (Max) | 14.51% | 23.19% | 39.46% | -1.21% | 10.71% |
| 2 | GPT 6 Astra (Max) | 12.55% | 21.85% | 40.61% | -5.46% | 5.33% |
| 3 | Claude Opus 5 (High) | 11.42% | 11.96% | 18.59% | 13.61% | 12.58% |
| 4 | Claude Opus 5 (Max) | 10.79% | 12.51% | 18.44% | 9.48% | 13.12% |
| 5 | Claude Fable 5 (High) | 9.15% | 6.46% | 18.75% | 11.00% | 9.17% |
| 6 | GPT 5.6 Sol (xHigh) | 7.75% | 5.13% | 19.52% | 8.77% | 4.93% |
| 7 | Claude Opus 4.8 (High) | 7.69% | 4.48% | 16.05% | 10.98% | 7.33% |
| 8 | Kimi K3 (Max) | 6.59% | 15.43% | 12.90% | -0.15% | 4.39% |
| 9 | Claude Sonnet 5 (High) | 6.31% | 1.04% | 12.40% | 9.74% | 8.13% |
| 10 | Hy4 preview | 5.37% | 12.00% | 8.98% | -1.92% | 7.52% |
| 11 | GPT 5.5 (xHigh) | 5.33% | 0.13% | 8.97% | 7.21% | 9.97% |
| 12 | GLM 5.2 (Max) | 4.64% | 8.06% | 11.03% | 2.92% | 0.79% |
| 13 | DeepSeek V4 Pro (High) (0813) | 4.43% | 11.13% | 6.03% | -1.09% | 5.68% |
| 14 | Gemini 3.8 Flash (High) | 4.01% | 8.84% | 13.46% | -3.34% | 0.79% |
| 15 | Grok 4.5 | 3.93% | 5.05% | 2.52% | 5.31% | 6.40% |
| 16 | Qwen 3.8 Max | 3.90% | 9.65% | 4.32% | 2.76% | 3.69% |
| 17 | Grok 4.6 (xHigh) | 3.24% | 4.92% | 2.98% | 2.44% | 5.49% |
| 18 | GPT 5.5 | 2.95% | -0.96% | 4.66% | 3.99% | 6.68% |
| 19 | GLM 5.3 (Max) | 2.56% | 11.04% | 5.36% | -1.04% | -2.94% |
| 20 | DeepSeek V4 Flash (High) | 2.22% | 6.74% | 4.64% | -1.43% | 0.78% |
| 21 | GLM 5.3 Flash | 2.00% | 12.59% | 3.18% | -1.13% | -5.05% |

图中以红框标出第 14 名 Gemini 3.8 Flash (High)。图中的数值是特定评测时点的结果，应按相应的评测时间与口径理解。

上半图比较 2023 年 5 月至 2026 年初美国与中国头部模型的 Arena 分数。美国曲线从 GPT-4-0314 的约 1310 分上升到 Claude Opus 4.6 的 **1503 分**；中国曲线从 ChatGLM-6B 的约 980 分上升到 Dola-seed-2.0-preview 的 **1464 分**。图中还标注了 Qwen-14B-Chat、GLM-4-0520、o1-preview、DeepSeek-R1、Qwen3-235B-A22B-Instruct-2507 和 Gemini-3-Pro 等阶段性模型，显示两条曲线的差距明显收窄。

下半图按对数坐标展示 2018—2025 年部分中美模型训练算力（petaFLOP-day）随发布时间的变化。标注模型包括 GPT-3 175B (davinci)、ERNIE 3.0 Titan、GPT-4 (Mar 2023)、Claude 3.5 Sonnet、Grok-2、Doubao-pro、Qwen2.5-72B、DeepSeek-V3、Qwen3-Max、GLM-4.6 与 Grok 4。总体趋势是训练算力跨越多个数量级增长，同时中国模型逐渐进入高算力区间。

## 大模型：革命性及其智能涌现所表现出的惊人行为
大模型可描述为规模足够大、训练充分且出现智能涌现的深度学习系统。所列关键能力包括**情境学习**（ICL）、**思维链推理**（CoT）与**自然指令学习**；这里强调其对生产效率与通用智能研究的影响。

## 智能涌现
智能涌现
参数量
能性tohs-0
当参数量大于68B时，
通过学习与人交流突然
涌现出了智能行为
可实现针对没见过的任
务的泛化性能
智能涌现的尺度率

## 预训练模型的模型架构
- Encoder-only (BERT)
- Pre-training : Masked Language Modeling (MLM)
- Great for classification tasks, but hard to do generation
- Decoder-only (GPT)
- Pre-training: Auto-regressive Language Modeling
- Stable training, faster convergence
- Better generalization after pre-training
- Encoder-decoder (T0/T5)
- Pre-training : Masked Span Prediction
- Good for tasks like MT, summarization

## BERT
- 采用Transformer的编码器（bidirectional）作为基础结构
- 全称是Bidirectional Encoder Representation from Transformers，即双向
Transformer的Encoder。
- 主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence
Prediction两种方法分别捕捉词语和句子级别的representation。
- 与Transformer相比，拥有更大的前馈神经网络和更多的attention head

## Embedding
- Token Embeddings：词向量，第一个单词是CLS标志，可以用于之
后的分类任务
- Segment Embeddings：用来区别两种句子，因为预训练不光做语
言模型还要做以两个句子为输入的分类任务
- Position Embeddings：和之前的Transformer不一样，不是三角函
数而是学习出来的

## 预训练
- Masked Language Model（句子中词的关系）
- 所谓MLM是指在训练的时候随机从输入预料上mask掉一些单词，然
后通过的上下文预测该单词，该任务非常像我们在中学时期经常做的
完形填空。
- 随机mask每个句子中15%的词，用其上下文做预测
- 例如：my dog is hairy → my dog is [mask]。实际处理为80%几率采用
[mask]，10%随机用一个词替代[mask]，10%保持不变
- 一定概率用随机词是为了防止模型最终学习得到[mask]就是hairy

- Next Sentence Prediction（句子之间的关系）
- 在有些任务中，比如问答，前后两个句子有一定的关联关系，我们希
望BERT Pretraining的模型能够学习到这种关系。因此BERT还增加了一个新的任务——预测两个句子是否有关联关系。
- 输入句子A和B，B有一半几率是A的下一句，其余的则从语料库中随
机选取

## BERT Finetuning
输入为两个序列的
普通的分类任务：输入是
任务：两序列的
一个序列，如图所有的
Token对应不同的
Token都是属于同一
Segment(Id=0/1)，
Segment(Id=0)，用第一个
用第一个特殊Token
特殊Token [CLS]的最后一
[CLS]的最后一层输
层输出接上softmax进行分
出接上softmax进行
类，用分类的数据来进行
分类，然后用分类
Fine-Tuning
数据进行Fine-
Tuning
序列标注：输入是一个句
问答：输入是一个问
子(Token序列)，除了
题和一段包含答案的
[CLS]和[SEP]的每个时刻
长文字(Paragraph)，
都会有输出的Tag，然后
输出在这段文字里找
用输出的Tag来进行Fine-
到问题的答案
Tuning

## GPT
- GPT(Generative pre-trained transformer)，即生成式的
预训练
- GPT也采用两阶段过程：非监督的预训练和监督方式的
微调

- 与BERT模型的不同
- 采用Transformer的解码器作为基础结构，即单向Transformer
- 预训练方式是通过上文预测下一个单词，而不像BERT同时利
用上下文的信息
- GPT2和GPT3舍弃第二阶段的fine-tuning，直接无监督的完成下
游任务，因此GPT2和GPT3能够完成生成式任务，而BERT不能

## 从GPT2到GPT3
- 事实上，GPT2和GPT3都是在前者基础上，采用了更多
参数的模型和更大的数据集进行训练后得到的扩展
- GPT3的参数量达到了1750亿，足足是GPT2的15亿的116
倍，用于训练的数据达到了45TB
- GPT3更聚焦于通用的NLP模型，目标在于用更少的特定
domain的数据，且不经过fine-tunning也能很好地解决
特定任务

## 预训练模型的计算结构
预训练 Transformer 的主要计算位于嵌入层、注意力层和前馈网络，核心运算多为矩阵乘法。这里比较稠密大模型与混合专家模型（MoE）：前者直接扩大所有参数，后者用门控网络选择部分小模型或专家，并对激活专家的输出加权汇合。

## 混合专家模型 MoE
**混合专家模型（MoE）**使用门控网络将输入路由到若干专家，目的是让不同专家专注于不同数据子集或任务情形。其输出可表示为 $y=\sum_{i\in\mathrm{TopK}(G(x))}\tilde G_i(x)E_i(x)$。

典型 MoE 层先以 $G(x)=\mathrm{softmax}(W_Gx)$ 得到各专家的门控分数，再选得分最高的 $K$ 个专家，并对它们的输出加权求和。例如 Mixtral-8×7B 从 8 个专家中激活 2 个。专家通常由前馈网络构成。

将 Transformer 块内的普通前馈网络（FFN）替换为 MoE 层：注意力层继续处理 token 间信息，路由器为各 token 选择专家前馈网络。这样可增加总参数量而限制单次激活的参数量。

## 混合专家模型(MoE)演进
时间线展示 MoE 架构从早期方法到 Switch Transformer、DeepSpeed-MoE、Mixtral、DeepSeek-MoE/DeepSeek-V2/V3、Qwen 等模型的发展，并列举视觉、多模态、推荐系统等应用。

## MoE有望延续Scaling Law
- CNN结构在2021年遇到瓶颈
- Dense TF结构目前也遇到性能瓶颈
- MoE结构的专家稀疏度是Dense结
构的性能杠杆
- Attention结构中的动态输入响
应为后续的FFN结构带来第一
个性能能力指数增益
- MoE的动态专家选择相当于在
Dense模型基础上引入第二个
性能指数增益

## 内容提纲
01 为什么需要Transformer？
02 Transformer架构及原理
03 大模型与预训练
04 几个扩展主题

## （1）如何在视觉任务中使用
Transformer中的Attention机制？

## ViT (Vision Transformer)
- 将Transformer架构用于视觉任务的开创工作
- 将输入图片划分为固定尺寸的小块（patch）
- 用线性层将patch映射为token embedding
- 加入positional embedding后，输入标准的Transformer Encoder进行序列特征提取
- 在大规模数据训练下，性能超越CNN

## Pyramid ViT
- 将金字塔结构首次引入Vision Transformer
- 在每个Stage中通过Patch Embedding逐渐降低分辨率
- 在分割等任务上取得了比ViT更好的性能

## Swin Transformer
- Hierarchical + Shifted Windows机制，解决了 ViT 计算复杂度高的问题
- 将token分为若干个窗口，每个token只在同一个窗口内与其它token计算Self Attention
- 为解决局部窗口视野受限的问题，采取了Hierarchical结构，每个token的视野随着 stage
增加逐渐变大
- 为解决窗口之间的信息交互问题，采取了Shifted Windows机制
- 性能全面超越ViT和CNN的SOTA模型，ICCV 2021 Best Paper （ Marr Prize ）

- Window Attention
- 仅仅计算每个窗口内的tokens的Attention
- 问题：每个Tokens仅与同一窗口内的其他Tokens交
互，没有跨窗口通信。

- Shifted Window Attention
- 仅仅计算每个窗口内的tokens的Attention
- 交替使用Windows Attention和Shifted Windows Attention
- 但Shifted Windows Attention会导致Windows大小不一致，且
Windows数目增多

- Shifted Window Attention
- 仅仅计算每个窗口内的tokens的Attention
- 交替使用Windows Attention和Shifted Windows Attention
- 使用Padding方式填补边缘Windows，并用mask来确保只在除
padding的部分做Attention计算（A部分一起计算Attention）。
- 通过移位特征图来保证Windows数目一致。
- 缺点：Operator较为复杂，难以部署

## （2）如何在多模态数据(图像和文本)
中使用Transformer?

## Visual-Language Pretraining
- Vision Encoder
- Text Encoder
- Multimodal Fusion
- Co-attention
- Merged attention
- Decoder(optional)

## Vision-and-language BERT
vision-language co-attention
ViLBERT的双流结构。视觉特征(绿色分支)和文本特征(紫色表示)都由一系列的transformer blocks(TRM)和co-attention transformer layers(Co-attention)组成。双流间有信息交互的层数是严格限定的(这里共交互k次)
训练ViLBERT采用了两个预训练任务：
预测图文是否对应
预测掩模区域的对象类别
J. Lu, D. Batra, D. Parikh, S. Lee, ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-

## CLIP(1)
- Connecting Text and Images
- 两个编码器，将image和text映射到统一的多模态嵌
入空间
- 收集更多的image-text数据获得更好性能

## CLIP(2)
- 图像编码器细节
- A modified ResNet (e.g., RN50, RN101) with attention pooling
- A Vision Transformer (ViT) with additional layer normalization
- 文本编码器细节
- Transformer架构: 12 layers, 63M parameters, 8 attention heads
- GPT-3: 96 layers, 175B parameters, 96 attention heads, 12,288
hidden dimension, 2,048 token context window.
- 统一嵌入空间
- 两个编码器均生成经过 L2 归一化的特征向量。
- 线性投影将这些特征映射到一个共享的嵌入空间。
- 余弦相似度（经由学习得到的参数）用于衡量图像与文本的匹
配度。

## CLIP(3)
- Contrastive loss: 匹配正确的图文对，并将错误的图文
对区分开来
- 在包含 N 个配对的批次中，存在 N² 种潜在的比较组合
- 针对图像-文本及文本-图像的预测，采用对称交叉熵损
失。

CLIP 使用对称的图文对比损失。给定图像和文本的相似度矩阵 $s_{ij}$，图到文和文到图的目标分别是 $-\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$ 与 $-\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}$，最终损失取两者平均。

## GLIP
- Grounded Language-Image Pretraining
- 短语定位(phrase grounding)和目标检测(object detection)等价
- 文字-重点区域对（word-region pair）
- 深度融合（deep fusion）对两个模态的特征向量计算交叉注意
力

## Token: 大模型的基本信息单位
- 作用：原子化的信息度量（类似日常中的米/克），是衡量信息
体量、资源消耗、处理能力的核心标尺
- 跨模态统一表述：文本（词元）、图像（图元）、3D（体元）
- 意义：大模型的训练与推理过程本质是预测下一个Token
Token的切分过程：可以去除歧义 大模型的预测下一个Token训练过程

## 大语言模型中的 Token: 语言信息的词元
- 词元是大语言模型处理文本的最小离散语义单元，并不等同于汉
字、词语或英文单词，而是经分词算法拆分形成的标准化片。词
元化将自然语言转化为模型可计算的序列，支撑语义理解与文本
生成，也是计费、速度和上下文容量评估的重要依据。

## 图像模型中的 Token: 视觉信息的图元
- 图元将连续的像素画面切分为固定大小的 Patch，并通过线性投
影和位置编码转化为模型可运算的向量序列。每个图元承载局部
区域的色彩、纹理、轮廓和位置信息，使模型能够在统一的
Transformer 架构下理解图像的局部细节与整体空间关系。
Patch 分块 向量序列
图像

## 视频模型中的Token: 时空信息的视频基元
- 视频基元是兼顾空间特征与时间序列的最小复合信息单元，实
现对视频时空特征的统一量化与处理 。高端视频大模型会区分
对象基元与上下文基元，前者承载核心物体的动态变化与形态
特征，后者承载背景场景的空间信息。
两类信息协同生成连续时空逻辑
驱动更准确的视频理解与生成

## （3）DeepSeek 中的 Multi Latent
Attention 机制

## 多头潜在注意力
(Multi-head Latent Attention, MLA）
- 核心思想：通过低秩矩阵联合压缩（Low-Rank Joint
Compression）来大幅缩减大模型在推理时的 KV Cache（键值缓存）占用，从而打破大语言模型长文本推理的
“显存墙”。
- 从 Attention 的公式 可以看到，在计算Attention时
， 位置 t的 q 只与t位置前的k、v 做计算，所以有如下
两个结论：
- 计算前面的 k，v 并不受后面的影响
- 后面计算 t+1，t+2，…, t+n 位置的 Attention，要使用前序的 1-
>t 位置的 k,v 的值是始终不变的

## 利用KV-Cache 计算加速推理
- KV-Cache
- 为了避免重复计算前序的
kv，研究者们提出把前序
计算好的k，v缓存起来，这
也就是目前主流的KV-cache
的机制
- KV-Cache 是一种空间换时
间的机制。但是当文本长度
过长时，GPU 的显存空间
是比较宝贵的，通过显存来
保存KV-cache势必会带来访
存的瓶颈

## 减小KV-Cache 方法
- 共享KV：多个Head共享使用1组KV，将原来每个Head一个
KV，变成1组Head一个KV来压缩KV的存储。代表方法： GQA
，MQA等
- 窗口KV：针对长序列控制一个计算KV的窗口，KV cache只
保存窗口内的结果（窗口长度远小于序列长度），超出窗
口的KV会被丢弃，通过这种方法能减少KV的存储，当然也
会损失一定的长文推理效果。代表方法： Longformer等
- 量化压缩：基于量化的方法，通过更低的Bit位来保存KV，
将单KV结果进一步压缩，代表方法： INT8等
- 计算优化：通过优化计算过程，减少访存换入换出的次数
，让更多计算在片上存储 SRAM进行，以提升推理性能，代
表方法：flashAttention等

## MLA 的核心数学机制：低秩压缩
MLA（Multi-head Latent Attention）用低秩联合压缩减小 KV cache。对位置 $t$ 的隐藏状态 $h_t$，先计算低维潜向量 $c_t^{KV}=W^{DKV}h_t$ 并缓存；使用时再上投影得到 $k_t^C=W^{UK}c_t^{KV}$、$v_t^C=W^{UV}c_t^{KV}$。Query 侧可类似地压缩为 $c_t^Q$，再上投影生成查询。潜向量维度远小于完整多头键值维度。

## 工程优化：矩阵吸收（Absorb）
**矩阵吸收**利用乘法结合律避免在推理时显式恢复完整的键和值。例如

$$
q_t(k_t^C)^\top=q_t(W^{UK}c_t^{KV})^\top.
$$

固定的 $W^{UK}$ 可预先并入 Query 投影，$W^{UV}$ 可并入输出投影，从而直接利用缓存的低维 $c_t^{KV}$ 计算注意力。

## 如何进行模型蒸馏?

## 知识蒸馏(Knowledge distillation, KD)的概念
- Knowledge distillation (KD): a model compression method
in which a small model is trained to mimic a pre-trained, larger model (or ensemble of models).
- 起源：文献[1]提出，文献[2]扩展泛化
- 这种训练设置有时被称为“师生”（Teacher-student）模式，其中
大模型充当教师，小模型充当学生
- 在蒸馏过程中，通过最小化一个损失函数，知识从教师模型传递
给学生模型；该损失函数的目标是教师模型预测的类别概率分布
- 具体而言，知识蒸馏（KD）是通过最小化教师模型与学生模型预
测结果之间的 KL 散度来实现的
1Cristian Buciluˇa, Rich Caruana, and Alexandru Niculescu-Mizil (2006). “Model compression”. In: Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data
mining, pp. 535–541.
2Geoffrey Hinton, Oriol Vinyals, and Jeff Dean (2015). “Distilling the knowledge in a neural network”. In NIPS 2014 Deep Learning Workshop

## KL散度
KL 散度衡量两个概率分布的差异。离散形式和连续形式分别为

$$
D_{KL}(P\|Q)=\sum_{x\in\mathcal X}P(x)\log\frac{P(x)}{Q(x)},\qquad D_{KL}(P\|Q)=\int P(x)\log\frac{P(x)}{Q(x)}\,dx.
$$

## KL损失
知识蒸馏让学生网络拟合教师网络产生的软类别分布。设教师和学生 logits 分别为 $y_i,z_i$，温度为 $\tau$，则

$$
p_i=\frac{e^{y_i/\tau}}{\sum_{j=1}^{C}e^{y_j/\tau}},\qquad q_i=\frac{e^{z_i/\tau}}{\sum_{j=1}^{C}e^{z_j/\tau}},\qquad \mathcal L_{KD}=-\sum_{i=1}^{C}p_i\log q_i.
$$

## 蒸馏方法（1）：离线蒸馏
- 离线知识蒸馏：知识从预训练的教师模型迁移到学生模
型。
- 在进行蒸馏之前，首先利用一组训练样本对大型教师模型进行
训练。
- 在蒸馏过程中，利用该教师模型提取以 Logits 或中间特征形式
呈现的知识，并用这些知识来指导学生模型的训练。

## 蒸馏方法（2）：在线蒸馏
- 在线知识蒸馏：教师模型和学生模型均进行更新，且整
个知识蒸馏框架支持端到端训练。。

## 蒸馏方法（3）：自蒸馏
- 自蒸馏：教师模型和学生模型使用相同的网络。这可以
被视为在线蒸馏的一种特例。

## 大模型蒸馏
- 利用LLM作为教师的蒸馏方法：将LLM的涌现能力（EA）提炼成小
语言模型（SLM），分为两类：标准 KD 和基于 EA 的 KD
- 标准KD旨在使学生模型学习LLM所拥有的常见知识，如输出分布和特
征信息。
- 基于 EA 的 KD 不仅仅迁移 LLM 的常识，还包括蒸馏他们的涌现能力。
涌现能力包含三个方面，包括上下文学习 (ICL)、思维链 (CoT) 和指令遵循 (IF)。

## 上下文学习蒸馏
- ICL 采用结构化自然语言提示，其中包含任务描述以及可能的一些
任务示例作为演示。通过这些任务示例，LLM可以掌握并执行新任
务，而无需显式梯度更新。
- ICL 蒸馏：将上下文小样本学习和语言建模功能从 LLM 转移到
SLM，通过将上下文学习目标与传统语言建模目标相结合来实现
- 元上下文调优 (Meta-ICT) ：语言模型使用上下文学习 objectives 在不
同任务中进行元训练，使其能够通过上下文学习来适应看不见的任务
，从而扩展其解决问题的能力
- 多任务上下文调优 (Multitask-ICT)：使用 ICL objectives 和 目标任务中
的一些示例对模型进行微调，再用上下文学习来对这些任务进行预测

## 思维链蒸馏
**思维链（CoT）蒸馏**把教师大模型生成的中间推理步骤与最终答案一并用作学生模型训练数据，而非只给输入—输出对。可采用利用教师解释进行多任务训练、采样多条推理路径扩充训练集等方式，使较小模型学习分步推理和解释能力。

## 指令遵循蒸馏
- IF 致力于仅基于阅读任务描述来增强语言模型执行新任务的能力
，而不依赖于少数样本。通过使用一系列以指令表示的任务进行
微调，语言模型展示了准确执行以前未见过的指令中描述的任务
的能力。
- Lion 利用LLM的适应性强的特点来提高学生模型的表现。它提
示LLM识别并生成“hard”指令，然后利用这些指令来增强学生
模型的能力。
1) Lion: Adversarial
闭源模型的知识蒸馏
Large Language Model

## Question？

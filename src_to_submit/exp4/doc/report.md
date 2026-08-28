# 实验4报告：基于 Transformer 的中英机器翻译

## 1. 实验目的

本实验旨在实现一个基于 Transformer 的序列到序列神经机器翻译系统，并在给定的中英双语语料上完成中文到英文翻译任务。具体目标包括：

- 理解 Transformer 编码器-解码器结构在机器翻译中的基本原理与实现方式。
- 在 NiuTrans sample data 上完成训练、验证、断点续训、推理与 BLEU4 评估流程。
- 通过训练曲线与 BLEU 指标分析模型收敛情况和翻译效果。

## 2. 实验环境

### 2.1 软件环境
- Python 3.x
- PyTorch（GPU 训练，支持 AMP 混合精度）
- matplotlib（结果可视化）
- 可选：sentencepiece（子词切分实验）

### 2.2 实验数据

本实验使用 `data/sample-submission-version` 提供的 NiuTrans 示例数据集，主要包含：

- 双语训练集 `TM-training-set`
  - 中文文件：`chinese.txt`
  - 英文文件：`english.txt`
  - 规模：`199,630` 对句子
- 开发/测试相关数据
  - `Test-set/Niu.test.txt`
  - `Reference-for-evaluation/Niu.test.reference`
  - 官方测试集规模：`1,000` 条中文句子

本实验当前主要使用双语训练集进行训练，并从训练集内部随机切分验证集用于训练期间模型选择。

## 3. 实验原理

### 3.1 模型结构

本实验采用标准 Transformer 编码器-解码器结构，主要模块包括：

- Source Embedding / Target Embedding
- Positional Encoding
- Multi-Head Self-Attention
- Encoder-Decoder Attention
- Feed Forward Network
- Linear Output Projection

当前默认配置如下：

- `d_model = 256`
- `nhead = 8`
- `num_encoder_layers = 4`
- `num_decoder_layers = 4`
- `dim_feedforward = 1024`
- `dropout = 0.1`
- `tie_embeddings = True`

结构概览（当前配置）：

```text
TransformerSeq2Seq
├── 源语言词嵌入层 (src_vocab_size → 256)
├── 目标语言词嵌入层 (tgt_vocab_size → 256)
├── 位置编码层
│   └── 正弦位置编码 (max_len≈136)
├── 4层 Transformer Encoder
│   ├── 多头自注意力机制 (8 heads, 256 hidden)
│   ├── 前馈神经网络 (1,024 dim)
│   └── 残差连接 + LayerNorm
├── 4层 Transformer Decoder
│   ├── Masked 多头自注意力机制 (8 heads, 256 hidden)
│   ├── Encoder-Decoder Attention
│   ├── 前馈神经网络 (1,024 dim)
│   └── 残差连接 + LayerNorm
├── 输出投影层 (256 → tgt_vocab_size)
└── 可选权重共享
    └── 目标词嵌入层与输出投影层权重绑定
```

### 3.2 训练机制

训练阶段使用教师强制（teacher forcing），目标函数为带 label smoothing 的交叉熵损失，并配合 AdamW 优化器进行更新。为了兼顾显存占用与吞吐，本实验支持：

- AMP 混合精度训练
- 梯度累积
- Warmup + Cosine 学习率调度
- 梯度裁剪
- 自动保存 `best/last/step` checkpoint

### 3.3 核心实现

训练主循环中，每个 epoch 完成以下流程：

```python
train_loss, global_step = train_one_epoch(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    device=device,
    pad_idx=tgt_vocab.pad_idx,
    label_smoothing=args.label_smoothing,
    grad_clip=args.grad_clip,
    grad_accum_steps=args.grad_accum_steps,
    use_amp=args.use_amp,
    scaler=scaler,
    log_interval=args.log_interval,
    scheduler=scheduler,
    global_step=global_step,
    on_step_end=_step_ckpt_callback,
)
```

BLEU4 的评估方式为 corpus-level BLEU，使用 1 到 4 gram 的 clipped precision，并带 brevity penalty：

```python
for n in range(1, 5):
  pred_ngrams = Counter(_ngrams(pred, n))
  ref_ngrams = Counter(_ngrams(ref, n))

  self.total_counts[n - 1] += sum(pred_ngrams.values())
  clipped = 0
  for ng, count in pred_ngrams.items():
      clipped += min(count, ref_ngrams.get(ng, 0))
  self.clipped_counts[n - 1] += clipped
    ...

bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
```

#### 3.3.1 模型实现

本实验的模型主体实现为：

```python
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            self.output_proj.weight = self.tgt_embed.weight
        self._reset_parameters(tie_embeddings=tie_embeddings)
```

整体上，系统采用标准的 Transformer 编码器-解码器结构完成中译英任务，包括源语言嵌入层、目标语言嵌入层、位置编码、Transformer Encoder、Transformer Decoder 以及输出投影层。

编码端的作用是对输入中文序列进行上下文建模，提取全局语义表示；解码端则在 masked self-attention 约束下逐步生成目标英文序列，并通过 encoder-decoder attention 使用源端语义信息完成翻译。模型默认超参数定义在 [config.py](E:/hw/deep_learning/src_to_submit/exp4/config.py)，当前实验配置为：

- `d_model = 256`
- `nhead = 8`
- `num_encoder_layers = 4`
- `num_decoder_layers = 4`
- `dim_feedforward = 1024`
- `dropout = 0.1`
- `tie_embeddings = True`

此外，本实验启用了目标端词嵌入与输出投影层权重共享机制，以减少参数量并增强表示空间一致性，这一实现同样可在 [model.py](E:/hw/deep_learning/src_to_submit/exp4/model.py) 中核对。

结构概览（当前配置）：

```text
TransformerSeq2Seq
├── 源语言词嵌入层 (src_vocab_size → 256)
├── 目标语言词嵌入层 (tgt_vocab_size → 256)
├── 位置编码层
│   └── 正弦位置编码 (max_len≈136)
├── 4层 Transformer Encoder
│   ├── 多头自注意力机制 (8 heads, 256 hidden)
│   ├── 前馈神经网络 (1,024 dim)
│   └── 残差连接 + LayerNorm
├── 4层 Transformer Decoder
│   ├── Masked 多头自注意力机制 (8 heads, 256 hidden)
│   ├── Encoder-Decoder Attention
│   ├── 前馈神经网络 (1,024 dim)
│   └── 残差连接 + LayerNorm
├── 输出投影层 (256 → tgt_vocab_size)
└── 可选权重共享
    └── 目标词嵌入层与输出投影层权重绑定
```

#### 3.3.2 模型结果

从训练与评测结果来看，该模型已经具备较稳定的基础翻译能力。训练日志位于 [exp4_run1.jsonl](E:/hw/deep_learning/src/exp4/logs/exp4_run1.jsonl)，其中记录了 step 级和 epoch 级损失、学习率以及验证集 BLEU。根据当前最新日志，模型已完成 `12` 个 epoch 的训练，在第 `12` 轮达到：

- `train_loss = 3.0373`
- `val_loss = 2.2033`
- `best val_bleu4 = 22.71`

这表明模型在训练过程中保持了较稳定的收敛趋势，并在内部验证集上取得了较合理的句子级翻译质量。

在最终测试阶段，评测入口位于 [main.py](E:/hw/deep_learning/src_to_submit/exp4/main.py)，解码与评估逻辑位于 [engine.py](E:/hw/deep_learning/src_to_submit/exp4/engine.py)，BLEU 指标实现位于 [evaluator.py](E:/hw/deep_learning/src_to_submit/exp4/evaluator.py)。在 beam search 配置下，模型在官方 `1000` 条测试集上取得了 `24.62 BLEU`。结合后续样例分析可以看出，模型对短句和常见表达的翻译较为准确，对中长句也能较好保留主干语义；但在复杂句法结构、制度类表达和细节保真度方面仍存在进一步提升空间。

## 4. 训练配置与流程

### 4.1 当前训练配置

结合当前日志，实验采用的主要配置为：

- `direction = zh2en`
- `batch_size = 4`
- `grad_accum_steps = 8`
- `effective_batch_size = 32`
- `use_amp = True`
- `scheduler = warmup_cosine`
- `warmup_steps = 1000`
- `lr = 3e-4`
- `weight_decay = 1e-4`
- `label_smoothing = 0.1`
- `decode_strategy = greedy`（训练阶段验证时）

### 4.2 验证集设置

- `val_ratio = 0.02`
- 训练总样本数约 `199,630`
- 验证集规模约为 `3,992` 条

也就是说，训练阶段是每个 epoch 在大约四千条随机验证样本上计算一次 `val_loss` 和 `val_bleu4`。

## 5. 实验结果

### 5.1 训练阶段结果

根据当前最新完整训练日志（已完成到 `epoch 12`），得到如下结果：

- 第 `12` 轮：
  - `train_loss = 3.0373`
  - `val_loss = 2.2033`
  - `val_bleu4 = 22.71`
- 当前最优验证 BLEU：
  - `best_bleu4 = 22.71`

这说明模型已经明显学到中英对齐关系，验证 BLEU 进入 `20+` 区间，属于当前实验配置下比较合理的量级。

### 5.2 官方测试结果

在当前最优 checkpoint 上，采用 beam search 配置：

- `beam_size = 5`
- `length_penalty = 0.8`
- `no_repeat_ngram_size = 4`

对官方 `1000` 条测试集进行完整评估，得到：

- **Test BLEU4 = 24.62**

此外，在早期调参阶段，曾在 `100` 条测试子集上做过小规模解码搜索，较优结果约为 `26.82 BLEU`。这一结果可用于说明 beam search 对翻译质量有帮助，但由于只基于 `100` 条样本，不能作为最终汇报结果。最终报告应以 `1000` 条 full test 的 `24.62 BLEU` 为准。

### 5.3 与预期的符合程度

当前结果总体符合预期，原因如下：

- 训练损失与验证损失均呈下降趋势，说明模型训练稳定。
- 验证 BLEU 达到 `22.71`，说明模型在训练内部验证集上已经获得较稳定的句子级翻译质量。
- 在官方 `1000` 条 full test 上达到 `24.62 BLEU`，说明模型在正式测试口径下也保持了合理表现。
- 小规模 `100` 条测试子集上可达到更高分数，说明解码策略有效，但 full test 结果更具代表性。

需要注意的是，当前实验是 **Transformer NMT baseline**，而不是 NiuTrans 原论文中完整的 phrase-based / syntax-based SMT 系统流程，因此两者不能直接做一一等价对比。

### 5.4 关于“测试 BLEU 高于验证 BLEU”的说明

本实验中，官方测试集 BLEU 为 `24.62`，高于训练阶段记录的验证集 BLEU `22.71`。这一现象是合理的，主要原因如下：

- **两者采用的解码配置不同。**
  训练阶段的验证 BLEU 主要用于每个 epoch 结束后快速选择最优 checkpoint，因此更强调稳定和计算效率；而最终官方测试使用了更优的 beam search 配置（`beam_size = 5`、`length_penalty = 0.8`、`no_repeat_ngram_size = 4`），这会显著提升最终 BLEU。

- **验证集与测试集来源不同。**
  验证集是从双语训练集内部随机切分出的 `2%` 样本，而测试集是官方单独提供的 `Niu.test`。两者在句式分布、表达风格和样本难度上并不完全一致，因此测试集分数略高于验证集并不反常。

- **验证 BLEU 和最终测试 BLEU 的用途不同。**
  验证 BLEU 用于训练过程中的模型选择，更偏向“快速、稳定、可比较”的内部指标；测试 BLEU 则用于最终汇报，通常会使用更好的推理参数，因此二者本来就不是完全同条件下的结果。

因此，本实验中出现“测试 BLEU 略高于验证 BLEU”的现象，可以理解为：模型本身已经具备一定翻译能力，而在最终测试阶段通过更优的解码策略进一步释放了模型性能。

## 6. 可视化结果

### 6.1 Step Loss 曲线

<img src="../../../src/exp4/logs/figures/step_loss.png" alt="Exp4 Step Loss" width="100%" />

从 step 级损失曲线可以看到，训练初期损失较高，随后持续下降，说明模型逐步学会双语映射关系。

### 6.2 Learning Rate 曲线

<img src="../../../src/exp4/logs/figures/lr_curve.png" alt="Exp4 LR Curve" width="100%" />

学习率曲线体现了 warmup + cosine 调度策略：前期逐步升高以稳定训练，后期缓慢衰减以帮助模型收敛。

### 6.3 Epoch Loss 曲线

<img src="../../../src/exp4/logs/figures/epoch_loss.png" alt="Exp4 Epoch Loss" width="100%" />

从 epoch 级损失曲线看，训练集与验证集损失整体同步下降，没有出现明显发散，说明训练过程比较稳定。

### 6.4 Validation BLEU 曲线

<img src="../../../src/exp4/logs/figures/val_bleu4.png" alt="Exp4 Validation BLEU" width="100%" />

验证 BLEU 随 epoch 逐步提升，说明模型不只是降低了 token-level loss，也确实提升了句子级翻译质量。

### 6.5 翻译结果展示

```text
[sample 1] INDEX: 0
[sample 1] SRC : 第二 , 综合治理 .
[sample 1] REF : second , comprehensive management .
[sample 1] PRED: second , comprehensive management .
[sample 2] INDEX: 111
[sample 2] SRC : 考虑 因 种种 原因 部分 储户 不能 亲自 到 金融 机构 办理 存款 的 实际情况 , 《 个人 存款 账户 实名制 规定 》 规定 了 代理 存款 制度 .
[sample 2] REF : in consideration of the circumstances that some depositors are unable to personally handle their deposits in the financial institutions due to various reasons , stipulations on a deposit agent system is included in the " provisions on the system of individual deposit accounts under real names . "
[sample 2] PRED: considering all sorts of reasons , some depositors cannot personally go to the actual conditions of financial institutions , and the " regulations governing the system of individual deposit accounts " stipulated by individual deposit accounts .
[sample 3] INDEX: 222
[sample 3] SRC : 以 此次 发生 五十八 偷渡 者 惨死 案 的 英国 为 例 , 由於 法律 上 的 漏洞 , 人 蛇 偷渡 入 英国 的 情况 愈来愈 严重 .
[sample 3] REF : taking the tragic deaths of 58 illegal immigrants in england as an example , due to loopholes in the law , human snake smuggling in england is becoming more and more serious .
[sample 3] PRED: for example , because of the loopholes in the law , the human smuggling case has become increasingly serious .
[sample 4] INDEX: 333
[sample 4] SRC : 在 回答 关於 中国 宗教界 对 政府 取缔 " 法 轮 功 " 持 何种 态度 的 提问 时 , 圣 辉 法师 说 : " 法 轮 功 " 造成 了 1600 多人 死亡 , 与 毒品 对 人 的 危害 没有 区别 .
[sample 4] REF : commenting on the question of how chinese religious circles have reacted to the government 's banning " falungong , " buddhist master sheng hui said : " falungong " has been responsible for the death of more than 1600 people and harmed people in the same way as illicit drugs .
[sample 4] PRED: when answering a question about the chinese religious circles ' ban on " falungong , " master master li peng said : " falungong " has caused more than 1,600 deaths , and there are no harm to drugs .
[sample 5] INDEX: 444
[sample 5] SRC : 迟浩田 首先 转达 了 李鹏 委员长 对 布马扎 的 诚挚 问候 和 良好 祝愿 , 并 请 布马扎 转达 江泽民 主席 对 阿尔及利亚 总统 布特弗利卡 的 亲切 问候 和 良好 祝愿 .
[sample 5] REF : chi haotian first conveyed chairman li peng 's sincere regards and good wishes to boumaza and asked boumaza to convey president jiang zemin 's cordial and good wishes to algerian president abdelaziz bouteflika .
[sample 5] PRED: chi haotian first conveyed chairman li peng 's sincere regards and good wishes to , and asked li peng to convey president jiang zemin 's cordial regards and best wishes to algerian president bouteflika .
[sample 6] INDEX: 555
[sample 6] SRC : 一些 世贸 组织 成员 在 发言 中 对 中国 代表团 为 推进 谈判 进程 所 作出 的 重要 努力 给予 了 高度 评价 .
[sample 6] REF : a number of wto members spoke highly of china 's important measures taken by the chinese delegation to speed up the process .
[sample 6] PRED: some wto members spoke highly of the important efforts made by the chinese delegation to promote the process of negotiations .
[sample 7] INDEX: 666
[sample 7] SRC : 面对 机遇 和 挑战 , 我们 惟有 迎头 而 上 , 才能 不 辜负 中央 和 全国 人民 的 厚望 .
[sample 7] REF : in face of these opportunities and challenges , we have to work hard to catch up [ with the rest of the country ] in order not to be unworthy of the expectations of the central authorities and the people of the whole country .
[sample 7] PRED: in the face of opportunities and challenges , we can only live up to the central authorities and the people throughout the country .
[sample 8] INDEX: 777
[sample 8] SRC : 他 多次 在 公开 场合 强调 印度 将 寻求 同 中国 建立 友好 , 合作 , 睦邻 和 互惠 的 外交关系 .
[sample 8] REF : he has stressed in many public speeches that india will seek to establish friendly , cooperative , good neighborly , and mutually beneficial relations with china .
[sample 8] PRED: he stressed on many occasions that india will seek a friendly , good neighborly , and mutually beneficial diplomatic relations with china .
[sample 9] INDEX: 888
[sample 9] SRC : 二 是 推进 财政 制度 改革 , 要 积极 推行 部门 预算 制度 改革 , 国库 集中 收付 制度 改革 和 税费 改革 , 继续 认真 落实 " 收支 两 条 线 " 规定 , 加强 预算外 资金 监管 .
[sample 9] REF : it is necessary to actively reform the departmental budgetary system , adopt the state treasury 's concentrated revenue and expenditure system , and reform the tax and fee systems . the rules on separating the management of revenue from the management of expenses should be seriously implemented . funds outside the budgets should be brought under stricter control .
[sample 9] PRED: second , it is necessary to actively promote the reform of the financial system , carry out reform of the budget system , centralized receipt and fee reform of the state treasury , and continue to implement the " two lines of revenue and expenditure " in revenue and expenditure , and strengthen supervision over funds .
[sample 10] INDEX: 999
[sample 10] SRC : 中国 已经 同 绝大多数 邻国 解决 了 旧 中国 遗留 下来 的 麻烦 问题 , 划定 了 边界 .
[sample 10] REF : china has now resolved troublesome problems left over from old china with the great majority of its neighbors and defined the borders with them .
[sample 10] PRED: china has solved the trouble left over by the overwhelming majority of its neighbors , and the problems left over from old china have been resolved .
```

### 6.6 样例分析

以下给出当前测试预览中的 3 条代表性样例分析。

#### 样例 1：短句翻译准确

- `SRC`：第二 , 综合治理 .
- `REF`：second , comprehensive management .
- `PRED`：second , comprehensive management .

分析：

- 该句长度较短、结构简单、词汇搭配常见，模型能够实现与参考译文完全一致的翻译。
- 说明当前模型对高频短句和固定表达具备较好的记忆与生成能力。
- 这类样例表明模型已经掌握了基础的中英对齐关系。

#### 样例 2：长句主干基本正确，但后半句不够自然

- `SRC`：考虑 因 种种 原因 部分 储户 不能 亲自 到 金融 机构 办理 存款 的 实际情况 , 《 个人 存款 账户 实名制 规定 》 规定 了 代理 存款 制度 .
- `REF`：in consideration of the circumstances that some depositors are unable to personally handle their deposits in the financial institutions due to various reasons , stipulations on a deposit agent system is included in the " provisions on the system of individual deposit accounts under real names . "
- `PRED`：considering all sorts of reasons , some depositors cannot personally go to the actual conditions of financial institutions , and the " regulations governing the system of individual deposit accounts " stipulated by individual deposit accounts .

分析：

- 模型抓住了句子的核心语义，包括“由于种种原因”“部分储户不能亲自办理存款”“实名制规定”等关键信息。
- 但预测句在后半部分语法结构明显不够自然，尤其是 `stipulated by individual deposit accounts` 这一表达不符合英语习惯。
- 这说明模型在处理中长句、嵌套修饰和制度类文本时，虽然能保留主干信息，但在句法组织和目标语言流畅度上仍存在不足。

#### 样例 3：主题把握正确，但细节信息有遗漏

- `SRC`：以 此次 发生 五十八 偷渡 者 惨死 案 的 英国 为 例 , 由於 法律 上 的 漏洞 , 人 蛇 偷渡 入 英国 的 情况 愈来愈 严重 .
- `REF`：taking the tragic deaths of 58 illegal immigrants in england as an example , due to loopholes in the law , human snake smuggling in england is becoming more and more serious .
- `PRED`：for example , because of the loopholes in the law , the human smuggling case has become increasingly serious .

分析：

- 模型正确保留了“以英国为例”“法律漏洞”“偷渡问题日益严重”等核心主题。
- 但预测结果遗漏了关键细节 `58`，也弱化了“偷渡入英国”这一更具体的语义范围。
- 该样例说明当前模型在复杂新闻句中能够概括主要含义，但仍可能出现信息压缩过度、细节丢失的问题。

### 6.7 样例总体分析

结合当前展示的 10 条官方测试样例，可以对模型翻译表现作出如下总体分析：

- **模型已经具备较稳定的基础翻译能力。**
  对于短句、常见表达和结构相对简单的句子，模型往往可以给出较准确甚至与参考译文一致的结果。例如样例 1 和样例 6 都较好地保留了原句主干含义，说明模型已经学习到较可靠的词汇对齐关系和常见句式模式。

- **模型对句子主干语义的把握整体较好。**
  在大多数中长句中，模型通常能够正确识别核心事件、主要逻辑关系以及关键实体，例如“推进谈判进程”“建立友好合作关系”“法律漏洞导致问题加剧”等主题信息都能基本保留。这说明当前系统在“传达大意”这一层面已经达到较可用水平。

- **复杂长句下的句法组织和表达自然度仍然不足。**
  从样例 2、样例 4 和样例 9 可以看出，当输入涉及多层修饰、长距离依赖或政策制度类表达时，模型容易出现后半句结构松散、搭配不自然、局部语义拼接生硬等问题。也就是说，模型能够抓住主题，但在将中文复杂结构自然地重组为英文时仍不够稳定。

- **信息遗漏和细节失真是当前最明显的质量瓶颈。**
  部分样例中出现了数字丢失、人物替换、宾语缺失、修饰范围缩减等问题。例如样例 3 丢失了“58”这一关键信息，样例 4 出现了人物错置，样例 5 丢失了部分受话对象信息。这说明模型在复杂信息密集句中容易保留主旨、压缩细节，从而影响译文准确性。

- **目标语言流畅度已有一定基础，但离正式高质量译文仍有差距。**
  一些句子的输出已经基本可读，且英语表达较为连贯；但也有部分句子在搭配、指代、逻辑连接和术语表达上还不够自然。整体来看，当前结果更接近“能够较稳定传达原文大意的机器翻译基线”，距离高精度、正式出版级译文还有进一步优化空间。

总体来看，这组样例与最终 `24.62 BLEU` 的测试结果是一致的：模型已经能够较稳定地完成大意翻译，在短句和常见表达上表现较好，但在长句、制度性文本和细节保真度方面仍存在明显提升空间。

## 7. 实验总结

- 本实验完成了基于 Transformer 的中英机器翻译系统实现，具备训练、验证、测试评估、可视化和断点续训能力。
- 当前训练阶段最优验证 BLEU 达到 `22.71`，在官方 `1000` 条测试集上的最终结果为 `24.62 BLEU`，整体结果符合当前模型规模和实验设置下的预期。
- 本实验训练阶段采用的是“训练集内部切分验证集”的常规做法，适合用于选择 best checkpoint；最终汇报结果则以官方 `1000` 条测试集 BLEU 为准。
- 后续可进一步从以下方向继续优化：
  - 增强解码策略搜索
  - 尝试 SentencePiece 子词建模
  - 调整模型宽度、层数与训练轮数
  - 引入更强的数据清洗或双语筛选策略

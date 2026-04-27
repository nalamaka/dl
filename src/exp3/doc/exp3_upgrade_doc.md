# Exp3 改造说明

## 1. 目标
本版本在两个约束下改造：
- 16GB 显存
- 不使用大语言模型微调（不使用 LoRA/QLoRA/外部 LLM）

主要完成：
1. 网络从 LSTM 升级为字符级因果 Transformer
2. 生成阶段增加“绝句格式约束 + 押韵约束”
3. 新增批量评估模式，支持多维指标

## 2. 开发内容
### 2.1 模型结构
- 文件：`src/exp3/model.py`
- 要点：
  - 字符级 token embedding + 绝对位置嵌入
  - 循环句位嵌入（cycle embedding），增强绝句结构感知
  - TransformerEncoder 因果 mask
  - tie embeddings（输出头与输入嵌入共享权重）

### 2.2 生成约束
- 文件：`src/exp3/engine.py`
- 字数/格式约束：
  - 按 `line_length` + `num_lines` 生成
  - 第1/3句句末强制逗号，第2/4句句末强制句号
- 押韵约束：
  - 第2句韵脚作为种子
  - 第4句韵脚强制落在同韵簇
- 工程细节：
  - 滑动窗口推理，避免超过 `max_seq_len`
  - top-k / top-p / repetition penalty 可配

### 2.3 韵部构建
- 文件：`src/exp3/rhyme_utils.py`
- 策略：
  - 从训练语料提取诗句
  - 以第2/4句尾字共现关系构建字符簇
  - 形成数据驱动的简化韵部词典

### 2.4 评估体系
- 文件：`src/exp3/evaluator.py`
- 指标：
  - `format_score`：行数/句长/标点综合分
  - `line_count_ok`：行数合规
  - `line_length_ok`：句长合规
  - `punctuation_ok`：标点位置合规
  - `rhyme_score`：押韵命中
  - `distinct1`：字符去重率（高更好）
  - `repetition2`：二元重复率（低更好）
  - `ppl`：困惑度
  - `quality_score`：综合分

## 3. 运行指南
### 3.1 训练
```bash
python main.py --mode train --data_path ../../data/tang.npz --epochs 30 --batch_size 16 --grad_accum_steps 2
```

### 3.2 生成（绝句 + 押韵）
```bash
python main.py --mode generate --ckpt ./checkpoints/poetry_transformer.pth --start_words 湖光秋月两相和 --use_constraints --use_rhyme_constraint --line_length 7 --num_lines 4 --rhyme_lines 2,4
```

### 3.3 批量评估
```bash
python main.py --mode evaluate --ckpt ./checkpoints/poetry_transformer.pth --eval_samples 20 --use_constraints --use_rhyme_constraint --line_length 7 --num_lines 4
```

## 4. 注意事项
- 旧 checkpoint 和新模型架构存在 key 不一致，已实现 `strict=False` 兼容加载。
- 要获得真正效果提升，建议重新训练新架构权重。
- 当前韵部是“语料统计簇”方式，后续可替换为平水韵显式词典。

# exp3 诗歌生成说明（16GB 显存友好版）

## 概述
本项目在以下约束下工作：
- 16GB 显存
- 不使用大语言模型微调（不使用 LoRA/QLoRA/外部 LLM）

已支持功能：
- 字符级因果 Transformer
- 绝句格式约束（句长 + 标点）
- 押韵约束（默认第 1/2/4 句）
- 藏头生成
- 可选“仄起平收”约束与评估（依赖 pypinyin）
- 断点续训
- 多维指标评估

## 常用命令
### 1) 训练
```bash
python main.py --mode train --data_path ../../data/tang.npz --ckpt ./checkpoints/poetry_transformer.pth --epochs 30
```

### 2) 断点续训
```bash
python main.py --mode train --data_path ../../data/tang.npz --ckpt ./checkpoints/poetry_transformer.pth --epochs 30 --resume
```

### 3) 生成（智能默认）
```bash
python main.py --mode generate --ckpt ./checkpoints/poetry_transformer.pth --start_words 湖光秋月两相和
```

### 4) 生成（藏头 + 押韵）
```bash
python main.py --mode generate --ckpt ./checkpoints/poetry_transformer.pth --start_words 春 --acrostic_text 春夏秋冬 --num_lines 4 --rhyme_lines 1,2,4
```

### 5) 评估
```bash
python main.py --mode evaluate --ckpt ./checkpoints/poetry_transformer.pth --eval_samples 20 --num_lines 4 --rhyme_lines 1,2,4 --verbose_eval
```

## 智能默认与校验
- `line_length` 现在可不填：
  - 输入不含 `，`/`。` 时，默认 `7`
  - 输入含 `，` 或 `。` 时，自动推断句长
- 会自动检查：
  - 各句句长是否一致
  - 是否超过最大句长（`Config.max_line_length`）
  - `acrostic_text` 长度是否超过 `num_lines`
  - `rhyme_lines` 是否超出 `num_lines`
- 默认参数：
  - `num_lines=4`
  - `rhyme_lines=1,2,4`
  - `use_constraints=True`
  - `use_rhyme_constraint=True`

## 评估指标
- `format_score` / `line_count_ok` / `line_length_ok` / `punctuation_ok`
- `rhyme_score` / `rhyme_124_score`
- `zeqi_pingshou_score` / `tone_valid_ratio`
- `acrostic_score`
- `distinct1`
- `repetition2`（已改进：字级重复 + 2-gram 重复的组合指标）
- `ppl`
- `quality_score`

## pypinyin 安装
```bash
python -m pip install pypinyin
```

若使用项目虚拟环境：
```bash
e:\hw\deep_learning\mineru_env\Scripts\python.exe -m pip install pypinyin
```

# 实验1 - MNIST CNN

## 安装

```bash
pip install -r requirements.txt
```

## 手动运行
- 训练：

```bash
python main.py
```

- 可视化原始 MNIST 样本：

```bash
python visualize.py
```

- 可视化训练曲线和预测样例：

```bash
python visualize_results.py
```

## 输出文件

- `log/train_log.csv`: 训练日志（按 step 记录）
- `checkpoint/best_cnn.pth`: 最优模型权重
- `visualizations/training_curve.png`: 训练曲线图
- `visualizations/case_showcase.png`: 预测样例网格图

# CHI3242 Assignment 2: 苏轼 vs 辛弃疾词作者分类器

## 研究问题

给定一首宋词的正文，判断它是苏轼写的还是辛弃疾写的。

- **标签 0**：苏轼
- **标签 1**：辛弃疾

## 数据来源

- 数据来自公开的中文诗词数据库 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- 苏轼词：352首
- 辛弃疾词：352首
- 共704条样本，两类平衡

## 文件结构

```
.
├── README.md              # 本文件
├── data/
│   └── poems.csv         # 标注好的训练数据（text, label）
├── scripts/
│   ├── train_classifier.py    # 训练+评估脚本
│   └── apply_to_new_text.py   # 应用到李清照词的预测脚本
├── output/
│   ├── test_predictions.csv   # 测试集预测结果
│   ├── metrics.csv            # 准确率/精确率/召回率/F1
│   ├── confusion_matrix.csv   # 混淆矩阵
│   └── applied_li_qingzhao.csv  # 李清照词的预测
├── report.md              # 报告 Markdown 版
└── report.pdf             # 报告 PDF 版
```

## 运行方法

### 在 Google Colab 中运行

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. 新建笔记本
3. 上传 `data/poems.csv` 到 Colab
4. 依次粘贴运行：
   - `scripts/train_classifier.py`（训练+评估）
   - `scripts/apply_to_new_text.py`（应用到李清照词）

### 在本地运行

```bash
pip install scikit-learn pandas jieba
python scripts/train_classifier.py
python scripts/apply_to_new_text.py
```

## 方法说明

- **分词**：jieba 中文分词
- **特征**：TF-IDF，取 top 200 词
- **切分**：80% 训练 / 20% 测试，分层抽样，random_state=42
- **模型**：LogisticRegression
- **评估**：准确率、混淆矩阵、精确率、召回率、F1

## 主要结果

- 测试集准确率：78%
- 详见 `report.md`

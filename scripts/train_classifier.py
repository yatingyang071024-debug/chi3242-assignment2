# ============================================================
# CHI3242 Assignment 2: 训练苏轼 vs 辛弃疾词作者分类器
# 这是主训练脚本：加载数据 → 切分 → 特征 → 训练 → 评估
# ============================================================

# ---------- 第一步：安装需要的库（Colab 里运行一次就行） ----------
# 如果在本地跑，先在终端执行：pip install scikit-learn pandas jieba
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# 在 Colab 上需要装 jieba；sklearn 和 pandas Colab 自带
try:
    import jieba
except ImportError:
    install("jieba")
    import jieba

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    classification_report
)
import os

# ---------- 第二步：读取数据 ----------
# 在 Colab 里，你需要先把 data/poems.csv 上传到 Colab
# （点左侧文件夹图标 → 上传按钮 → 选 poems.csv）
# 如果你在本地跑，路径直接指向 data/poems.csv

DATA_PATH = "poems.csv"  # Colab 里上传后就在根目录
# 如果在本地跑，改成："data/poems.csv"

df = pd.read_csv(DATA_PATH)
print(f"总共 {len(df)} 条词")
print(f"苏轼(label=0): {sum(df['label']==0)} 条")
print(f"辛弃疾(label=1): {sum(df['label']==1)} 条")
print()

# ---------- 第三步：中文分词 ----------
# 英文可以按空格分，但中文没有空格，要用 jieba 把句子切成词
def tokenize(text):
    """用 jieba 把一段中文切成词语列表"""
    return list(jieba.cut(text))

# 测试一下
print("分词示例：")
print("原文:", df.iloc[0]['text'][:30])
print("分词:", "/".join(tokenize(df.iloc[0]['text'])[:15]))
print()

# ---------- 第四步：80/20 分层切分 ----------
# stratify=df['label'] 保证训练集和测试集里两类比例一样
# random_state=42 是固定种子，每次跑结果都一样（作业要求）
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% 当测试集
    random_state=42,       # 固定种子
    stratify=y              # 按标签分层抽样
)

print(f"训练集: {len(X_train)} 条 (苏轼 {sum(y_train==0)}, 辛弃疾 {sum(y_train==1)})")
print(f"测试集: {len(X_test)} 条 (苏轼 {sum(y_test==0)}, 辛弃疾 {sum(y_test==1)})")
print()

# ---------- 第五步：特征提取（TF-IDF，取最常见的200个词） ----------
# 作业里问：用词频？TF-IDF？还是 z-score？
# 这里用 TF-IDF —— 它既考虑词频，又降低常见词的权重，是文本分类最常用的
# max_features=200 表示只取最重要的200个词当特征

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,      # 用我们写的 jieba 分词函数
    max_features=200,        # 只取前200个高频词
    token_pattern=None,     # 关掉 sklearn 自带的分词，全用我们的
    lowercase=False         # 中文不分大小写
)

# 对训练集拟合并转换，对测试集只转换（不能 fit 测试集）
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"特征矩阵大小（训练集）: {X_train_vec.shape}")
print(f"特征矩阵大小（测试集）: {X_test_vec.shape}")
print(f"用到的200个词（前20个）: {vectorizer.get_feature_names_out()[:20].tolist()}")
print()

# ---------- 第六步：训练 LogisticRegression 模型 ----------
model = LogisticRegression(max_iter=1000)  # max_iter 加大保证收敛
model.fit(X_train_vec, y_train)
print("模型训练完成！")
print()

# ---------- 第七步：在测试集上评估 ----------
y_pred = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("测试集评估结果")
print("=" * 50)
print(f"准确率 Accuracy:  {acc:.4f}")
print(f"精确率 Precision: {prec:.4f}  (预测为辛弃疾的词里，真的是辛弃疾的比例)")
print(f"召回率 Recall:    {rec:.4f}  (所有辛弃疾的词里，被找出来的比例)")
print(f"F1 分数:          {f1:.4f}")
print()
print("混淆矩阵 Confusion Matrix:")
print("                 预测:苏轼   预测:辛弃疾")
print(f"实际:苏轼          {cm[0][0]:>4}        {cm[0][1]:>4}")
print(f"实际:辛弃疾        {cm[1][0]:>4}        {cm[1][1]:>4}")
print()
print("详细分类报告：")
print(classification_report(y_test, y_pred, target_names=["苏轼", "辛弃疾"]))

# ---------- 第八步：保存结果到 output/ ----------
os.makedirs("output", exist_ok=True)

# 保存测试集预测结果
results = pd.DataFrame({
    'text': X_test.values,
    'actual_label': y_test.values,
    'predicted_label': y_pred,
    'prob_苏轼': y_pred_proba[:, 0],
    'prob_辛弃疾': y_pred_proba[:, 1]
})
results.to_csv("output/test_predictions.csv", index=False, encoding='utf-8-sig')
print("已保存 output/test_predictions.csv")

# 保存评估指标
metrics = pd.DataFrame({
    'metric': ['accuracy', 'precision', 'recall', 'f1'],
    'value': [acc, prec, rec, f1]
})
metrics.to_csv("output/metrics.csv", index=False)
print("已保存 output/metrics.csv")

# 保存混淆矩阵
cm_df = pd.DataFrame(cm,
    index=['实际:苏轼', '实际:辛弃疾'],
    columns=['预测:苏轼', '预测:辛弃疾'])
cm_df.to_csv("output/confusion_matrix.csv")
print("已保存 output/confusion_matrix.csv")

# 看看哪些词对分类最有帮助
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
top_sushi = sorted(zip(feature_names, coef), key=lambda x: x[1])[:15]  # 最偏向苏轼的词
top_xin = sorted(zip(feature_names, coef), key=lambda x: -x[1])[:15]   # 最偏向辛弃疾的词

print()
print("最能区分两人的词：")
print("  偏向苏轼的词:", [w for w, c in top_sushi])
print("  偏向辛弃疾的词:", [w for w, c in top_xin])

# ============================================================
# CHI3242 Assignment 2: 把训练好的模型用到另一段文本上
# 这里拿李清照的词当"未知文本"，让模型猜它像苏轼还是辛弃疾
# ============================================================

import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

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
import os

# ---------- 第一步：重新训练模型（和 train_classifier.py 一样的流程） ----------
# 为了让这个脚本独立可跑，我们重新加载数据、训练一遍
# （在 Colab 里 poems.csv 要先上传）

DATA_PATH = "poems.csv"
df = pd.read_csv(DATA_PATH)

def tokenize(text):
    return list(jieba.cut(text))

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    tokenizer=tokenize, max_features=200,
    token_pattern=None, lowercase=False
)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
print("模型重新训练完成！\n")

# ---------- 第二步：准备"另一段自选文本" ----------
# 这里选李清照的词。李清照是宋代女词人，风格和苏辛都不同。
# 我们让模型看：这些词在它眼里，更像苏轼还是辛弃疾？

li_qingzhao_poems = [
    {
        "title": "如梦令·昨夜雨疏风骤",
        "text": "昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧。知否，知否？应是绿肥红瘦。"
    },
    {
        "title": "一剪梅·红藕香残玉簟秋",
        "text": "红藕香残玉簟秋。轻解罗裳，独上兰舟。云中谁寄锦书来？雁字回时，月满西楼。花自飘零水自流。一种相思，两处闲愁。此情无计可消除，才下眉头，却上心头。"
    },
    {
        "title": "声声慢·寻寻觅觅",
        "text": "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。乍暖还寒时候，最难将息。三杯两盏淡酒，怎敌他、晚来风急！雁过也，正伤心，却是旧时相识。满地黄花堆积，憔悴损，如今有谁堪摘？守着窗儿，独自怎生得黑！梧桐更兼细雨，到黄昏、点点滴滴。这次第，怎一个愁字了得！"
    },
    {
        "title": "武陵春·风住尘香花已尽",
        "text": "风住尘香花已尽，日晚倦梳头。物是人非事事休，欲语泪先流。闻说双溪春尚好，也拟泛轻舟。只恐双溪舴艋舟，载不动许多愁。"
    },
    {
        "title": "夏日绝句",
        "text": "生当作人杰，死亦为鬼雄。至今思项羽，不肯过江东。"
    },
]

# ---------- 第三步：用模型预测 ----------
print("=" * 60)
print("模型对李清照词的判断")
print("=" * 60)
print(f"{'词牌':<25} {'预测':<8} {'苏轼概率':>10} {'辛弃疾概率':>10}")
print("-" * 60)

results = []
for p in li_qingzhao_poems:
    vec = vectorizer.transform([p['text']])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label_name = "苏轼" if pred == 0 else "辛弃疾"
    print(f"{p['title']:<25} {label_name:<8} {proba[0]:>10.4f} {proba[1]:>10.4f}")
    results.append({
        'title': p['title'],
        'text': p['text'],
        'predicted_label': label_name,
        'prob_苏轼': proba[0],
        'prob_辛弃疾': proba[1]
    })

# ---------- 第四步：保存结果 ----------
os.makedirs("output", exist_ok=True)
pd.DataFrame(results).to_csv("output/applied_li_qingzhao.csv", index=False, encoding='utf-8-sig')
print()
print("已保存 output/applied_li_qingzhao.csv")
print()
print("解读：")
print("- 如果概率接近 0.5，说明模型也拿不准（这是合理的，李清照本来就不是苏辛）")
print("- 如果明显偏向某一边，说明模型学到的苏/辛用词习惯在李清照词里也有体现")
print("- 《夏日绝句》风格豪放，可能更像辛弃疾——这就是分析的有趣点")

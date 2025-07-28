import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义混淆矩阵
confusion_matrix = np.array([
    [18, 0, 0, 0],
    [0, 10, 1, 0],
    [0, 5, 12, 3],
    [0, 3, 5, 15]
])

# 计算总体样本数
total = np.sum(confusion_matrix)

# 初始化存储结果的字典
metrics = {}
class_labels = [f'Class {i}' for i in range(4)]

# 遍历每个类别计算指标
for i in range(4):
    tp = confusion_matrix[i, i]
    fn = np.sum(confusion_matrix[:, i]) - tp
    fp = np.sum(confusion_matrix[i, :]) - tp
    tn = total - (tp + fp + fn)

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    accuracy = (tp + tn) / total

    metrics[class_labels[i]] = {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1 Score': f1,
        'Accuracy': accuracy
    }

# 转换为DataFrame
metrics_df = pd.DataFrame(metrics).T.round(3)

# 设置绘图参数
plt.rcParams['font.family'] = 'Arial'  # 设置SCI论文常用字体
plt.figure(figsize=(8, 6))

# 绘制热力图
ax = sns.heatmap(
    metrics_df.T,  # 转置使指标作为行显示
    annot=True,
    cmap="GnBu",
    fmt=".3f",
    cbar=True,
    linewidths=0.5,
    linecolor='black',
    vmin=0,
    vmax=1,
    annot_kws={"size": 12}
)

# 调整坐标轴
plt.title("Classification Performance Metrics Heatmap", fontsize=16, weight='bold', pad=20)
plt.xlabel("Class Labels", fontsize=14, weight='bold')
plt.ylabel("Metrics", fontsize=14, weight='bold')
plt.xticks(fontsize=12, weight='bold', rotation=45)
plt.yticks(fontsize=12, weight='bold', rotation=0)

# 调整颜色条
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# 保存图片
plt.tight_layout()
plt.savefig("D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-决策树分类/cm_heatmap.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
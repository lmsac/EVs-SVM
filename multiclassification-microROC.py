from sklearn.model_selection import KFold
import random  # 导入random模块
import numpy as np  # 从numpy包中导入np模块
from sklearn.feature_selection import RFE  # 从sklearn包中导入feature_selection模块的RFE模块
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import stats
from scipy.stats import mode
from collections import Counter
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import f1_score

random.seed(42)

#mean_y_score_valids_svm = np.genfromtxt('/Users/apple/Desktop/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machine learning-Python/multiclassification/导入/多分类ROC/test_probability_multi.csv', delimiter=',')  #导入每个分类对应的概率
#y_valid = np.genfromtxt('/Users/apple/Desktop/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machine learning-Python/multiclassification/导入/多分类ROC/y-valid-T.csv', delimiter=',') #导入实际标签

#mean_y_score_valids_svm = np.genfromtxt('/Users/apple/Desktop/课题2_Proteomics/20240303-dataanalysis/SVM多分类/blue module/sign-gene/exosome protien_siggene_test_probability_R.csv', delimiter=',')  #导入每个分类对应的概率
#y_valid = np.genfromtxt('/Users/apple/Desktop/课题2_Proteomics/20240303-dataanalysis/SVM多分类/blue module/sign-gene/exosome protein-signgene-y-valid-test.csv', delimiter=',') #导入实际标签
mean_y_score_valids_svm = np.genfromtxt("D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machine learning-Python/multiclassification/导入/多分类ROC/test_probability_multi-修改.csv", delimiter=',')  #导入每个分类对应的概率
y_valid = np.genfromtxt("D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machine learning-Python/multiclassification/导入/多分类ROC/y-valid-T-修改.csv", delimiter=',') #导入实际标签

print("Mean Y Score Valids SVM Shape:", mean_y_score_valids_svm.shape)
print("Y Valid Shape:", y_valid.shape)


# 假设有多个类的标签 (y_valid) 和对应的预测概率 (mean_y_score_valids_svm)
# 将标签二值化 (假设有 n_classes 个类别)
n_classes = np.unique(y_valid).shape[0]
y_valid_bin = label_binarize(y_valid, classes=np.arange(n_classes))

# 分别计算每个类的ROC曲线
fpr = dict()
tpr = dict()
roc_auc = dict()

# 对每个类计算 fpr, tpr 和阈值
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_valid_bin[:, i], mean_y_score_valids_svm[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# 计算微平均ROC曲线
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_valid_bin.ravel(), mean_y_score_valids_svm.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

# 计算宏平均ROC曲线
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# 绘制多分类的ROC曲线
plt.figure(figsize=(400 / 80, 400 / 80))
colors = cycle(['#F898CB', '#FFCB5B', '#A8D3A0', '#8582BD'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# 微平均ROC曲线
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='#0076B9', linestyle=':', linewidth=3)

# 宏平均ROC曲线
plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='#D65190', linestyle=':', linewidth=3)
Font = {'size': 12, 'family': 'Arial'}
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('True Positive Rate', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=12)
plt.tick_params(labelsize=12)
plt.savefig("D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machineLearning-R/multiclassification/20250618修改/exosome-protein-roc-with-ci.svg")
plt.show()

# 由于已经有了每个类别的预测概率mean_y_score_valids_svm，我们需要将这些概率转换为预测标签
# 这里我们使用np.argmax来获取概率最高的类别索引作为预测标签
y_pred = np.argmax(mean_y_score_valids_svm, axis=1)

# 计算微平均F1-score
f1_micro = f1_score(y_valid, y_pred, average='micro')
print("Micro-average F1-score:", f1_micro)

# 计算宏平均F1-score
f1_macro = f1_score(y_valid, y_pred, average='macro')
print("Macro-average F1-score:", f1_macro)
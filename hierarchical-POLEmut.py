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

random.seed(234)
def feature_selector(X_train,y_train,X_test,y_test): #X_train 和 y_train 是训练数据的特征和标签

    rfe_selector = RFE(
        estimator=LogisticRegression(
            penalty='l1',  #指定正则化类型：L1正则化
            solver='liblinear',  #用于优化的算法
            class_weight='balanced'  #处理不平衡数据集
        ),
        n_features_to_select=30  #使用逻辑回归作为基估计器来选择最重要的30个特征
    )
    X_train_RFE = rfe_selector.fit_transform(X_train, y_train)  #fit方法用于训练 RFE 模型，它会根据 y_train（标签）来评估 X_train（特征）的重要性，并选择最重要的特征，transform 方法用于将训练好的模型应用到数据上，以选择特征并创建一个新的数组，其中只包含被选中的特征
    selected_features = rfe_selector.support_  #获取由RFE选择器选出的特征的布尔值
    selected_feature_indices = np.where(selected_features)[0] #是一个numpy函数，返回一个元组，找出被选中特征的索引
    X_test_RFE = X_test[:, selected_features] #从测试集中提取相应的特征，：代表选择所有行，后面代表选择的列的索引
    X_valid_RFE = X_valid[:, selected_features] #从验证集中提取相应的特征
    return X_train_RFE, y_train, X_test_RFE, y_test, X_valid_RFE, y_valid, selected_feature_indices
X = np.genfromtxt("D:/yjs/FDU/数据处理/20240905-peaks/X_409peaks.csv", delimiter=',')
y = np.genfromtxt("D:/yjs/FDU/数据处理/20240905-peaks/y_409peaks_1&234.csv", delimiter=',')

X_traintest = np.genfromtxt("D:/yjs/FDU/数据处理/20240905-peaks/X_traintest.csv", delimiter=',')
y_traintest = np.genfromtxt("D:/yjs/FDU/数据处理/20240905-peaks/y_traintest.csv", delimiter=',')
X_valid = np.genfromtxt("D:/yjs/FDU/数据处理/20240905-peaks/X_valid.csv", delimiter=',')
y_valid = np.genfromtxt("D:/yjs/FDU/数据处理/20240905-peaks/y_valid.csv", delimiter=',')
print("X_traintest shape:", X_traintest.shape)
print("y_traintest shape:", y_traintest.shape)
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)



# 设定十折交叉验证
kf = KFold(n_splits=10, shuffle=True,random_state=42) #数据集被分成10个部分，对数据集中的样本随机打乱

# 创建各个分类器
lr = make_pipeline(StandardScaler(),
                       LogisticRegression(penalty='l1',
                                          solver='liblinear',
                                          class_weight='balanced',
                                          ))
mlp = make_pipeline(StandardScaler(),  #多层感知器
                       MLPClassifier(solver='lbfgs',
                                     activation='relu',
                                     max_iter=20000,
                                     hidden_layer_sizes=(500,500,500,500),
                                     random_state=42,
                                     alpha=0.001,
                                     learning_rate_init=0.001))
knei = make_pipeline(StandardScaler(),  #K最邻近算法KNN
                        KNeighborsClassifier(5))
#svm = make_pipeline(StandardScaler(),
#                       svm.LinearSVC(
#                           class_weight='balanced', max_iter=20000))
svm = make_pipeline(StandardScaler(),
                        svm.LinearSVC(class_weight='balanced', max_iter=20000, C=10.0, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, random_state=42)
)
gdbt = make_pipeline(StandardScaler(),  #梯度提升决策树
                        GradientBoostingClassifier(random_state=10))
xgb = make_pipeline(StandardScaler(),  #梯度提升树
                       XGBClassifier())
rf = make_pipeline(StandardScaler(),  #随机森林
                      RandomForestClassifier())


#空列表
y_pred_valids_combine, y_score_valids_combine = [], []   #用于储存模型对验证集的预测标签和对验证集预测的分数或概率
y_pred_valids_lr, y_score_valids_lr = [], []
y_pred_valids_svm, y_score_valids_svm = [], []
y_pred_valids_mlp, y_score_valids_mlp = [], []
y_pred_valids_knei, y_score_valids_knei = [], []
y_pred_valids_gdbt, y_score_valids_gdbt = [], []
y_pred_valids_xgb, y_score_valids_xgb = [], []
y_pred_valids_rf, y_score_valids_rf = [], []
select_features = []

# 进行五折交叉验证

for train_index, test_index in kf.split(X_traintest):  #接收数据集X-traintest作为参数，并生成训练和测试索引，train-index包含被选为训练集的样本索引

    # 分割训练集和测试集
    X_train, X_test = X_traintest[train_index], X_traintest[test_index] #crosstest参数[outer_test]表示在独立测试集上测试[test_index]表示在训练集交叉验证分出的测试集上进行测试
    y_train, y_test = y_traintest[train_index], y_traintest[test_index]

    X_train, y_train, X_test, y_test, X_valid_cross, y_valid_cross, feature = feature_selector(X_train, y_train, X_test, y_test)
    select_features.append(feature)
    # 训练模型
    svm.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    knei.fit(X_train, y_train)
    gdbt.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr_test = lr.predict(X_test)
    y_pred_svm_test = svm.predict(X_test)
    y_pred_mlp_test = mlp.predict(X_test)
    y_pred_knei_test = knei.predict(X_test)
    y_pred_gdbt_test = gdbt.predict(X_test)
    y_pred_xgb_test = xgb.predict(X_test)
    y_pred_rf_test = rf.predict(X_test)
    testaccuracy_svm = accuracy_score(y_test, y_pred_svm_test)
    testaccuracy_lr = accuracy_score(y_test, y_pred_lr_test)
    testaccuracy_mlp = accuracy_score(y_test, y_pred_mlp_test)
    testaccracy_knei = accuracy_score(y_test, y_pred_knei_test)
    testaccracy_gdbt = accuracy_score(y_test, y_pred_gdbt_test)
    testaccracy_xgb = accuracy_score(y_test, y_pred_xgb_test)
    testaccracy_rf = accuracy_score(y_test, y_pred_rf_test)
    print(f"Test Accuracy SVM: {testaccuracy_svm}")
    print(f"Test Accuracy LR: {testaccuracy_lr}")
    print(f"Test Accuracy MLP: {testaccuracy_mlp}")
    print(f"Test Accuracy KNEI: {testaccracy_knei}")
    print(f"Test Accuracy gdbt: {testaccracy_gdbt}")
    print(f"Test Accuracy XGB: {testaccracy_xgb}")
    print(f"Test Accuracy RF: {testaccracy_rf}")
    # 在验证集上评估模型
    y_pred_lr_valid = lr.predict(X_valid_cross)
    y_pred_svm_valid = svm.predict(X_valid_cross)
    y_pred_mlp_valid = mlp.predict(X_valid_cross)
    y_pred_knei_valid = knei.predict(X_valid_cross)
    y_pred_gdbt_valid = gdbt.predict(X_valid_cross)
    y_pred_xgb_valid = xgb.predict(X_valid_cross)
    y_pred_rf_valid = rf.predict(X_valid_cross)
    validaccuracy_svm = accuracy_score(y_valid, y_pred_svm_valid)
    validaccuracy_lr = accuracy_score(y_valid, y_pred_lr_valid)
    validaccuracy_mlp = accuracy_score(y_valid, y_pred_mlp_valid)
    validaccuracy_knei = accuracy_score(y_valid, y_pred_knei_valid)
    validaccuracy_gdbt = accuracy_score(y_valid, y_pred_gdbt_valid)
    validaccuracy_xgb = accuracy_score(y_valid, y_pred_xgb_valid)
    validaccuracy_rf = accuracy_score(y_valid, y_pred_rf_valid)
    print(f"Validation Accuracy SVM: {validaccuracy_svm}")
    print(f"Validation Accuracy LR: {validaccuracy_lr}")
    print(f"Validation Accuracy MLP: {validaccuracy_mlp}")
    print(f"Validation Accuracy KNEI: {validaccuracy_knei}")
    print(f"Validation Accuracy gdbt: {validaccuracy_gdbt}")
    print(f"Validation Accuracy XGB: {validaccuracy_xgb}")
    print(f"Validation Accuracy RF: {validaccuracy_rf}")
    #七种机器学习方法
    y_pred_valids_lr.append(y_pred_lr_valid) #y_pred_lr_valid 被添加到 y_pred_valids_lr 列表中，y_pred_lr_valid是模型在当前验证集上的预测结果，而y_pred_valids_lr是用于储存每次模型验证过程中的预测结果
    y_pred_valids_svm.append(y_pred_svm_valid)
    y_pred_valids_mlp.append(y_pred_mlp_valid)
    y_pred_valids_knei.append(y_pred_knei_valid)
    y_pred_valids_gdbt.append(y_pred_gdbt_valid)
    y_pred_valids_xgb.append(y_pred_xgb_valid)
    y_pred_valids_rf.append(y_pred_rf_valid)
    y_score_valids_lr.append(lr.decision_function(X_valid_cross))
    y_score_valids_svm.append(svm.decision_function(X_valid_cross))
    y_score_valids_mlp.append(mlp.predict_proba(X_valid_cross)[:, 1])  #选择predict-proba返回的属于正类的概率
    y_score_valids_knei.append(knei.predict_proba(X_valid_cross)[:, 1])
    y_score_valids_gdbt.append(gdbt.predict_proba(X_valid_cross)[:, 1])
    y_score_valids_xgb.append(xgb.predict_proba(X_valid_cross)[:, 1])
    y_score_valids_rf.append(rf.predict_proba(X_valid_cross)[:, 1])
    #结合模型，1&234只选择svm
    y_pred_valids_combine.append(y_pred_lr_valid)
    y_pred_valids_combine.append(y_pred_svm_valid)
    y_pred_valids_combine.append(y_pred_mlp_valid)
    y_score_lr_valid = lr.decision_function(X_valid_cross)
    y_score_svm_valid = svm.decision_function(X_valid_cross)
    y_score_mlp_valid = mlp.predict_proba(X_valid_cross)[:, 1]
    y_score_valids_combine.append(y_score_lr_valid)
    y_score_valids_combine.append(y_score_svm_valid)
    y_score_valids_combine.append(y_score_mlp_valid)


#七种方法评估
mean_y_pred_valids_lr = mode(y_pred_valids_lr, axis=0).mode  #计算y_pred_valids_lr中每个元素的众数
mean_y_score_valids_lr = np.mean(y_score_valids_lr, axis=0)   #计算y_score_valids_lr中每个元素的平均值
mean_valid_acc_lr = accuracy_score(y_valid, mean_y_pred_valids_lr)  #用于存储计算出的准确率
mean_conf_mat_lr = confusion_matrix(y_valid, mean_y_pred_valids_lr, labels=[0, 1])  #用于存储计算出的混淆矩阵
fpr_lr, tpr_lr, thres_lr = metrics.roc_curve(y_valid, mean_y_score_valids_lr)   #用于计算假正例率（FPR）、真正例率（TPR）和阈值（thresholds）
roc_auc_lr = metrics.auc(fpr_lr, tpr_lr) #计算AUC

mean_y_pred_valids_svm = mode(y_pred_valids_svm, axis=0).mode
mean_y_score_valids_svm = np.mean(y_score_valids_svm, axis=0)
mean_valid_acc_svm = accuracy_score(y_valid, mean_y_pred_valids_svm)
mean_conf_mat_svm = confusion_matrix(y_valid, mean_y_pred_valids_svm, labels=[0, 1])
fpr_svm, tpr_svm, thres_svm = metrics.roc_curve(y_valid, mean_y_score_valids_svm)
roc_auc_svm = metrics.auc(fpr_svm, tpr_svm)

mean_y_pred_valids_mlp = mode(y_pred_valids_mlp, axis=0).mode
mean_y_score_valids_mlp = np.mean(y_score_valids_mlp, axis=0)
mean_valid_acc_mlp = accuracy_score(y_valid, mean_y_pred_valids_mlp)
mean_conf_mat_mlp = confusion_matrix(y_valid, mean_y_pred_valids_mlp, labels=[0, 1])
fpr_mlp, tpr_mlp, thres_mlp = metrics.roc_curve(y_valid, mean_y_score_valids_mlp)
roc_auc_mlp = metrics.auc(fpr_mlp, tpr_mlp)

mean_y_pred_valids_knei = mode(y_pred_valids_knei, axis=0).mode
mean_y_score_valids_knei = np.mean(y_score_valids_knei, axis=0)
mean_valid_acc_knei = accuracy_score(y_valid, mean_y_pred_valids_knei)
mean_conf_mat_knei = confusion_matrix(y_valid, mean_y_pred_valids_knei, labels=[0, 1])
fpr_knei, tpr_knei, thres_knei = metrics.roc_curve(y_valid, mean_y_score_valids_knei)
roc_auc_knei = metrics.auc(fpr_knei, tpr_knei)

mean_y_pred_valids_gdbt = mode(y_pred_valids_gdbt, axis=0).mode
mean_y_score_valids_gdbt = np.mean(y_score_valids_gdbt, axis=0)
mean_valid_acc_gdbt = accuracy_score(y_valid, mean_y_pred_valids_gdbt)
mean_conf_mat_gdbt = confusion_matrix(y_valid, mean_y_pred_valids_gdbt, labels=[0, 1])
fpr_gdbt, tpr_gdbt, thres_gdbt = metrics.roc_curve(y_valid, mean_y_score_valids_gdbt)
roc_auc_gdbt = metrics.auc(fpr_gdbt, tpr_gdbt)

mean_y_pred_valids_xgb = mode(y_pred_valids_xgb, axis=0).mode
mean_y_score_valids_xgb = np.mean(y_score_valids_xgb, axis=0)
mean_valid_acc_xgb = accuracy_score(y_valid, mean_y_pred_valids_xgb)
mean_conf_mat_xgb = confusion_matrix(y_valid, mean_y_pred_valids_xgb, labels=[0, 1])
fpr_xgb, tpr_xgb, thres_xgb = metrics.roc_curve(y_valid, mean_y_score_valids_xgb)
roc_auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)

mean_y_pred_valids_rf = mode(y_pred_valids_rf, axis=0).mode
mean_y_score_valids_rf = np.mean(y_score_valids_rf, axis=0)
mean_valid_acc_rf = accuracy_score(y_valid, mean_y_pred_valids_rf)
mean_conf_mat_rf = confusion_matrix(y_valid, mean_y_pred_valids_rf, labels=[0, 1])
fpr_rf, tpr_rf, thres_rf = metrics.roc_curve(y_valid, mean_y_score_valids_rf)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)

#结合模型
mean_y_pred_valids_combine = mode(y_pred_valids_combine, axis=0).mode
mean_y_score_valids_combine = np.mean(y_score_valids_combine, axis=0)
mean_valid_acc_combine = accuracy_score(y_valid, mean_y_pred_valids_combine)
mean_conf_mat_combine = confusion_matrix(y_valid, mean_y_pred_valids_combine, labels=[0, 1])
fpr_combine, tpr_combine, thres_combine = metrics.roc_curve(y_valid, mean_y_score_valids_combine)
roc_auc_combine = metrics.auc(fpr_combine, tpr_combine)
#将数据保存为CSV文件
#np.savetxt('combine模型预测标签4&123.csv', [mean_y_pred_valids_combine ], delimiter=',')
#np.savetxt('真实标签4&123.csv', [y_valid], delimiter=',')


#print(f"Test Accuracy: {mean_test_acc1}")
print(f"Combine Validation Accuracy: {mean_valid_acc_combine}")
print(f"Combine Confusion Matrix:\n{mean_conf_mat_combine}")

def machinelearning_roc():
    # 设置字体类型及大小
    Font = {'size': 10, 'family': 'Arial'}
    # 绘制ROC曲线图并计算因此所得的各分类器的最大KS值
    plt.figure(figsize=(300 / 72, 300 / 72))  #180为宽和高的像素，72为分辨率
    plt.plot(fpr_svm, tpr_svm, label='SVM = %0.3f' % roc_auc_svm, color='#d62424', linewidth=0.5)
    plt.plot(fpr_lr, tpr_lr, label='LR = %0.3f' % roc_auc_lr, color='#ff7c17', linewidth=0.5)
    plt.plot(fpr_mlp, tpr_mlp, label='MLP = %0.3f' % roc_auc_mlp, color='#fec603', linewidth=0.5)
    plt.plot(fpr_xgb, tpr_xgb, label='XGBoost = %0.3f' % roc_auc_xgb, color='#3aba8f', linewidth=0.5)
    plt.plot(fpr_gdbt, tpr_gdbt, label='GDBT = %0.3f' % roc_auc_gdbt, color='#16d8fa', linewidth=0.5)
    plt.plot(fpr_rf, tpr_rf, label='RF = %0.3f' % roc_auc_rf, color='#5274ff', linewidth=0.5)
    plt.plot(fpr_knei, tpr_knei, label='K-Neighbors = %0.3f' % roc_auc_knei, color='#af64e8', linewidth=0.5)

    plt.legend(loc='lower right', prop=Font)  #设置图例，loc用于指定图例的位置，prop指定图例的属性
    plt.plot([0, 1], [0, 1], 'r--', linewidth=0.5)  #绘制一条对角线虚线，r代表红色，--代表虚线danshed
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=10)

    plt.savefig("七种机器学习方法评估1&234.svg")
    #plt.show()

machinelearning_roc()

def combine_roc():  #包含lr、mlp、svm算法

    # 设置字体类型及大小
    Font = {'size': 10, 'family': 'Arial'}

    plt.figure(figsize=(300/72, 300/72))

    plt.plot(fpr_combine, tpr_combine, label='Combine = %0.3f' % roc_auc_combine, color='#d62424',linewidth = 0.5)


    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'k--',linewidth=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=10)

    #plt.savefig("4&123-combine_roc_lr&svm&mlp.svg")
    #plt.show()

#combine_roc()

def combine_cm():  #结合算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_combine, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_combine))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_combine.max() / 2.

    for i in range(len(mean_conf_mat_combine)):
        for j in range(len(mean_conf_mat_combine[i])):
            plt.text(j, i, format(mean_conf_mat_combine[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_combine[i, j] > thresh else "black")

    plt.savefig(
        "1&234-combine_cm.svg")
    #plt.show()
#combine_cm()

def svm_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_svm, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_svm))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_svm.max() / 2.

    for i in range(len(mean_conf_mat_svm)):
        for j in range(len(mean_conf_mat_svm[i])):
            plt.text(j, i, format(mean_conf_mat_svm[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_svm[i, j] > thresh else "black")

    plt.savefig(
        "1&234-svm_cm.svg")
    #plt.show()
svm_cm()

def combine_cm():  #结合算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_combine, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_combine))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_combine.max() / 2.

    for i in range(len(mean_conf_mat_combine)):
        for j in range(len(mean_conf_mat_combine[i])):
            plt.text(j, i, format(mean_conf_mat_combine[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_combine[i, j] > thresh else "black")

    plt.savefig(
        "1&234-combine_cm.svg")
    #plt.show()
#combine_cm()

def lr_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_lr, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_lr))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_lr.max() / 2.

    for i in range(len(mean_conf_mat_lr)):
        for j in range(len(mean_conf_mat_lr[i])):
            plt.text(j, i, format(mean_conf_mat_lr[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_lr[i, j] > thresh else "black")

    plt.savefig(
        "1&234-lr_cm.svg")
    #plt.show()
lr_cm()

def mlp_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_mlp, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_mlp))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_mlp.max() / 2.

    for i in range(len(mean_conf_mat_mlp)):
        for j in range(len(mean_conf_mat_mlp[i])):
            plt.text(j, i, format(mean_conf_mat_mlp[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_mlp[i, j] > thresh else "black")

    plt.savefig(
        "1&234-mlp_cm.svg")
    #plt.show()
mlp_cm()

def knei_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_knei, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_knei))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_knei.max() / 2.

    for i in range(len(mean_conf_mat_knei)):
        for j in range(len(mean_conf_mat_knei[i])):
            plt.text(j, i, format(mean_conf_mat_knei[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_knei[i, j] > thresh else "black")

    plt.savefig(
        "1&234-knei_cm.svg")
    #plt.show()
knei_cm()

def gdbt_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_gdbt, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_gdbt))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_gdbt.max() / 2.

    for i in range(len(mean_conf_mat_gdbt)):
        for j in range(len(mean_conf_mat_gdbt[i])):
            plt.text(j, i, format(mean_conf_mat_gdbt[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_gdbt[i, j] > thresh else "black")

    plt.savefig(
        "1&234-gdbt_cm.svg")
    #plt.show()
gdbt_cm()

def xgb_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_xgb, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_xgb))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_xgb.max() / 2.

    for i in range(len(mean_conf_mat_xgb)):
        for j in range(len(mean_conf_mat_xgb[i])):
            plt.text(j, i, format(mean_conf_mat_xgb[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_xgb[i, j] > thresh else "black")

    plt.savefig(
        "1&234-xgb_cm.svg")
    #plt.show()
xgb_cm()

def rf_cm():  #svm算法的混淆矩阵
    # 设置图片大小
    plt.figure(figsize=(300 / 72, 300 / 72))

    # 绘制热度图
    plt.imshow(mean_conf_mat_rf, cmap=plt.cm.Blues)
    colorbar = plt.colorbar()
    colorbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
    colorbar.ax.tick_params(labelsize=10)  # 设置colorbar旁边的数字大小为5pt

    # 设置坐标轴显示列表
    indices = range(len(mean_conf_mat_rf))
    classes = ['234', '1']  #类别可修改
    plt.xticks(indices, classes, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 设置坐标轴标题、字体
    plt.ylabel('True', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.tick_params(labelsize=10)

    # 显示数据
    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = mean_conf_mat_rf.max() / 2.

    for i in range(len(mean_conf_mat_rf)):
        for j in range(len(mean_conf_mat_rf[i])):
            plt.text(j, i, format(mean_conf_mat_rf[i][j], fmt),
                     fontsize=10,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if mean_conf_mat_rf[i, j] > thresh else "black")

    plt.savefig(
        "1&234-rf_cm.svg")
    #plt.show()
rf_cm()

print("end")

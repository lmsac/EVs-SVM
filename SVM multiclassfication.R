library(e1071)
library(caret)
library(MASS)
library(reshape2)
library(ggplot2)
library(kernlab)
library(dplyr)
library(pROC)
library(ggpubr)
library(caTools) #有sample.split函数，用原图将数据按预定义的比例分成两组,同时保留数据中不同标签的相对比率，用于将数据分为训练和测试子集
library(ROCR)
#library(MetaboAnalystR) #用于数据的归一化
library(tidyverse)

pima <- read.csv("D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machineLearning/multiclassification/0729result_normalized.csv", row.names = 1) #修改导入数据
pima <- pima[order(rownames(pima)), ] #对数据进行排序

#metaboanalyst进行数据归一化处理
mSet<-InitDataObjects("pktable", "stat", FALSE)
mSet<-Read.TextData(mSet, "E:/Study/R/supplement/buchong/final.csv", "rowu", "disc")
mSet<-SanityCheckData(mSet)
mSet<-ReplaceMin(mSet)
mSet<-SanityCheckData(mSet)
mSet<-PreparePrenormData(mSet)
mSet<-Normalization(mSet, "SumNorm", "LogNorm", "AutoNorm", ratio=FALSE, ratioNum=20)

pima.scale <- pima #若数据已经过归一化处理

#其余数据处理
#pima.scale <- mSet[["dataSet"]][["norm"]]
pima.scale <- pima.scale[order(rownames(pima.scale)), ] #对归一化后的数据进行排序
colnames(pima.scale) <- paste0("X", colnames(pima.scale)) #在峰值前加X
pima.scale$Label <- factor(pima.scale$Label) #将标签的数字转化成文本

#权重筛选
forrfe <- cbind(pima.scale$Label, pima.scale[, -ncol(pima.scale)])
rfecolnames <- colnames(forrfe)
rfecolnames[1] <- 'Label'
colnames(forrfe) <- rfecolnames

#随机划分训练集和测试集
set.seed(205)
ind <- sample(2, 80, replace = TRUE, prob = c(0.75, 0.25)) #划分为训练集和测试集两类，样本数，概率
ind3 <- rep(ind, each = 3)
train <- pima.scale[ind3 == 1, ]
test <- pima.scale[ind3 == 2, ]

#分层抽样
set.seed(266)
# 创建样本ID和类别标签
sample_ids <- rep(1:80, each = 3)  # 每个样本重复3次
class_labels <- rep(1:4, each = 60) # 每个类别60张谱图(20个样本×3)
# 按类别分层抽样，确保每个类别在训练集和测试集中分布均匀
train_indices <- unlist(lapply(1:4, function(cls) {
  cls_samples <- unique(sample_ids[class_labels == cls])  # 获取当前类别的所有样本ID
  train_samples <- sample(cls_samples, size = 14, replace = FALSE)  # 每个类别选14个样本到训练集
  which(sample_ids %in% train_samples)  # 返回这些样本的所有3次重复的索引
}))
# 创建索引向量(1=训练集，2=测试集)
ind <- rep(2, length(sample_ids))  # 初始全设为测试集
ind[train_indices] <- 1  # 训练集样本设为1
# 检查分布
table(class_labels[ind == 1]) 
table(class_labels[ind == 2]) 
# 划分数据集
train <- pima.scale[ind == 1, ]
test <- pima.scale[ind == 2, ]
# 验证样本数
nrow(train)  # 应为168 (56样本×3)
nrow(test)   # 应为72 (24样本×3)
write.csv(test, file = "D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machineLearning-R/multiclassification/20250618修改/test0618.csv")  #修改文件名称
write.csv(train, file = "D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machineLearning-R/multiclassification/20250618修改/train0618.csv")  #修改文件名称

# SVM-linear
set.seed(134)
linear.tune <- tune.svm(Label ~., data = train, kernal = "linear",
                        cost = c(0.001, 0.01, 0.1, 1, 5, 10),probability = TRUE)
summary(linear.tune)
linear.tune <- tune.svm(Label ~., data = train, kernal = "linear",
                         cost = seq(0.1, 5, 0.1),probability = TRUE) #seq表示从0.1-5，步长为0.1
summary(linear.tune)
best.linear <- linear.tune$best.model
linear.test <- predict(best.linear, newdata = test, probability = TRUE, decision.values = TRUE) #判别分数
linear.tab <- table(linear.test, test$Label)
linear.tab
sum(diag(linear.tab))/sum(linear.tab) #计算准确率

# SVM-polynomial
set.seed(123)
poly.tune <- tune.svm(Label ~., data = train, kernel = "polynomial",probability = TRUE,
                      degree = c(3, 4, 5), coef0 = c(0.1, 0.5, 1, 2, 3, 4))
summary(poly.tune)

poly.tune <- tune.svm(Label ~ ., data = train, kernel = "polynomial",probability = TRUE,
                      degree = seq(3, 5, 0.1), coef0 = seq(0, 0.5, 0.1))
summary(poly.tune)
best.poly <- poly.tune$best.model
best.poly
poly.train <- predict(best.poly, newdata = train, probability = TRUE)
poly.test <- predict(best.poly, newdata = test, probability = TRUE)
poly.cm <- table(poly.train, train$Label)
poly.cm
sum(diag(poly.cm))/sum(poly.cm)
poly.tab <- table(poly.test, test$Label)
poly.tab
sum(diag(poly.tab))/sum(poly.tab)

# SVM-radial basis function
set.seed(123)
rbf.tune <- tune.svm(Label ~., data = train, kernal = "radial", probability = TRUE,
                     gamma = c(0.1, 0.5, 1, 2, 3, 4))
summary(rbf.tune)
best.rbf <- rbf.tune$best.model
rbf.test <- predict(best.rbf, newdata = test, probability = TRUE)
rbf.tab <- table(rbf.test, test$Label)
rbf.tab
sum(diag(rbf.tab))/sum(rbf.tab)

# SVM-sigmoid
set.seed(123)
sigmod.tune <- tune.svm(Label ~., data = train, kernal = "sigmoid", probability = TRUE,
                        gamma = c(0.1, 0.5, 1, 2, 3, 4), 
                        coef0 = c(0.1, 0.5, 1, 2, 3, 4))
summary(sigmod.tune)
best.sigmod <- sigmod.tune$best.model
sigmod.test <- predict(best.sigmod, newdata = test, probability = TRUE,)
sigmod.tab <- table(sigmod.test, test$Label)
sigmod.tab
sum(diag(sigmod.tab))/sum(sigmod.tab)

# 模型比较
confusionMatrix(linear.test, test$Label)
confusionMatrix(poly.test, test$Label)
confusionMatrix(poly.train, train$Label)
confusionMatrix(rbf.test, test$Label)
confusionMatrix(sigmod.test, test$Label)

#绘制roc曲线，二分类
prob_estimates <- attr(poly.test, "probabilities")  #此行代码提取出效果最好的模型预测的测试集的概率
write.csv(prob_estimates, file = "D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machineLearning-R/multiclassification/20250618修改/test_probability_R.csv")  #修改文件名称
data_pre <- data.frame(prob = prob_estimates[,1],obs = test$Label) #data_pre中包含预测为阳性的概率和实际的标签
data_pre <- data_pre[order(data_pre$prob),]   #排序



modelroc <- roc(test$Label,prob_estimates[,1],direction=c("auto"),
                plot = T, legacy.axes = TRUE, thresholds="best", print.thres="best", xlab = "False positive rate", 
                ylab = "True postitive rate",print.auc = TRUE,levels = c("2", "134"))
ci.auc(modelroc) # 95%置信区间,levels设定类别标签（对照,疾病）,direction设定比较方向c("auto", "<", ">")


#最佳阈值点
tiff("/Users/apple/Desktop/Poly.tiff", width=5, height=5,units = "in",res=600) #修改名称
plot(modelroc,
     legacy.axes= TRUE,
     main="ROC-Poly", #修改名称
     thresholds="best", # 基于youden指数选择roc曲线最佳阈值点
     print.thres="best",
     print.auc = TRUE,
     print.ci.auc = TRUE,
     levels = c("4", "123")) # 在roc曲线上显示最佳阈值点
text(0.38,0.4, labels = '(95% CI: 0.9333-1)', font = 1) #修改置信区间
roc_result <- coords(modelroc, "best")
dev.off()

#plot(1-modelroc$specificities, modelroc$sensitivities, type = "l", lty = 1, lwd = 2, col = "red", xlab = "False positive rate", ylab = "True positive rate", xlim = c(0,1), ylim = c(0,1), asp = 1) 
#参数legacy.axes = TRUE是把纵坐标设置为1-sensitivity,然后方便修改纵坐标
#text(0.8,0.15, labels = 'AUC:998', font = 1)#AUC值由modelroc得到
#type,lty,lwd都是线的一些参数，xlab、ylab设置的是横纵坐标的名称

#混淆矩阵
tiff("/Users/apple/Desktop/Linear-Poly.tiff", width=5, height=5,units = "in",res=600)
plot_confusion = function(cm) {
  as.table(cm) %>% 
    as_tibble() %>% 
    mutate(response = factor(response),
           truth = factor(truth, rev(levels(response)))) %>% 
    ggplot(aes(response, truth, fill = n)) +
    geom_tile() +
    geom_text(aes(label = n)) +
    scale_fill_gradientn(colors = rev(hcl.colors(60, "Blues")), breaks = seq(0,60,10), limits = c(0, 60)) +
    coord_fixed() +
    theme_minimal()
}

#根据前面结果创建混淆矩阵
species = c("1", "2", "3", "4")
cm = matrix(c(11,0,0,0,
              1,19,3,0,
              0,2,12,1,
              0,3,3,5), nrow = 4,
            dimnames = list(truth = species, response = species))


plot_confusion(cm)
dev.off()

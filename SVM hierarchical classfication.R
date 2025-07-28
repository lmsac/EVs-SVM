library(e1071)
library(caret)
library(pROC)
library(dplyr)

# 读取数据
pima <- read.csv("D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-machineLearning/multiclassification/0729result_normalized.csv", row.names = 1)
pima <- pima[order(rownames(pima)), ] # 对数据进行排序

# 数据处理
pima.scale <- pima # 使用归一化后的数据
pima.scale$Label <- factor(pima.scale$Label) # 将标签转化为因子

# 分层抽样 - 确保每个类别的样本在训练集和测试集中均匀分布
set.seed(266)
sample_ids <- rep(1:80, each = 3)  # 每个样本重复3次
class_labels <- rep(1:4, each = 60) # 每个类别60张谱图(20个样本×3)

# 按类别分层抽样
train_indices <- unlist(lapply(1:4, function(cls) {
  cls_samples <- unique(sample_ids[class_labels == cls])
  train_samples <- sample(cls_samples, size = 14, replace = FALSE)  # 每个类别选15个样本到训练集
  which(sample_ids %in% train_samples)
}))

# 创建索引向量
ind <- rep(2, length(sample_ids))
ind[train_indices] <- 1

# 划分数据集
train <- pima.scale[ind == 1, ]
test <- pima.scale[ind == 2, ]

# 第一步：分离类型1与其他类型 (2,3,4)
cat("\n===== 第一步：分离类型1与其他类型 (2,3,4) =====\n")

# 创建二分类标签
train_step1 <- train
test_step1 <- test

train_step1$BinaryLabel <- factor(ifelse(train_step1$Label == 1, "Type1", "Other"))
test_step1$BinaryLabel <- factor(ifelse(test_step1$Label == 1, "Type1", "Other"))

# 第一步参数调优
set.seed(723)
step1_tune <- tune.svm(
  BinaryLabel ~ . - Label, 
  data = train_step1, 
  kernel = "polynomial",
  probability = TRUE,
  degree = c(3, 4, 5), 
  coef0 = c(0.1, 0.5, 1, 2, 3, 4)
)

# 获取最佳模型
step1_model <- step1_tune$best.model
cat("第一步最佳参数:\n")
cat("degree =", step1_model$degree, "\n")
cat("coef0 =", step1_model$coef0, "\n")

# 第一步预测和评估
step1_pred <- predict(step1_model, newdata = test_step1, probability = TRUE)
step1_prob <- attr(step1_pred, "probabilities")[, "Type1"]

# 混淆矩阵
cat("\n第一步混淆矩阵:\n")
step1_cm <- confusionMatrix(step1_pred, test_step1$BinaryLabel)
print(step1_cm)

# ROC曲线
roc_step1 <- roc(response = as.numeric(test_step1$BinaryLabel == "Type1"),
                 predictor = step1_prob,
                 levels = c(0, 1))
ci.auc(roc_step1)
plot(roc_step1, main = "第一步ROC曲线 (类型1 vs 其他)", col = "blue", legacy.axes = TRUE)
text(0.6, 0.2, paste0("AUC = ", round(auc(roc_step1), 3)), col = "blue")
cat("\n第一步AUC值:", round(auc(roc_step1), 3), "\n")

# 患者分类情况
step1_results <- data.frame(
  SampleID = rownames(test_step1),
  TrueLabel = test_step1$Label,
  PredictedStep1 = step1_pred,
  Step1_Probability = step1_prob
)
cat("\n第一步患者分类情况:\n")
print(step1_results)

# 第二步：在第一步预测为非类型1的患者中分离类型3与其他类型 (2,4)
cat("\n===== 第二步：在第一步预测为非类型1的患者中分离类型3与其他类型 (2,4) =====\n")

# 获取非类型1的训练数据（使用真实标签）
non_type1_train <- train_step1[train_step1$Label != 1, ]

# 获取第一步预测为非类型1的测试数据（基于预测结果）
non_type1_test <- test_step1[step1_pred == "Other", ]

# 创建二分类标签
non_type1_train$BinaryLabel2 <- factor(ifelse(non_type1_train$Label == 3, "Type3", "Other"))
non_type1_test$BinaryLabel2 <- factor(ifelse(non_type1_test$Label == 3, "Type3", "Other"))

# 第二步参数调优
set.seed(216)
step2_tune <- tune.svm(
  BinaryLabel2 ~ . - Label - BinaryLabel, 
  data = non_type1_train,
  kernel = "polynomial",
  probability = TRUE,
  degree = c(3, 4, 5), 
  coef0 = c(0.1, 0.5, 1, 2, 3, 4)
)

# 获取最佳模型
step2_model <- step2_tune$best.model
cat("第二步最佳参数:\n")
cat("degree =", step2_model$degree, "\n")
cat("coef0 =", step2_model$coef0, "\n")

# 第二步预测和评估
step2_pred <- predict(step2_model, newdata = non_type1_test, probability = TRUE)
step2_prob <- attr(step2_pred, "probabilities")[, "Type3"]

# 混淆矩阵
cat("\n第二步混淆矩阵:\n")
step2_cm <- confusionMatrix(step2_pred, non_type1_test$BinaryLabel2)
print(step2_cm)

# ROC曲线
roc_step2 <- roc(response = as.numeric(non_type1_test$BinaryLabel2 == "Type3"),
                 predictor = step2_prob,
                 levels = c(0, 1))
ci.auc(roc_step2)
plot(roc_step2, main = "第二步ROC曲线 (类型3 vs 其他)", col = "red", legacy.axes = TRUE)
text(0.6, 0.2, paste0("AUC = ", round(auc(roc_step2), 3)), col = "red")
cat("\n第二步AUC值:", round(auc(roc_step2), 3), "\n")

# 患者分类情况
step2_results <- data.frame(
  SampleID = rownames(non_type1_test),
  TrueLabel = non_type1_test$Label,
  PredictedStep2 = step2_pred,
  Step2_Probability = step2_prob
)
cat("\n第二步患者分类情况:\n")
print(step2_results)



# 第三步：在第二步预测为非类型3的患者中分离类型4与其他类型 (类型2)
cat("\n===== 第三步：在第二步预测为非类型3的患者中分离类型4与其他类型 (类型2) =====\n")

# 获取特征列名（排除所有标签列）
feature_names <- setdiff(names(train), "Label")

# 准备训练数据
non_type1_3_train <- non_type1_train[non_type1_train$BinaryLabel2 == "Other", ] %>%
  filter(Label %in% c(2, 4)) %>%
  mutate(BinaryLabel3 = factor(ifelse(Label == 4, "Type4", "Other")))

# 准备测试数据
non_type1_3_test <- non_type1_test[step2_pred == "Other", ] %>%
  mutate(BinaryLabel3_eval = factor(ifelse(Label == 4, "Type4", "Other")))

# 第三步参数调优 - 使用纯特征数据
set.seed(214)
step3_tune <- tune.svm(
  x = non_type1_3_train[, feature_names],  # 仅特征矩阵
  y = non_type1_3_train$BinaryLabel3,      # 目标变量
  kernel = "polynomial",
  probability = TRUE,
  degree = c(3, 4, 5), 
  coef0 = c(0.1, 0.5, 1, 2, 3, 4)
)

# 获取最佳模型
step3_model <- step3_tune$best.model
cat("第三步最佳参数:\n")
cat("degree =", step3_model$degree, "\n")
cat("coef0 =", step3_model$coef0, "\n")

# 第三步预测 - 仅使用特征
step3_pred <- predict(
  step3_model, 
  newdata = non_type1_3_test[, feature_names],
  probability = TRUE
)
step3_prob <- attr(step3_pred, "probabilities")[, "Type4"]

# 混淆矩阵
cat("\n第三步混淆矩阵 (类型4 vs 其他(类型2)):\n")
step3_cm <- confusionMatrix(step3_pred, non_type1_3_test$BinaryLabel3_eval)
print(step3_cm)

# ROC曲线
roc_step3 <- roc(
  response = as.numeric(non_type1_3_test$BinaryLabel3_eval == "Type4"),
  predictor = step3_prob,
  levels = c(0, 1)
)
plot(roc_step3, main = "第三步ROC曲线 (类型4 vs 其他(类型2))", col = "green", legacy.axes = TRUE)
text(0.6, 0.2, paste0("AUC = ", round(auc(roc_step3), 3)), col = "green")
ci.auc(roc_step3)
cat("\n第三步AUC值:", round(auc(roc_step3), 3), "\n")

# 患者分类情况
step3_results <- data.frame(
  SampleID = rownames(non_type1_3_test),
  TrueLabel = non_type1_3_test$Label,
  PredictedBinary = step3_pred,
  Step3_Probability = step3_prob
) %>%
  mutate(PredictedLabel = ifelse(PredictedBinary == "Type4", 4, 2))

cat("\n第三步患者分类情况:\n")
print(step3_results)





# 整合最终结果
final_results <- data.frame(
  SampleID = rownames(test),
  TrueLabel = test$Label
) %>%
  left_join(select(step1_results, SampleID, PredictedStep1, Step1_Probability), by = "SampleID") %>%
  left_join(select(step2_results, SampleID, PredictedStep2, Step2_Probability), by = "SampleID") %>%
  left_join(select(step3_results, SampleID, PredictedLabel, Step3_Probability), by = "SampleID")

# 生成最终预测标签
final_results$FinalPrediction <- with(final_results, {
  ifelse(PredictedStep1 == "Type1", 1,
         ifelse(PredictedStep2 == "Type3", 3,
                as.character(PredictedLabel)))
})

# 确保预测标签为因子
final_results$FinalPrediction <- factor(final_results$FinalPrediction, levels = c(1, 2, 3, 4))

cat("\n最终分类结果:\n")
print(final_results)
write.csv(final_results, file = "D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-决策树分类/finalresults.csv")  #修改文件名称
# 最终混淆矩阵 (四分类)
final_cm <- confusionMatrix(
  final_results$FinalPrediction,
  factor(final_results$TrueLabel, levels = c(1, 2, 3, 4))
)
cat("\n整体混淆矩阵 (四分类):\n")
print(final_cm)
write.csv(final_cm, file = "D:/yjs/MALDIquant/课题1_MALDI_molecular type/20240808-SVM机器学习/SVM-决策树分类/finalcm.csv")  #修改文件名称

# 绘制所有ROC曲线
par(mfrow = c(1, 3))
plot(roc_step1, main = "Step1: Type1 vs Others", col = "blue", legacy.axes = TRUE)
text(0.6, 0.2, paste0("AUC = ", round(auc(roc_step1), 3)), col = "blue")
plot(roc_step2, main = "Step2: Type3 vs Others", col = "red", legacy.axes = TRUE)
text(0.6, 0.2, paste0("AUC = ", round(auc(roc_step2), 3)), col = "red")
plot(roc_step3, main = "Step3: Type2 vs Type4", col = "green", legacy.axes = TRUE)
text(0.6, 0.2, paste0("AUC = ", round(auc(roc_step3), 3)), col = "green")
par(mfrow = c(1, 1))


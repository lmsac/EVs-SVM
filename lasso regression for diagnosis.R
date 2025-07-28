library(dplyr)
library(limma)
library(glmnet)
data <- read.csv("D:/yjs/MALDIquant/课题0_MALDI_cancer diagnose/20241006特征选择修改/data_normalized.csv", row.names=1)
control <- data[,c(1:20)]  
disease <- data[,c(21:40)] 
set.seed(2000)
num_control <- sample(c(1:ncol(control)),round(3*ncol(control)/4)) #从对照组 (control) 中随机抽取 75% 的样本索引
num_disease <- sample(c(1:ncol(disease)),round(3*ncol(disease)/4))

train <- cbind(control[,num_control],disease[,num_disease]) 
test <- cbind(control[,-num_control],disease[,-num_disease]) 
write.csv(train,"D:/train.csv")
write.csv(test,"D:/test.csv")

pvalue <- c() #用于储存每个特征的p值
ratio <- c()  #用于存储每个特征的疾病组相对于对照组的平均表达比值
my.t.test.p.value <- function(...) {
  obj<-try(wilcox.test(...), silent=TRUE) #进行Wilcoxon秩和检验
  if (is(obj, "try-error")) return(NA) else return(obj$p.value) # 如果出错返回NA，否则返回p值
}
for(i in 1:nrow(train)){
  bb1 <- as.numeric(train[i,])
  # 执行组间差异检验
  pvalue[i]<- my.t.test.p.value(bb1[c(1:length(num_control))],bb1[c((length(num_control)+1):ncol(train))])
  ratio[i] <- mean(bb1[c((length(num_control)+1):ncol(train))])/mean(bb1[c(1:length(num_control))]) #2^(mean()-mean())
} # 计算疾病组/对照组的表达量比值
BH<- p.adjust(pvalue,method="BH") # Benjamini-Hochberg 校正

traintemp <- train
traintemp$pvalue <- pvalue
traintemp$ratio <- ratio
traintemp$BH <- BH #添加原始p值列，比值列，和校正p值列

# 筛选显著差异特征
traintemp2 <- filter(traintemp,BH<0.05&(ratio>1.2|ratio<0.83)) #p<0.05且FC>1.2
traindif <- traintemp2[,c(1:(ncol(traintemp2)-3))] # 提取差异表达量数据子集

# 差异表达数据的组别分割
result <- data.frame()
control2 <- traindif[,c(1:length(num_control))]
disease2 <- traindif[,c((length(num_control)+1):ncol(traindif))]

for (i in 1:200){
  datadisease <- as.matrix(t(sample(disease2,round(0.8*ncol(disease2)), replace = TRUE))) #有放回地抽取80%的样本，通过重复抽样评估特征选择的稳定性
  datacontrol <- as.matrix(t(sample(control2,round(0.8*ncol(control2)), replace = TRUE)))
  dataxun <- rbind(datacontrol,datadisease) #合并数据
  yy=c(rep(0,nrow(datacontrol)),rep(1,nrow(datadisease))) #创建二进制标签向量：0对照组，1疾病组
  cv.lasso <- cv.glmnet(dataxun, yy,alpha = 1, family = "binomial",nfolds=10) #使用10折交叉验证，选择最优L1正则化参数lambda
  fit2 <- glmnet(dataxun, yy, alpha = 1, family = "binomial",
                 lambda = cv.lasso$lambda.min) #通过交叉验证找到最优lambda，拟合最终Lasso模型
  result_name <-rownames(fit2$beta)
  result_a <- result_name[which(fit2$beta!=0)] #提取非零系数的特征名
  for (j in 1:length(result_a)){
    result[j,i] <-result_a[j] #1 and i
  } #将每个特征名放入result的第j行第i列
} #进行200次自助抽样（bootstrap Sampling），每次抽样构建一个训练集（80%的样本），然后用Lasso回归选择特征，记录每次被选中的特征。最后，我们可以统计每个特征在200次中被选中的频率，以此评估特征的重要性。

xy.list <- split(result, seq(nrow(result))) #将结果矩阵 result 按行拆分成列表
result_list <- unlist(xy.list) #将列表展平为单个向量
result_all<- as.data.frame(table(result_list)) #统计特征出现频率
write.csv(result_all,"/Users/apple/Desktop/lassoresult-2.csv")

#回归系数路径图和交叉验证曲线
mydata <- read.csv("/Users/apple/Desktop/train-T.csv")
colnames(mydata[,1:15])#查看前15列的列名（根据自己数据调整）
y <- as.matrix(mydata[, 1])  # 提取第1列作为结局（建议放在第一列）
x <- as.matrix(mydata[, 2:151])  # 第2至第17列为自变量

lasso_model <- glmnet(x, y, family = "binomial",
                      alpha = 1) # 表示采用L1正则化，即Lasso回归。
max(lasso_model$lambda)
print(lasso_model) 
#绘制LASSO图
plot(lasso_model,
     xvar = "lambda")
#交叉验证并绘制可视化结果
cv_model <- cv.glmnet(x, y, family = "binomial",alpha = 1,nfolds = 10)
plot(cv_model)
#根据交叉验证结果，选择lambda值，lambda.min或lambda.1se。
lambda_min <- cv_model$lambda.min
lambda_min
lambda_1se <- cv_model$lambda.1se
lambda_1se
#s为Lambda大小，Lambda越大表示模型的正则化强度越大，选择的自变量也越少。
#这里选择的是刚刚得到的lambda_1se的值
coef_lasso <- coef(lasso_model,
                   s =  0.2573323)
coef_lasso
#结果显示后边带有数值的变量为筛选得到的变量
library("MALDIquant")
library("MALDIquantForeign")
s=importTxt("D:/yjs/文章/AC/20250720-wenzhangbuchong/联合对峰")
length(s) #检查数据对象的长度
s[1:2] #查看前两个数据对象
any(sapply(s,isEmpty)) #检查是否有空数据
table(sapply(s,length)) #统计不同长度的数据对象数量，例如m/z值有35087个的数量有40个，m/z值有35091个20个
all(sapply(s,isRegular)) #检查数据点间质量差是否相等或单调递增
plot(s[[11]]) #绘制第一个数据对象的谱图
plot(s[[2]])
m=transformIntensity(s,method="sqrt") #强度转换
v=smoothIntensity(m,method="SavitzkyGolay",halfWindowSize=50) #平滑处理（较小的窗口会导致平滑不足，保留更多的细节，但噪声会更多；较大的窗口谱图更平滑，但会模糊重要的特征）
baseline=estimateBaseline(v[[11]],method="SNIP",iterations=100) #基线估计，interaction=100指定了算法迭代的次数，使算法在早期迭代中捕捉较高的噪声水平，在后期迭代中捕捉较低的噪声水平
plot(v[[11]]) #绘制平滑后的谱图
lines(baseline,col="red",lwd=2) #在谱图上绘制基线，lwd指线条宽度
mixed=removeBaseline(v,method="SNIP",iterations=100)  # 去除基线
plot(mixed[[11]]) # 绘制去基线后的谱图
noise=estimateNoise(mixed[[8]]) # 估计噪声
plot(mixed[[11]],xlim=c(2000,20000),ylim=c(0,15)) # 绘制去基线且估计噪声的谱图，规定x和y轴显示的范围
lines(noise,col="red") # 在谱图上绘制噪声
lines(noise[,1],noise[,2]*3,col="blue") # 绘制信噪比为3的线
lines(noise[,1],noise[,2]*6,col="green") # 绘制信噪比为6的线

peaks=detectPeaks(mixed,method="MAD",halfWindowSize=50,SNR=3) #检测峰值，MAD中位数绝对偏差，hWS定义在检测峰值是考虑的窗口大小的一半，在每个数据点的左右各考虑 10 个数据点来计算局部统计量。SNR信噪比，只有当数据点的强度至少是其周围噪声水平的 3 倍时，该数据点才会被检测为一个峰。
plot(mixed[[11]],xlim=c(2000,20000),ylim=c(0,15)) #绘制谱图
points(peaks[[11]],col="red",pch=4) # 在谱图上标记峰值
peaks1=determineWarpingFunctions(peaks,tolerance=0.05,method="lowess",plot=TRUE,plotInteractive=TRUE)# 确定质量校正函数，tolerance决定了峰值匹配的精确程度，较小的容忍度值意味着更精确的匹配
warpedpeaks=warpMassPeaks(peaks,peaks1) # 应用质量校正
peaks2=binPeaks(warpedpeaks, method="strict",tolerance=0.005) # 峰值合并
Matrix=intensityMatrix(peaks2,mixed)  #构建特征矩阵
samples = sub('D:/yjs/文章/AC/20250720-wenzhangbuchong/联合对峰', '', sapply(mixed, function(x) metaData(x)$file), fixed=T) # 提取样本路径
samples = sub('\\.txt', '', samples) # 移除文件名中的 ".txt"
rownames(Matrix) = samples # 将样本路径设置为特征矩阵的行名
dir.create('D:/yjs/文章/AC/20250720-wenzhangbuchong/联合对峰/result', recursive = TRUE)
write.csv(Matrix,file="D:/yjs/文章/AC/20250720-wenzhangbuchong/联合对峰/result/result-all.csv")


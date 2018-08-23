################################################################################
##                                                                            ##
##                                  Setup                                     ##
##                                                                            ##
################################################################################

library(parallel)
library(foreach)
library(doSNOW)
library(caret)
library(RSNNS)
library(neuralnet)

set.seed(42) 
options(width = 70, digits = 2)


################################################################################
##                                                                            ##
##               Building Neural Networks - Digit Recognition                 ##
##                                                                            ##
################################################################################

dataDirectory <- "../data"
if (!file.exists(paste(dataDirectory,'/train.csv',sep="")))
{
  link <- 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/mnist_csv.zip'
  if (!file.exists(paste(dataDirectory,'/mnist_csv.zip',sep="")))
    download.file(link, destfile = paste(dataDirectory,'/mnist_csv.zip',sep=""))
  unzip(paste(dataDirectory,'/mnist_csv.zip',sep=""), exdir = dataDirectory)
  if (file.exists(paste(dataDirectory,'/test.csv',sep="")))
    file.remove(paste(dataDirectory,'/test.csv',sep=""))
}

digits <- read.csv("../data/train.csv")
dim(digits)
head(colnames(digits), 4)
tail(colnames(digits), 4)
head(digits[, 1:4])

## use nnet package to visualise a neural network
digits_nn <- digits[(digits$label==5) | (digits$label==6),]
digits_nn$y <- digits_nn$label
digits_nn$label <- NULL
table(digits_nn$y)
digits_nn$y <- ifelse(digits_nn$y==5, 0, 1)
table(digits_nn$y)

set.seed(42)
sample <- sample(nrow(digits_nn), nrow(digits_nn)*0.8)
test <- setdiff(seq_len(nrow(digits_nn)), sample)

digits.X2 <- digits_nn[,apply(digits_nn[sample,1:(ncol(digits_nn)-1)], 2, var, na.rm=TRUE) != 0]
length(digits.X2)

df.pca <- prcomp(digits.X2[sample,],center = TRUE,scale. = TRUE) 
s<-summary(df.pca)
cumprop<-s$importance[3, ]
plot(cumprop, type = "l",main="Cumulative sum",xlab="PCA component")

# keep pca components that account for 50% variance in the data
num_cols <- min(which(cumprop>0.5))
cumprop[num_cols]
newdat<-data.frame(df.pca$x[,1:num_cols])
newdat$y<-digits_nn[sample,"y"]

col_names <- names(newdat)
f <- as.formula(paste("y ~", paste(col_names[!col_names %in% "y"],collapse="+")))
nn <- neuralnet(f,data=newdat,hidden=c(4,2),linear.output = FALSE)
plot(nn)

test.data <- predict(df.pca, newdata=digits_nn[test,colnames(digits.X2)])
test.data <- as.data.frame(test.data)
preds <- compute(nn,test.data[,1:num_cols])
preds <- ifelse(preds$net.result > 0.5, "1", "0")
t<-table(digits_nn[test,"y"], preds,dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/sum(t),2)
print(t)
print(sprintf(" accuracy = %1.2f%%",acc))


# use nnet / caret
sample <- sample(nrow(digits), 6000)
train <- sample[1:5000]
test <- sample[5001:6000]

digits.X <- digits[train, -1]
digits.y_n <- digits[train, 1]
digits$label <- factor(digits$label, levels = 0:9)
digits.y <- digits[train, 1]

digits.test.X <- digits[test, -1]
digits.test.y <- digits[test, 1]
rm(sample,train,test)

barplot(table(digits.y),main="Distribution of y values (train)")

set.seed(42)
tic <- proc.time()
digits.m1 <- caret::train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(5),
             .decay = 0.1),
           trControl = trainControl(method = "none"),
           MaxNWts = 10000,
           maxit = 100)
print(proc.time() - tic)

digits.yhat1 <- predict(digits.m1,newdata=digits.test.X)
accuracy <- 100.0*sum(digits.yhat1==digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%",accuracy))
barplot(table(digits.yhat1),main="Distribution of y values (model 1)")
caret::confusionMatrix(xtabs(~digits.yhat1 + digits.test.y))

set.seed(42)
tic <- proc.time()
digits.m2 <- caret::train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(10),
             .decay = 0.1),
           trControl = trainControl(method = "none"),
            MaxNWts = 50000,
            maxit = 100)
print(proc.time() - tic)

digits.yhat2 <- predict(digits.m2,newdata=digits.test.X)
accuracy <- 100.0*sum(digits.yhat2==digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%",accuracy))
barplot(table(digits.yhat2),main="Distribution of y values (model 2)")
caret::confusionMatrix(xtabs(~digits.yhat2 + digits.test.y))

set.seed(42)
tic <- proc.time()
digits.m3 <- caret::train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(40),
             .decay = 0.1),
           trControl = trainControl(method = "none"),
           MaxNWts = 50000,
           maxit = 100)
print(proc.time() - tic)

digits.yhat3 <- predict(digits.m3,newdata=digits.test.X)
accuracy <- 100.0*sum(digits.yhat3==digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%",accuracy))
barplot(table(digits.yhat3),main="Distribution of y values (model 3)")
caret::confusionMatrix(xtabs(~digits.yhat3 + digits.test.y))

## using RSNNS package

head(decodeClassLabels(digits.y))

set.seed(42) 
tic <- proc.time()
digits.m4 <- mlp(as.matrix(digits.X),
             decodeClassLabels(digits.y),
             size = 40,
             learnFunc = "Rprop",
             shufflePatterns = FALSE,
             maxit = 80)
print(proc.time() - tic)

digits.yhat4 <- predict(digits.m4,newdata=digits.test.X)
digits.yhat4 <- encodeClassLabels(digits.yhat4)
accuracy <- 100.0*sum(I(digits.yhat4 - 1)==digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%",accuracy))
barplot(table(digits.yhat4),main="Distribution of y values (model 4)")
caret::confusionMatrix(xtabs(~ I(digits.yhat4 - 1) + digits.test.y))


################################################################################
##                                                                            ##
##                        Predictions - Digit Recognition                     ##
##                                                                            ##
################################################################################

digits.yhat4_b <- predict(digits.m4,newdata=digits.test.X)
head(round(digits.yhat4_b, 2))

table(encodeClassLabels(digits.yhat4_b,method = "WTA", l = 0, h = 0))

table(encodeClassLabels(digits.yhat4_b,method = "WTA", l = 0, h = .5))

table(encodeClassLabels(digits.yhat4_b,method = "WTA", l = .2, h = .5))

table(encodeClassLabels(digits.yhat4_b,method = "402040", l = .4, h = .6))

################################################################################
##                                                                            ##
##                        Over Fitting - Digit Recognition                    ##
##                                                                            ##
################################################################################

digits.yhat4.train <- predict(digits.m4)
digits.yhat4.train <- encodeClassLabels(digits.yhat4.train)
accuracy <- 100.0*sum(I(digits.yhat4.train - 1)==digits.y)/length(digits.y)
print(sprintf(" accuracy = %1.2f%%",accuracy))

digits.yhat1.train <- predict(digits.m1)
digits.yhat2.train <- predict(digits.m2)
digits.yhat3.train <- predict(digits.m3)
digits.yhat4.train <- predict(digits.m4)
digits.yhat4.train <- encodeClassLabels(digits.yhat4.train)

measures <- c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")

n5.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat1.train))
n5.outsample <- caret::confusionMatrix(xtabs(~digits.test.y + digits.yhat1))

n10.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat2.train))
n10.outsample <- caret::confusionMatrix(xtabs(~digits.test.y + digits.yhat2))

n40.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat3.train))
n40.outsample <- caret::confusionMatrix(xtabs(~digits.test.y + digits.yhat3))

n40b.insample <- caret::confusionMatrix(xtabs(~digits.y + I(digits.yhat4.train - 1)))
n40b.outsample <- caret::confusionMatrix(xtabs(~ digits.test.y + I(digits.yhat4 - 1)))

shrinkage <- rbind(
  cbind(Size = 5, Sample = "In", as.data.frame(t(n5.insample$overall[measures]))),
  cbind(Size = 5, Sample = "Out", as.data.frame(t(n5.outsample$overall[measures]))),
  cbind(Size = 10, Sample = "In", as.data.frame(t(n10.insample$overall[measures]))),
  cbind(Size = 10, Sample = "Out", as.data.frame(t(n10.outsample$overall[measures]))),
  cbind(Size = 40, Sample = "In", as.data.frame(t(n40.insample$overall[measures]))),
  cbind(Size = 40, Sample = "Out", as.data.frame(t(n40.outsample$overall[measures]))),
  cbind(Size = 40, Sample = "In", as.data.frame(t(n40b.insample$overall[measures]))),
  cbind(Size = 40, Sample = "Out", as.data.frame(t(n40b.outsample$overall[measures])))
  )
shrinkage$Pkg <- rep(c("nnet", "RSNNS"), c(6, 2))
dodge <- position_dodge(width=0.4)

ggplot(shrinkage, aes(interaction(Size, Pkg, sep = " : "), Accuracy,
                      ymin = AccuracyLower, ymax = AccuracyUpper,
                      shape = Sample, linetype = Sample)) +
  geom_point(size = 2.5, position = dodge) +
  geom_errorbar(width = .25, position = dodge) +
  xlab("") + ylab("Accuracy + 95% CI") +
  theme_classic() +
  theme(legend.key.size = unit(1, "cm"), legend.position = c(.8, .2))

################################################################################
##                                                                            ##
##                                   Use Case                                 ##
##                                                                            ##
################################################################################


## http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

## Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

use.train.x <- read.table("../data/UCI HAR Dataset/train/X_train.txt")
use.train.y <- read.table("../data/UCI HAR Dataset/train/y_train.txt")[[1]]

use.test.x <- read.table("../data/UCI HAR Dataset/test/X_test.txt")
use.test.y <- read.table("../data/UCI HAR Dataset/test/y_test.txt")[[1]]

use.labels <- read.table("../data/UCI HAR Dataset/activity_labels.txt")

barplot(table(use.train.y),main="Distribution of y values (UCI HAR Dataset)")


## choose tuning parameters
tuning <- list(
  size = c(40, 20, 20, 50, 50),
  maxit = c(60, 100, 100, 100, 100),
  shuffle = c(FALSE, FALSE, TRUE, FALSE, FALSE),
  params = list(FALSE, FALSE, FALSE, FALSE, c(0.1, 20, 3)))

## setup cluster using 5 cores
## load packages, export required data and variables
## and register as a backend for use with the foreach package
cl <- makeCluster(5)
clusterEvalQ(cl, {source("cluster_inc.R")})
clusterExport(cl,
  c("tuning", "use.train.x", "use.train.y",
    "use.test.x", "use.test.y")
  )
registerDoSNOW(cl)

## train models in parallel
use.models <- foreach(i = 1:5, .combine = 'c') %dopar% {
  if (tuning$params[[i]][1]) {
    set.seed(42) 
    list(Model = mlp(
      as.matrix(use.train.x),
      decodeClassLabels(use.train.y),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      learnFuncParams = tuning$params[[i]],
      maxit = tuning$maxit[[i]]
      ))
  } else {
    set.seed(42) 
    list(Model = mlp(
      as.matrix(use.train.x),
      decodeClassLabels(use.train.y),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      maxit = tuning$maxit[[i]]
    ))
  }
}

## export models and calculate both in sample,
## 'fitted' and out of sample 'predicted' values
clusterExport(cl, "use.models")
use.yhat <- foreach(i = 1:5, .combine = 'c') %dopar% {
  list(list(
    Insample = encodeClassLabels(fitted.values(use.models[[i]])),
    Outsample = encodeClassLabels(predict(use.models[[i]],
                                          newdata = as.matrix(use.test.x)))
    ))
}

use.insample <- cbind(Y = use.train.y,
  do.call(cbind.data.frame, lapply(use.yhat, `[[`, "Insample")))
colnames(use.insample) <- c("Y", paste0("Yhat", 1:5))

performance.insample <- do.call(rbind, lapply(1:5, function(i) {
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
  use.dat <- use.insample[use.insample[,paste0("Yhat", i)] != 0, ]
  use.dat$Y <- factor(use.dat$Y, levels = 1:6)
  use.dat[, paste0("Yhat", i)] <- factor(use.dat[, paste0("Yhat", i)], levels = 1:6)
  res <- caret::confusionMatrix(xtabs(f, data = use.dat))

  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))

use.outsample <- cbind(Y = use.test.y,
  do.call(cbind.data.frame, lapply(use.yhat, `[[`, "Outsample")))
colnames(use.outsample) <- c("Y", paste0("Yhat", 1:5))
performance.outsample <- do.call(rbind, lapply(1:5, function(i) {
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
  use.dat <- use.outsample[use.outsample[,paste0("Yhat", i)] != 0, ]
  use.dat$Y <- factor(use.dat$Y, levels = 1:6)
  use.dat[, paste0("Yhat", i)] <- factor(use.dat[, paste0("Yhat", i)], levels = 1:6)
  res <- caret::confusionMatrix(xtabs(f, data = use.dat))

  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))


options(width = 80, digits = 3)
performance.insample[,-4]

performance.outsample[,-4]


require(mxnet)
library(ggplot2)
library(reshape2)

dfFMnist <- read.csv("../data/fashion-mnist_train.csv", header=TRUE)
yvars <- dfFMnist$label
dfFMnist$label <- NULL

set.seed(42)
train <- sample(nrow(dfFMnist),0.9*nrow(dfFMnist))
test <- setdiff(seq_len(nrow(dfFMnist)),train)
train.y <- yvars[train]
test.y <- yvars[test]
train <- data.matrix(dfFMnist[train,])
test <- data.matrix(dfFMnist[test,])

rm(dfFMnist,yvars)

train <- t(train / 255.0)
test <- t(test / 255.0)

train.array <- train
dim(train.array) <- c(28, 28, 1, ncol(train))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

rm(train,test)
table(train.y)

act_type1="relu"
devices <- mx.cpu()
mx.set.seed(0)

data <- mx.symbol.Variable('data')
# first convolution layer
convolution1 <- mx.symbol.Convolution(data=data, kernel=c(5,5),
                                      stride=c(1,1), pad=c(2,2), num_filter=64)
activation1  <- mx.symbol.Activation(data=convolution1, act_type=act_type1)
pool1        <- mx.symbol.Pooling(data=activation1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
 
# second convolution layer
convolution2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5),
                                      stride=c(1,1), pad=c(2,2), num_filter=32)
activation2  <- mx.symbol.Activation(data=convolution2, act_type=act_type1)
pool2        <- mx.symbol.Pooling(data=activation2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
 
# flatten layer and then fully connected layers with activation and dropout
flatten <- mx.symbol.Flatten(data=pool2)
fullconnect1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=512)
activation3 <- mx.symbol.Activation(data=fullconnect1, act_type=act_type1)
drop1 <- mx.symbol.Dropout(data=activation3,p=0.4)
fullconnect2 <- mx.symbol.FullyConnected(data=drop1, num_hidden=10)
# final softmax layer
softmax <- mx.symbol.SoftmaxOutput(data=fullconnect2)

logger <- mx.metric.logger$new()
model2 <- mx.model.FeedForward.create(softmax, X=train.array, y=train.y,
                                     ctx=devices, num.round=20,
                                     array.batch.size=64,
                                     learning.rate=0.05, momentum=0.9,
                                     wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     eval.data=list(data=test.array,labels=test.y),
                                     epoch.end.callback=mx.callback.log.train.metric(100,logger))

# evaluate model
preds2 <- predict(model2, test.array)
pred.label2 <- max.col(t(preds2)) - 1
res2 <- data.frame(cbind(test.y,pred.label2))
table(res2)
accuracy2 <- sum(res2$test.y == res2$pred.label2) / nrow(res2)
accuracy2

# use the log data collected during model training
dfLogger<-as.data.frame(round(logger$train,3))
dfLogger2<-as.data.frame(round(logger$eval,3))
dfLogger$eval<-dfLogger2[,1]
colnames(dfLogger)<-c("train","eval")
dfLogger$epoch<-as.numeric(row.names(dfLogger))

data_long <- melt(dfLogger, id="epoch") 

ggplot(data=data_long,
       aes(x=epoch, y=value, colour=variable,label=value)) +
  ggtitle("Model Accuracy") +
  ylab("accuracy") +
  geom_line()+geom_point() +
  geom_text(aes(label=value),size=3,hjust=0, vjust=1) +
  theme(legend.title=element_blank()) +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(limits= 1:nrow(dfLogger))


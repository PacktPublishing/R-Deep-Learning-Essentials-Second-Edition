#
# note: the file '../dunnhumby/predict.csv' should exist,
# if it does not, then create it using prepare_data.R
#

library(glmnet)
library(readr)

set.seed(42)
fileName <- "../dunnhumby/predict.csv"
dfData <- read_csv(fileName,
                   col_types = cols(
                     .default = col_double(),
                     CUST_CODE = col_character(),
                     Y_categ = col_integer())
)
nobs <- nrow(dfData)
train <- sample(nobs, 0.9*nobs)
test <- setdiff(seq_len(nobs), train)
predictorCols <- colnames(dfData)[!(colnames(dfData) %in% c("CUST_CODE","Y_numeric","Y_numeric"))]

dfData[, c("Y_numeric",predictorCols)] <- log(0.01+dfData[, c("Y_numeric",predictorCols)])
trainData <- dfData[train, c(predictorCols,"Y_numeric")]
testData <- dfData[test, c(predictorCols,"Y_numeric")]

xtrain <- model.matrix(Y_numeric~.,trainData)
xtest <- model.matrix(Y_numeric~.,testData)

require(mxnet)

# MXNet expects matrices
train_X <- data.matrix(trainData[, predictorCols])
test_X <- data.matrix(testData[, predictorCols])
train_Y <- trainData$Y_numeric

set.seed(42)
# hyper-parameters
num_hidden <- c(256,128,128,64)
drop_out <- c(0.4,0.4,0.4,0.4)
wd=0.00001
lr <- 0.0002
num_epochs <- 100
activ <- "tanh"

# create our model architecture
# using the hyper-parameters defined above
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=num_hidden[1])
act1 <- mx.symbol.Activation(fc1, name="activ1", act_type=activ)
drop1 <- mx.symbol.Dropout(data=act1,p=drop_out[1])

fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=num_hidden[2])
act2 <- mx.symbol.Activation(fc2, name="activ2", act_type=activ)
drop2 <- mx.symbol.Dropout(data=act2,p=drop_out[2])

fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=num_hidden[3])
act3 <- mx.symbol.Activation(fc3, name="activ3", act_type=activ)
drop3 <- mx.symbol.Dropout(data=act3,p=drop_out[3])

fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=num_hidden[4])
act4 <- mx.symbol.Activation(fc4, name="activ4", act_type=activ)
drop4 <- mx.symbol.Dropout(data=act4,p=drop_out[4])

fc5 <- mx.symbol.FullyConnected(drop4, name="fc5", num_hidden=1)
lro <- mx.symbol.LinearRegressionOutput(fc5)

# run on cpu, change to 'devices <- mx.gpu()'
#  if you have a suitable GPU card
devices <- mx.cpu()
mx.set.seed(0)
tic <- proc.time()
# This actually trains the model
model <- mx.model.FeedForward.create(lro, X = train_X, y = train_Y,
                                     ctx = devices,num.round = num_epochs,
                                     learning.rate = lr, momentum = 0.9,
                                     eval.metric = mx.metric.rmse,
                                     initializer = mx.init.uniform(0.1),
                                     wd=wd,
                                     epoch.end.callback = mx.callback.log.train.metric(1))
print(proc.time() - tic)

pr4 <- predict(model, test_X)[1,]
rmse <- sqrt(mean((exp(pr4)-exp(testData[,"Y_numeric"]$Y_numeric))^2))
print(sprintf(" Deep Learning Regression RMSE = %1.2f",rmse))
mae <- mean(abs(exp(pr4)-exp(testData[,"Y_numeric"]$Y_numeric)))
print(sprintf(" Deep Learning Regression MAE = %1.2f",mae))
rm(data,fc1,act1,fc2,act2,fc3,act3,fc4,lro,model)


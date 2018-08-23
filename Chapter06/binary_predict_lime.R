#
# note: the file '../dunnhumby/predict.csv' should exist,
# if it does not, then create it using chapter4/prepare_data.R
#

library(readr)
library(ggplot2)
library(lime)
require(mxnet)

set.seed(42)
fileName <- "../dunnhumby/predict.csv"
dfData <- read_csv(fileName,
                   col_types = cols(
                     .default = col_double(),
                     CUST_CODE = col_character(),
                     Y_categ = col_integer())
                   )
# add feature (bad_var) that is highly correlated to the variable to be predicted
dfData$bad_var <- 0
dfData[dfData$Y_categ==1,]$bad_var <- 1
dfData[sample(nrow(dfData), 0.02*nrow(dfData)),]$bad_var <- 0
dfData[sample(nrow(dfData), 0.02*nrow(dfData)),]$bad_var <- 1
table(dfData$Y_categ,dfData$bad_var)
cor(dfData$Y_categ,dfData$bad_var)

nobs <- nrow(dfData)
train <-    sample(nobs, 0.8*nobs)
validate <- sample(setdiff(seq_len(nobs), train), 0.1*nobs)
test <-     setdiff(setdiff(seq_len(nobs), train),validate)
predictorCols <- colnames(dfData)[!(colnames(dfData) %in% c("CUST_CODE","Y_numeric","Y_categ"))]

# remove columns with zero variance in train-set
predictorCols <- predictorCols[apply(dfData[train, predictorCols], 2, var, na.rm=TRUE) != 0]

# for our test data, set the bad_var to zero
# our test data-set is not from the same distribution
#   as the data used to train and evaluate the model
dfData[test,]$bad_var <- 0

# look at all our predictor variables and 
#   see how they correlate with the y variable
corr <- as.data.frame(cor(dfData[,c(predictorCols,"Y_categ")]))
corr <- corr[order(-corr$Y_categ),]
old.par <- par(mar=c(7,4,3,1))

barplot(corr[2:11,]$Y_categ,names.arg=row.names(corr)[2:11],
        main="Feature Correlation to target variable",cex.names=0.8,las=2)
par(old.par)

############## deep learning model ##############

# hyper-parameters
num_hidden <- c(128,64,32)
drop_out <- c(0.2,0.2,0.2)
wd=0.00001
lr <- 0.03
num_epochs <- 40
activ <- "relu"

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
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="sm")

# run on cpu, change to 'devices <- mx.gpu()'
#  if you have a suitable GPU card
devices <- mx.cpu()
mx.set.seed(0)
tic <- proc.time()
# This actually trains the model
model <- mx.model.FeedForward.create(softmax, X = data.matrix(dfData[train, predictorCols]),
                                     y = dfData[train, "Y_categ"]$Y_categ,
                                     ctx = devices,num.round = num_epochs,
                                     learning.rate = lr, momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     initializer = mx.init.uniform(0.1),
                                     wd=wd,
                                     epoch.end.callback = mx.callback.log.train.metric(1))
print(proc.time() - tic)

pr <- predict(model, data.matrix(dfData[validate, predictorCols]))
pred.label <- max.col(t(pr)) - 1
t <- table(data.frame(cbind(dfData[validate,"Y_categ"]$Y_categ,pred.label)),
           dnn=c("Actual", "Predicted"))
acc_v<-round(100.0*sum(diag(t))/length(validate),2)
#print(t)

pr <- predict(model, data.matrix(dfData[test, predictorCols]))
pred.label <- max.col(t(pr)) - 1
t <- table(data.frame(cbind(dfData[test,"Y_categ"]$Y_categ,pred.label)),
           dnn=c("Actual", "Predicted"))
acc_t<-round(100.0*sum(diag(t))/length(test),2)
#print(t)

rm(data,fc1,act1,fc2,act2,fc3,act3,fc4,softmax)

#### Verifying the model using LIME

# compare performance on validation and test set 
print(sprintf(" Deep Learning Model accuracy on validate (expected in production) = %1.2f%%",acc_v))
print(sprintf(" Deep Learning Model accuracy in (actual in production) = %1.2f%%",acc_t))

# apply LIME to MXNet deep learning model
model_type.MXFeedForwardModel <- function(x, ...) {return("classification")}
predict_model.MXFeedForwardModel <- function(m, newdata, ...)
{
  pred <- predict(m, as.matrix(newdata),array.layout="rowmajor")
  pred <- as.data.frame(t(pred))
  colnames(pred) <- c("No","Yes")
  return(pred)
}
#explain <- lime(train_X, model, bin_continuous = TRUE, n_bins = 5, n_permutations = 1000)
explain <- lime(dfData[train, predictorCols], model, bin_continuous = FALSE)
val_first_10 <- validate[1:10]

explaination <- lime::explain(dfData[val_first_10, predictorCols],explainer=explain,
                              n_labels=1,n_features=3)
plot_features(explaination) + labs(title="Churn Model - variable explanation")

plot_explanations (explaination) +
  labs (title = "LIME Feature Importance Heatmap (Test-set)")


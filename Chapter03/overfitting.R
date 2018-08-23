################################################################################
##                                                                            ##
##                                  Setup                                     ##
##                                                                            ##
################################################################################

library(glmnet)
library(MASS)
library(deepnet)
library(RSNNS)
library(parallel)
library(foreach)
library(doSNOW)
options(width = 70, digits = 2)

################################################################################
##                                                                            ##
##                                L1 Penalty                                  ##
##                                                                            ##
################################################################################

## set seed and simulate data
set.seed(1234)
X <- mvrnorm(n = 200, mu = c(0, 0, 0, 0, 0),
  Sigma = matrix(c(
    1, .9999, .99, .99, .10,
    .9999, 1, .99, .99, .10,
    .99, .99, 1, .99, .10,
    .99, .99, .99, 1, .10,
    .10, .10, .10, .10, 1
  ), ncol = 5))
y <- rnorm(200, 3 + X %*% matrix(c(1, 1, 1, 1, 0)), .5)

m.ols <- lm(y[1:100] ~ X[1:100, ])
m.lasso.cv <- cv.glmnet(X[1:100, ], y[1:100], alpha = 1)
plot(m.lasso.cv)

cbind(OLS = coef(m.ols),Lasso = coef(m.lasso.cv)[,1])

################################################################################
##                                                                            ##
##                                L2 Penalty                                  ##
##                                                                            ##
################################################################################

m.ridge.cv <- cv.glmnet(X[1:100, ], y[1:100], alpha = 0)
plot(m.ridge.cv)

cbind(OLS = coef(m.ols),Lasso = coef(m.lasso.cv)[,1],Ridge = coef(m.ridge.cv)[,1])


################################################################################
##                                                                            ##
##                                 Use Case                                   ##
##                                                                            ##
################################################################################

set.seed(42)
if (!file.exists('../data/train.csv'))
{
  link <- 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/mnist_csv.zip'
  if (!file.exists(paste(dataDirectory,'/mnist_csv.zip',sep="")))
    download.file(link, destfile = paste(dataDirectory,'/mnist_csv.zip',sep=""))
  unzip(paste(dataDirectory,'/mnist_csv.zip',sep=""), exdir = dataDirectory)
  if (file.exists(paste(dataDirectory,'/test.csv',sep="")))
    file.remove(paste(dataDirectory,'/test.csv',sep=""))
}

## same data as from previous chapter
digits.train <- read.csv("../data/train.csv")

## convert to factor
digits.train$label <- factor(digits.train$label, levels = 0:9)

sample <- sample(nrow(digits.train), 6000)
train <- sample[1:5000]
test <- sample[5001:6000]

digits.X <- digits.train[train, -1]
digits.y <- digits.train[train, 1]
test.X <- digits.train[test, -1]
test.y <- digits.train[test, 1]

## try various weight decays and number of iterations
## register backend so that different decays can be
## estimated in parallel
cl <- makeCluster(parallel::detectCores() -1)
clusterEvalQ(cl, {source("cluster_inc.R")})
registerDoSNOW(cl)

set.seed(1234)
digits.decay.m1 <- lapply(c(100, 150), function(its) {
  caret::train(digits.X, digits.y,
           method = "nnet",
           tuneGrid = expand.grid(
             .size = c(10),
             .decay = c(0, .1)),
           trControl = caret::trainControl(method="cv", number=5, repeats=1),
           MaxNWts = 10000,
           maxit = its)
})

digits.decay.m1[[1]]
digits.decay.m1[[2]]


################################################################################
##                                                                            ##
##                                Model Averaging                             ##
##                                                                            ##
################################################################################

## simulated data
set.seed(1234)
d <- data.frame(
  x = rnorm(400))
d$y <- with(d, rnorm(400, 2 + ifelse(x < 0, x + x^2, x + x^2.5), 1))
d.train <- d[1:200, ]
d.test <- d[201:400, ]

## three different models
m1 <- lm(y ~ x, data = d.train)
m2 <- lm(y ~ I(x^2), data = d.train)
m3 <- lm(y ~ pmax(x, 0) + pmin(x, 0), data = d.train)

## In sample R2
cbind(M1=summary(m1)$r.squared,
      M2=summary(m2)$r.squared,M3=summary(m3)$r.squared)

## correlations in the training data
cor(cbind(M1=fitted(m1),
          M2=fitted(m2),M3=fitted(m3)))


## generate predictions and the average prediction
d.test$yhat1 <- predict(m1, newdata = d.test)
d.test$yhat2 <- predict(m2, newdata = d.test)
d.test$yhat3 <- predict(m3, newdata = d.test)
d.test$yhatavg <- rowMeans(d.test[, paste0("yhat", 1:3)])

## correlation in the testing data
cor(d.test)

################################################################################
##                                                                            ##
##                                 Use Case                                   ##
##                                                                            ##
################################################################################

## Fit Models
nn.models <- foreach(i = 1:4, .combine = 'c') %dopar% {
set.seed(1234)
 list(nn.train(
    x = as.matrix(digits.X),
    y = model.matrix(~ 0 + digits.y),
    hidden = c(40, 80, 40, 80)[i],
    activationfun = "tanh",
    learningrate = 0.8,
    momentum = 0.5,
    numepochs = 150,
    output = "softmax",
    hidden_dropout = c(0, 0, .5, .5)[i],
    visible_dropout = c(0, 0, .2, .2)[i]))
}

nn.yhat <- lapply(nn.models, function(obj) {
  encodeClassLabels(nn.predict(obj, as.matrix(digits.X)))
})

perf.train <- do.call(cbind, lapply(nn.yhat, function(yhat) {
  caret::confusionMatrix(xtabs(~ I(yhat - 1) + digits.y))$overall
}))
colnames(perf.train) <- c("N40", "N80", "N40_Reg", "N80_Reg")

options(digits = 4)
perf.train

nn.yhat.test <- lapply(nn.models, function(obj) {
  encodeClassLabels(nn.predict(obj, as.matrix(test.X)))
})

perf.test <- do.call(cbind, lapply(nn.yhat.test, function(yhat) {
  caret::confusionMatrix(xtabs(~ I(yhat - 1) + test.y))$overall
}))
colnames(perf.test) <- c("N40", "N80", "N40_Reg", "N80_Reg")

perf.test

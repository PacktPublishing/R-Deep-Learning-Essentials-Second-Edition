library(clusterSim)
library(e1071)

getData <- function(data_sel){
  set.seed(42)
  if (data_sel =="bulls_eye")
    data<-shapes.bulls.eye(200)
  else if (data_sel =="worms")
    data<-shapes.worms(200)
  else if (data_sel =="moon")
    data<-shapes.two.moon(200)
  else
    data<-shapes.blocks3d(200)
  df <- as.data.frame(data$data)
  df$Y<-data$clusters-1
  sample<-sample(nrow(df),20)
  df[sample,]$Y <-df[sample,]$Y+1
  df[df$Y==2,]$Y <-0
  return(df)
}

activation_function <- function(activation_ftn,v)
{
  if (activation_ftn == "sigmoid")
    res <- sigmoid(v)
  else if (activation_ftn == "tanh")
    res <- tanh(v)
  else if (activation_ftn == "relu")
  {
    v[v<0] <- 0
    res <- v
  }
  else
    res <- sigmoid(v)
  return (res)
}

derivative_function <- function(activation_ftn,v)
{
  if (activation_ftn == "sigmoid")
    upd <- (v * (1 - v))
  else if (activation_ftn == "tanh")
    upd <- (1 - (v^2))
  else if (activation_ftn == "relu")
    upd <- ifelse(v > 0.0,1,0)
  else
    upd <- (v * (1 - v))
  return (upd)
}

forward_prop <- function(X,activation_ftn,weights1,bias1,weights2,bias2)
{
  # broadcast hack
  bias1a<-bias1
  for (i in 2:ncol(X))
    bias1a<-cbind(bias1a,bias1)
  
  Z1 <<- weights1 %*% X + bias1a
  activation1 <<- activation_function(activation_ftn,Z1)
  bias2a<-bias2
  for (i in 2:ncol(activation1))
    bias2a<-cbind(bias2a,bias2)
  Z2 <<- weights2 %*% activation1 + bias2a
  activation2 <<- sigmoid(Z2)
  return (activation2)
}

cost_f <- function(activation2,Y)
{
  cost = -mean((log(activation2) * Y)+ (log(1-activation2) * (1-Y)))
  return(cost)
}

backward_prop <- function(X,Y,activation_ftn,weights1,weights2,activation1,activation2)
{
  m <- ncol(Y)
  derivative2 <- activation2-Y
  dweights2 <<- (derivative2 %*% t(activation1)) / m
  dbias2 <<- rowSums(derivative2) / m
  upd <- derivative_function(activation_ftn,activation1)
  derivative1 <- t(weights2) %*% derivative2 * upd
  dweights1 <<- (derivative1 %*% t(X)) / m
  dbias1 <<- rowSums(derivative1) / m
}


source("nnet_functions.R")

data_sel <- "bulls_eye"
#data_sel <- "worms"
#data_sel <- "moon"
#data_sel <- "blocks"

df <- getData(data_sel)

model <- glm(Y ~.,family=binomial(link='logit'),data=df)
res <- predict(model,newdata=df,type='response')
res <- ifelse(res > 0.5,1,0)
df$Y2<-res
logreg_accuracy <- sum(df$Y==df$Y2) / nrow(df)

plot(df[,1:2],col=rainbow(2)[(1+df$Y)],main=paste("Logistic regression accuracy on",data_sel,"=",logreg_accuracy))

####################### neural network ######################
hidden <- 3
epochs <- 3000
lr <- 0.5
activation_ftn <- "sigmoid"
#activation_ftn <- "tanh"
#activation_ftn <- "relu"

df <- getData(data_sel) # from nnet_functions
X <- as.matrix(df[,1:2])
Y <- as.matrix(df$Y)
n_x=ncol(X)
n_h=hidden
n_y=1
m <- nrow(X)

# initialise weights
set.seed(42)
weights1 <- matrix(0.01*runif(n_h*n_x)-0.005, ncol=n_x, nrow=n_h)
weights2 <- matrix(0.01*runif(n_y*n_h)-0.005, ncol=n_h, nrow=n_y)
bias1 <- matrix(rep(0,n_h),nrow=n_h,ncol=1)
bias2 <- matrix(rep(0,n_y),nrow=n_y,ncol=1)

for (i in 0:epochs)
{
  activation2 <- forward_prop(t(X),activation_ftn,weights1,bias1,weights2,bias2)
  cost <- cost_f(activation2,t(Y))
  backward_prop(t(X),t(Y),activation_ftn,weights1,weights2,activation1,activation2)
  weights1 <- weights1 - (lr * dweights1)
  bias1 <- bias1 - (lr * dbias1)
  weights2 <- weights2 - (lr * dweights2)
  bias2 <- bias2 - (lr * dbias2)
  
  if ((i %% 500) == 0)
    print (paste(" Cost after",i,"epochs =",cost))
}

# predict is the same as a single forward propogation
res <- forward_prop(t(X),activation_ftn,weights1,bias1,weights2,bias2)
res <- ifelse(res > 0.5,1,0)
df$Y2<-res[1,]
nn_accuracy <- sum(df$Y==df$Y2) / nrow(df)
print(sprintf("Neural network Accuracy = %1.2f",nn_accuracy))



##### model is "undertrained", lets do a few more epochs ####
for (i in epochs:(epochs*2))
{
  activation2 <- forward_prop(t(X),activation_ftn,weights1,bias1,weights2,bias2)
  cost <- cost_f(activation2,t(Y))
  backward_prop(t(X),t(Y),activation_ftn,weights1,weights2,activation1,activation2)
  weights1 <- weights1 - (lr * dweights1)
  bias1 <- bias1 - (lr * dbias1)
  weights2 <- weights2 - (lr * dweights2)
  bias2 <- bias2 - (lr * dbias2)
  
  if ((i %% 500) == 0)
    print (paste(" Cost after",i,"epochs =",cost))
}

# predict is the same as a single forward propogation
res <- forward_prop(t(X),activation_ftn,weights1,bias1,weights2,bias2)
res <- ifelse(res > 0.5,1,0)
df$Y2<-res[1,]
nn_accuracy <- sum(df$Y==df$Y2) / nrow(df)
print(sprintf("Neural network Accuracy = %1.2f",nn_accuracy))

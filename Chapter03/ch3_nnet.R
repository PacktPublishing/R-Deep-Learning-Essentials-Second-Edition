library(clusterSim)
source("nnet_functions.R")

set.seed(42)

data<-shapes.bulls.eye(200)
#data<-shapes.worms(200)
#data<-shapes.two.moon(200)
#data<-shapes.blocks3d(200)

df <- as.data.frame(data$data)
df$Y<-data$clusters-1
X<-as.matrix(df[,1:2])
Y<-as.matrix(df$Y)
sample<-sample(nrow(df),20)
df[sample,]$Y <-df[sample,]$Y+1
df[df$Y==2,]$Y <-0
plot(df[,1:2],col=rainbow(2)[(1+df$Y)])

model <- glm(Y ~.,family=binomial(link='logit'),data=df)
res <- predict(model,newdata=df,type='response')
res <- ifelse(res > 0.5,1,0)
df$Y2<-res
logreg_accuracy <- sum(df$Y==df$Y2) / nrow(df)
logreg_accuracy

rm(data,model,res,logreg_accuracy)

n_x=ncol(X)
n_h=3
n_y=1
m <- nrow(X)

# initialise weights
weights1 <- matrix(0.01*runif(n_h*n_x), ncol=n_x, nrow=n_h)
weights2 <- matrix(0.01*runif(n_y*n_h), ncol=n_h, nrow=n_y)
bias1 <- matrix(rep(0,n_h),nrow=n_h,ncol=1)
bias2 <- matrix(rep(0,n_y),nrow=n_y,ncol=1)

lr <- 0.6
for (i in 0:1500)
{
  activation2 <- forward_prop(t(X),weights1,bias1,weights2,bias2)
  cost <- cost_f(activation2,t(Y))
  backward_prop(t(X),t(Y),weights1,weights2,activation1,activation2)
  weights1 <- weights1 - (lr * dweights1)
  bias1 <- bias1 - (lr * dbias1)
  weights2 <- weights2 - (lr * dweights2)
  bias2 <- bias2 - (lr * dbias2)
  
  if ((i %% 500) == 0)
    print (paste("Cost after",i,"epochs=",cost))
}

res <- forward_prop(t(X),weights1,bias1,weights2,bias2)
res <- ifelse(res > 0.5,1,0)
df$Y2<-res[1,]
nn_accuracy <- sum(df$Y==df$Y2) / nrow(df)
nn_accuracy

for (i in 1500:4000)
{
  activation2 <- forward_prop(t(X),weights1,bias1,weights2,bias2)
  cost <- cost_f(activation2,t(Y))
  backward_prop(t(X),t(Y),weights1,weights2,activation1,activation2)
  weights1 <- weights1 - (lr * dweights1)
  bias1 <- bias1 - (lr * dbias1)
  weights2 <- weights2 - (lr * dweights2)
  bias2 <- bias2 - (lr * dbias2)
  
  if ((i %% 500) == 0)
    print (paste("Cost after",i,"epochs=",cost))
}

res <- forward_prop(t(X),weights1,bias1,weights2,bias2)
res <- ifelse(res > 0.5,1,0)
df$Y2<-res[1,]
nn_accuracy <- sum(df$Y==df$Y2) / nrow(df)
nn_accuracy


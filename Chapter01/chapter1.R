
##########################
# straight line
##########################
par(mfrow=c(1,1))
set.seed(42)
m <- 1.4
b <- -1.2
x <- 0:9
jitter<-0.6
xline <- x
y <- m*x+b
x <- x+rnorm(10)*jitter
title <- paste("y = ",m,"x ",b,sep="")
plot(xline,y,type="l",lty=2,col="red",main=title,xlim=c(0,max(y)),ylim=c(0,max(y)))
points(x[seq(1,10,2)],y[seq(1,10,2)],pch=1)
points(x[seq(2,11,2)],y[seq(2,11,2)],pch=4)

##########################
# polynomial regression / overfitting
##########################
par(mfrow=c(1,2))
set.seed(1)
x1 <- seq(-2,2,0.5)

# y=x^2-6
jitter<-0.3
y1 <- (x1^2)-6
x1 <- x1+rnorm(length(x1))*jitter
plot(x1,y1,xlim=c(-8,12),ylim=c(-8,10),pch=1)
x <- x1
y <- y1

# y=-x
jitter<-0.8
x2 <- seq(-7,-5,0.4)
y2 <- -x2
x2 <- x2+rnorm(length(x2))*jitter
points(x2,y2,pch=2)
x <- c(x,x2)
y <- c(y,y2)

# y=0.4 *rnorm(length(x3))*jitter
jitter<-1.2
x3 <- seq(5,9,0.5)
y3 <- 0.4 *rnorm(length(x3))*jitter
points(x3,y3,pch=3)
x <- c(x,x3)
y <- c(y,y3)

df <- data.frame(cbind(x,y))
plot(x,y,xlim=c(-8,12),ylim=c(-8,10),pch=4)

model1 <- lm(y~.,data=df)
abline(coef(model1),lty=2,col="red")

max_degree<-3
for (i in 2:max_degree)
{
  col<-paste("x",i,sep="")
  df[,col] <- df$x^i
}
model2 <- lm(y~.,data=df)
xplot <- seq(-8,12,0.1)
yplot <- (xplot^0)*model2$coefficients[1]
for (i in 1:max_degree)
  yplot <- yplot +(xplot^i)*model2$coefficients[i+1]
points(xplot,yplot,col="blue", cex=0.5)

max_degree<-12
for (i in 2:max_degree)
{
  col<-paste("x",i,sep="")
  df[,col] <- df$x^i
}
model3 <- lm(y~.,data=df)
xplot <- seq(-8,12,0.1)
yplot <- (xplot^0)*model3$coefficients[1]
for (i in 1:max_degree)
  yplot <- yplot +(xplot^i)*model3$coefficients[i+1]
points(xplot,yplot,col="green", cex=0.5,pch=2)

MSE1 <- c(crossprod(model1$residuals)) / length(model1$residuals)
MSE2 <- c(crossprod(model2$residuals)) / length(model2$residuals)
MSE3 <- c(crossprod(model3$residuals)) / length(model3$residuals)
print(sprintf(" Model 1 MSE = %1.2f",MSE1))
print(sprintf(" Model 2 MSE = %1.2f",MSE2))
print(sprintf(" Model 3 MSE = %1.2f",MSE3))


#######################




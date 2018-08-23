
library(neuralnet)
library(keras)
library(corrplot)

options(width = 70, digits = 2)
options(scipen=999)

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

data <- read.csv("../data/train.csv", header=TRUE)

set.seed(42)
sample<-sample(nrow(data),0.5*nrow(data))
test <- setdiff(seq_len(nrow(data)),sample)
train.x <- data[sample,-1]
test.x <- data[test,-1]

train.y <- data[sample,1]
test.y <- data[test,1]

rm(data)
train.x <- train.x/255
test.x <- test.x/255

train.x <- data.matrix(train.x)
test.x <- data.matrix(test.x)

input_dim <- 28*28 #784

# model 1
inner_layer_dim <- 16
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='tanh')(input_layer)
decoder <- layer_dense(units=784)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam',
                        loss='mean_squared_error',metrics='accuracy')
history <- autoencoder %>% fit(train.x,train.x,
                               epochs=40, batch_size=128,validation_split=0.2)
summary(autoencoder)
plot(history)

# model 2
inner_layer_dim <- 32
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='tanh')(input_layer)
decoder <- layer_dense(units=784)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam',
                        loss='mean_squared_error',metrics='accuracy')
history <- autoencoder %>% fit(train.x,train.x,
                               epochs=40, batch_size=128,validation_split=0.2)
plot(history)

# model 3
inner_layer_dim <- 64
input_layer <- layer_input(shape=c(input_dim))
encoder <- layer_dense(units=inner_layer_dim, activation='tanh')(input_layer)
decoder <- layer_dense(units=784)(encoder)
autoencoder <- keras_model(inputs=input_layer, outputs = decoder)

autoencoder %>% compile(optimizer='adam',
                        loss='mean_squared_error',metrics='accuracy')
history <- autoencoder %>% fit(train.x,train.x,
                               epochs=40, batch_size=128,validation_split=0.2)
plot(history)

############
# run model 1 first
encoder <- keras_model(inputs=input_layer, outputs=encoder)
encodings <- encoder %>% predict(test.x)
encodings<-as.data.frame(encodings)
M <- cor(encodings)
corrplot(M, method = "circle", sig.level = 0.1)


# attach the actual y variable
encodings$y <- test.y
encodings <- encodings[encodings$y==5 | encodings$y==6,]
encodings[encodings$y==5,]$y <- 0
encodings[encodings$y==6,]$y <- 1
table(encodings$y)

nobs <- nrow(encodings)
train <- sample(nobs, 0.9*nobs)
test <- setdiff(seq_len(nobs), train)
trainData <- encodings[train,]
testData <- encodings[test,]

col_names <- names(trainData)
f <- as.formula(paste("y ~", paste(col_names[!col_names %in%
                                               "y"],collapse="+")))
nn <- neuralnet(f,data=trainData,hidden=c(4,2),linear.output = FALSE)
preds_nn <- compute(nn,testData[,1:(-1+ncol(testData))])

preds_nn <- ifelse(preds_nn$net.result > 0.5, "1", "0")
t<-table(testData$y, preds_nn,dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/sum(t),2)
print(t)
print(sprintf(" accuracy = %1.2f%%",acc))


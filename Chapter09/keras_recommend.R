library(readr)
library(keras)
set.seed(42)
use_session_with_seed(42, disable_gpu = FALSE, disable_parallel_cpu = FALSE)
df<-read_csv("../dunnhumby/recommend.csv")

custs <- as.data.frame(unique(df$cust_id))
custs$cust_id2 <- as.numeric(row.names(custs))
colnames(custs) <- c("cust_id","cust_id2")
custs$cust_id2 <- custs$cust_id2 - 1
prods <- as.data.frame(unique(df$prod_id))
prods$prod_id2 <- as.numeric(row.names(prods))
colnames(prods) <- c("prod_id","prod_id2")
prods$prod_id2 <- prods$prod_id2 - 1
df<-merge(df,custs)
df<-merge(df,prods)
n_custs = length(unique(df$cust_id2))
n_prods = length(unique(df$prod_id2))
# shuffle the data
trainData <- df[sample(nrow(df)),]

n_factors<-10
# define the model
cust_in <- layer_input(shape = 1)
cust_embed <- layer_embedding(
  input_dim = n_custs
  ,output_dim = n_factors
  ,input_length = 1
  ,embeddings_regularizer=regularizer_l2(0.0001)
  ,name = "cust_embed"
)(cust_in)
prod_in <- layer_input(shape = 1)
prod_embed <- layer_embedding(
  input_dim = n_prods
  ,output_dim = n_factors
  ,input_length = 1
  ,embeddings_regularizer=regularizer_l2(0.0001)
  ,name = "prod_embed"
)(prod_in)

ub = layer_embedding(
  input_dim = n_custs,
  output_dim = 1,
  input_length = 1,
  name = "custb_embed"
)(cust_in)
ub_flat <- layer_flatten()(ub)
mb = layer_embedding(
  input_dim = n_prods,
  output_dim = 1,
  input_length = 1,
  name = "prodb_embed"
)(prod_in)
mb_flat <- layer_flatten()(mb)
cust_flat <- layer_flatten()(cust_embed)
prod_flat <- layer_flatten()(prod_embed)
x <- layer_dot(list(cust_flat, prod_flat), axes = 1)
x <- layer_add(list(x, ub_flat))
x <- layer_add(list(x, mb_flat))

model <- keras_model(list(cust_in, prod_in), x)
compile(model,optimizer="adam", loss='mse')
model.optimizer.lr=0.001
fit(model,list(trainData$cust_id2,trainData$prod_id2),trainData$rating,
    batch_size=128,epochs=40,validation_split = 0.1 )

##### model use-case, find products that customers 'should' be purchasing
######
df$preds<-predict(model,list(df$cust_id2,df$prod_id2))
# remove index variables, do not need them anymore
df$cust_id2 <- NULL
df$prod_id2 <- NULL
mse<-mean((df$rating-df$preds)^2)
rmse<-sqrt(mse)
mae<-mean(abs(df$rating-df$preds))
print (sprintf("DL Collaborative filtering model: MSE=%1.3f, RMSE=%1.3f, MAE=%1.3f",mse,rmse,mae))
df <- df[order(-df$preds),]
head(df)

df[df$preds>5,]$preds <- 5
df[df$preds<1,]$preds <- 1
mse<-mean((df$rating-df$preds)^2)
rmse<-sqrt(mse)
mae<-mean(abs(df$rating-df$preds))
print (sprintf("DL Collaborative filtering model (adjusted): MSE=%1.3f, RMSE=%1.3f, MAE=%1.3f",mse,rmse,mae))

	
df$diff <- df$preds - df$rating
df <- df[order(-df$diff),]
head(df,20)

df <- df[order(df$diff),]
head(df,20)


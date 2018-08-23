source("img_ftns.R")
data <- read.csv("../data/train.csv", header=TRUE)

set.seed(RANDOM_SEED_VALUE)
sample<-sample(nrow(data),0.9*nrow(data))
train <- data[sample,]
test <- data[setdiff(seq_len(nrow(data)),sample),]
rm(data,sample)

aug_df <- test[1==0,]
aug_df_temp <- test[1==0,]
print(paste("0 :",Sys.time()))
for (i in 1:nrow(test))
{
  aug_df <- rbind(aug_df,test[i,])
  row <- as.numeric(test[i,2:ncol(test)])
  img_row <- matrix(row,nrow=28,byrow=TRUE)
  img_rotate<-rotateInstance (img_row,15)
  new_row <- as.vector(t(img_rotate))
  aug_df <- rbind(aug_df,append(test[i,1],new_row))
  
  img_rotate<-rotateInstance (img_row,360-15)
  new_row <- as.vector(t(img_rotate))
  aug_df <- rbind(aug_df,append(test[i,1],new_row))
  
  if ((i %% 1000) == 0)
  {
    aug_df <- rbind(aug_df,aug_df_temp)
    aug_df_temp <- test[1==0,]
    print(paste(i,":",Sys.time()))
  }
}
colnames(aug_df) <- colnames(test)
write.csv(aug_df,"../data/test_augment.csv", quote = FALSE,row.names=FALSE)
write.csv(test,"../data/test_0.csv", quote = FALSE,row.names=FALSE)

aug_df <- train[1==0,]
aug_df_temp <- train[1==0,]
print(paste("0 :",Sys.time()))
for (i in 1:nrow(train))
{
  aug_df <- rbind(aug_df,train[i,])
  row <- as.numeric(train[i,2:ncol(train)])
  img_row <- matrix(row,nrow=28,byrow=TRUE)
  img_rotate<-rotateInstance (img_row,15)
  new_row <- as.vector(t(img_rotate))
  aug_df <- rbind(aug_df,append(train[i,1],new_row))

  img_rotate<-rotateInstance (img_row,360-15)
  new_row <- as.vector(t(img_rotate))
  aug_df <- rbind(aug_df,append(train[i,1],new_row))

  if ((i %% 1000) == 0)
  {
    aug_df <- rbind(aug_df,aug_df_temp)
    aug_df_temp <- train[1==0,]
    print(paste(i,":",Sys.time()))
  }
}
colnames(aug_df) <- colnames(train)
write.csv(aug_df,"../data/train_augment.csv", quote = FALSE,row.names=FALSE)



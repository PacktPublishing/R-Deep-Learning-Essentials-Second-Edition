library(keras)

# the reuters dataset is in Keras
?dataset_reuters
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_reuters()
word_index <- dataset_reuters_word_index()

# convert the word index into a dataframe
idx<-unlist(word_index)
dfWords<-as.data.frame(idx)
dfWords$word <- row.names(dfWords)
row.names(dfWords)<-NULL
dfWords <- dfWords[order(dfWords$idx),]

# create a dataframe for the train data
# for each row in the train data, we have a list of index values
#  for words in the dfWords dataframe
dfTrain <- data.frame(y_train)
dfTrain$sentence <- ""
colnames(dfTrain)[1] <- "y"
for (r in 1:length(x_train))
{
  row <- x_train[r]
  line <- ""
  for (i in 1:length(row[[1]]))
  {
     index <- row[[1]][i]
     if (index >= 3)
       line <- paste(line,dfWords[index-3,]$word)
  }
  dfTrain[r,]$sentence <- line
  if ((r %% 100) == 0)
    print (r)
}
write.table(dfTrain,"../data/reuters.train.tab",sep="\t",row.names = FALSE)

# create a dataframe for the test data
# for each row in the train data, we have a list of index values
#  for words in the dfWords dataframe
dfTest <- data.frame(y_test)
dfTest$sentence <- ""
colnames(dfTest)[1] <- "y"
for (r in 1:length(x_test))
{
  row <- x_test[r]
  line <- ""
  for (i in 1:length(row[[1]]))
  {
    index <- row[[1]][i]
    if (index >= 3)
      line <- paste(line,dfWords[index-3,]$word)
  }
  dfTest[r,]$sentence <- line
  if ((r %% 100) == 0)
    print (r)
}
write.table(dfTest,"../data/reuters.test.tab",sep="\t",row.names = FALSE)


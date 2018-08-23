library(tm)
require(nnet)
require(kernlab)
library(randomForest)
library(e1071)
options(digits=4)

TextClassification <-function (w,stem=0,stop=0,verbose=1)
{
  df <- read.csv("../data/reuters.train.tab", sep="\t", stringsAsFactors = FALSE)
  df2 <- read.csv("../data/reuters.test.tab", sep="\t", stringsAsFactors = FALSE)
  df <- rbind(df,df2)

  # df <- df[df$y %in% c(3,4),]
  # df$y <- df$y-3
  df[df$y!=3,]$y<-0
  df[df$y==3,]$y<-1
  rm(df2)

  corpus <- Corpus(DataframeSource(data.frame(df[, 2])))
  corpus <- tm_map(corpus, content_transformer(tolower))
  
  # hyper-parameters
  if (stop==1)
    corpus <- tm_map(corpus, function(x) removeWords(x, stopwords("english")))
  if (stem==1)
    corpus <- tm_map(corpus, stemDocument)
  if (w=="tfidf")
    dtm <- DocumentTermMatrix(corpus,control=list(weighting=weightTfIdf))
  else if (w=="tf")
    dtm <- DocumentTermMatrix(corpus,control=list(weighting=weightTf))
  else if (w=="binary")
    dtm <- DocumentTermMatrix(corpus,control=list(weighting=weightBin))
  
  # keep terms that cover 95% of the data
  dtm2<-removeSparseTerms(dtm, 0.95)
  m <- as.matrix(dtm2)
  remove(dtm,dtm2,corpus)
  
  data<-data.frame(m)
  data<-cbind(df[, 1],data)
  colnames(data)[1]="y"
  
  # create train, test sets for machine learning
  seed <- 42 
  set.seed(seed) 
  nobs <- nrow(data)
  sample <- train <- sample(nrow(data), 0.8*nobs)
  validate <- NULL
  test <- setdiff(setdiff(seq_len(nrow(data)), train), validate)
  
  # create Naive Bayes model
  nb <- naiveBayes(as.factor(y) ~., data=data[sample,])
  pr <- predict(nb, newdata=data[test, ])
  # Generate the confusion matrix showing counts.
  tab<-table(na.omit(data[test, ])$y, pr,
             dnn=c("Actual", "Predicted"))
  if (verbose) print (tab)
  nb_acc <- 100*sum(diag(tab))/length(test)
  if (verbose) print(sprintf("Naive Bayes accuracy = %1.2f%%",nb_acc))
  
  # create SVM model
  if (verbose) print ("SVM")
  if (verbose) print (Sys.time())
  ksvm <- ksvm(as.factor(y) ~ .,
               data=data[sample,],
               kernel="rbfdot",
               prob.model=TRUE)
  if (verbose) print (Sys.time())
  pr <- predict(ksvm, newdata=na.omit(data[test, ]))
  # Generate the confusion matrix showing counts.
  tab<-table(na.omit(data[test, ])$y, pr,
             dnn=c("Actual", "Predicted"))
  if (verbose) print (tab)
  svm_acc <- 100*sum(diag(tab))/length(test)
  if (verbose) print(sprintf("SVM accuracy = %1.2f%%",svm_acc))
  
  # create Neural Network model
  rm(pr,tab)
  set.seed(199)
  if (verbose) print ("Neural Network")
  if (verbose) print (Sys.time())
  nnet <- nnet(as.factor(y) ~ .,
               data=data[sample,],
               size=10, skip=TRUE, MaxNWts=10000, trace=FALSE, maxit=100)
  if (verbose) print (Sys.time())
  pr <- predict(nnet, newdata=data[test, ], type="class")
  # Generate the confusion matrix showing counts.
  tab<-table(data[test, ]$y, pr,
             dnn=c("Actual", "Predicted"))
  if (verbose) print (tab)
  nn_acc <- 100*sum(diag(tab))/length(test)
  if (verbose) print(sprintf("Neural Network accuracy = %1.2f%%",nn_acc))
  
  # create Random Forest model
  rm(pr,tab)
  if (verbose) print ("Random Forest")
  if (verbose) print (Sys.time())
  rf_model<-randomForest(as.factor(y) ~., data=data[sample,])
  if (verbose) print (Sys.time())
  pr <- predict(rf_model, newdata=data[test, ], type="class")
  # Generate the confusion matrix showing counts.
  tab<-table(data[test, ]$y, pr,
             dnn=c("Actual", "Predicted"))
  if (verbose) print (tab)
  rf_acc <- 100*sum(diag(tab))/length(test)
  if (verbose) print(sprintf("Random Forest accuracy = %1.2f%%",rf_acc))
  
  dfParams <- data.frame(w,stem,stop)
  dfParams$nb_acc <- nb_acc
  dfParams$svm_acc <- svm_acc
  dfParams$nn_acc <- nn_acc
  dfParams$rf_acc <- rf_acc
  
  return(dfParams)
}

######################################
dfResults <- TextClassification("tfidf",verbose=1)                     # tf-idf, no stemming
dfResults<-rbind(dfResults,TextClassification("tf",verbose=1))         # tf, no stemming
dfResults<-rbind(dfResults,TextClassification("binary",verbose=1))     # binary, no stemming

dfResults<-rbind(dfResults,TextClassification("tfidf",1,verbose=1))    # tf-idf, stemming
dfResults<-rbind(dfResults,TextClassification("tf",1,verbose=1))       # tf, stemming
dfResults<-rbind(dfResults,TextClassification("binary",1,verbose=1))   # binary, stemming

dfResults<-rbind(dfResults,TextClassification("tfidf",0,1,verbose=1))  # tf-idf, no stemming, remove stopwords
dfResults<-rbind(dfResults,TextClassification("tf",0,1,verbose=1))     # tf, no stemming, remove stopwords
dfResults<-rbind(dfResults,TextClassification("binary",0,1,verbose=1)) # binary, no stemming, remove stopwords

dfResults<-rbind(dfResults,TextClassification("tfidf",1,1,verbose=1))  # tf-idf, stemming, remove stopwords
dfResults<-rbind(dfResults,TextClassification("tf",1,1,verbose=1))     # tf, stemming, remove stopwords
dfResults<-rbind(dfResults,TextClassification("binary",1,1,verbose=1)) # binary, stemming, remove stopwords

dfResults[, "best_acc"] <- apply(dfResults[, c("nb_acc","svm_acc","nn_acc","rf_acc")], 1, max)
dfResults <- dfResults[order(-dfResults$best_acc),]
dfResults

strResult <- sprintf("Best accuracy score was %1.2f%%. Hyper-parameters: ",dfResults[1,"best_acc"])
strResult <- paste(strResult,dfResults[1,"w"],",",sep="")
strResult <- paste(strResult,
                   ifelse(dfResults[1,"stem"] == 0,"no stemming,","stemming,"))
strResult <- paste(strResult,
                   ifelse(dfResults[1,"stop"] == 0,"no stop word processing,","removed stop words,"))
if (dfResults[1,"best_acc"] == dfResults[1,"nb_acc"]){
  strResult <- paste(strResult,"Naive Bayes model")
} else if (dfResults[1,"best_acc"] == dfResults[1,"svm_acc"]){
  strResult <- paste(strResult,"SVM model")
} else if (dfResults[1,"best_acc"] == dfResults[1,"nn_acc"]){
  strResult <- paste(strResult,"Neural Network model")
}else if (dfResults[1,"best_acc"] == dfResults[1,"rf_acc"]){
  strResult <- paste(strResult,"Random Forest model")
}

print (strResult)

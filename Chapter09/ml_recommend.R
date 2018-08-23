library(readr)
library(recommenderlab)
library(reshape2)

set.seed(0)
in_file <- "../dunnhumby/recommend.csv"
df <- read_csv(in_file)
dfPivot <-dcast(df, cust_id ~ prod_id)
m <- as.matrix(dfPivot[,2:ncol(dfPivot)])

recommend <- as(m,"realRatingMatrix")
e <- evaluationScheme(recommend,method="split",
                      train=0.9,given=-1,
                      goodRating=5)
e

r1 <- Recommender(getData(e,"train"),"UBCF")
r1
p1 <- predict(r1,getData(e,"known"),type="ratings")
err1<-calcPredictionAccuracy(p1,getData(e,"unknown"))
print(sprintf(" User based collaborative filtering model MSE = %1.4f",err1[2]))




options(digits=3)
options(scipen = 999) # disabling scientific notation


printDiagosticMessage <- function(strMessage)
{
  print (strMessage)
}


data<-read.csv("../data/train.csv", header=TRUE,nrow=100)
train.y <- data[,1]
train.x <- data[,-1]
rm(data)



plotInstance <-function (df,title="")
{
  mat <- as.matrix(df)
  mat <- t(apply(mat, 2, rev))
  image(mat, main = title,axes = FALSE, col = grey(seq(0, 1, length = 256)))
}


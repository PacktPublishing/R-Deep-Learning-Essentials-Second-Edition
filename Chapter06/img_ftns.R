library(OpenImageR)

RANDOM_SEED_VALUE <- 42

plotInstance <-function (mat, title="")
{
  mat <- t(apply(mat, 2, rev))
  image(mat, main = title,axes = FALSE, col = grey(seq(0, 1, length = 256)))
}

rotateInstance <-function (df,degrees)
{
  mat <- as.matrix(df)
  mat2 <- rotateImage(mat, degrees, threads = 1)
  df <- data.frame(mat2)
  return (df)
}

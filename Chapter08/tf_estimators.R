#
# note: the file '../dunnhumby/predict.csv' should exist,
# if it does not, then create it using chapter4/prepare_data.R
#
#install.packages("tfestimators")
library(readr)
library(tensorflow)
library(tfestimators)

set.seed(42)
fileName <- "../dunnhumby/predict.csv"
dfData <- read_csv(fileName,
                   col_types = cols(
                     .default = col_double(),
                     CUST_CODE = col_character(),
                     Y_categ = col_integer()),
                   progress=FALSE
                   )
nobs <- nrow(dfData)
train <- sample(nobs, 0.9*nobs)
test <- setdiff(seq_len(nobs), train)
predictorCols <- colnames(dfData)[!(colnames(dfData) %in% c("CUST_CODE","Y_numeric","Y_categ"))]

# data is right-skewed, apply log transformation
dfData[, c("Y_numeric",predictorCols)] <- log(0.01+dfData[, c("Y_numeric",predictorCols)])

trainData <- dfData[train, c(predictorCols)]
testData <- dfData[test, c(predictorCols)]
trainData$Y_categ <- dfData[train, "Y_categ"]$Y_categ
testData$Y_categ <- dfData[test, "Y_categ"]$Y_categ
rm(dfData)

response <- function() "Y_categ"
features <- function() predictorCols

FLAGS <- flags(
  flag_numeric("layer1", 256),
  flag_numeric("layer2", 128),
  flag_numeric("layer3", 64),
  flag_numeric("layer4", 32),
  flag_numeric("dropout", 0.2),
  flag_string("activ","relu")
)
num_hidden <- c(FLAGS$layer1,FLAGS$layer2,FLAGS$layer3,FLAGS$layer4)

classifier <- dnn_classifier(
  feature_columns = feature_columns(column_numeric(predictorCols)),
  hidden_units = num_hidden,
  activation_fn = FLAGS$activ,
  dropout = FLAGS$dropout,
  n_classes = 2
)

bin_input_fn <- function(data)
{
  input_fn(data, features = features(), response = response())
}

tr <- train(classifier, input_fn = bin_input_fn(trainData),
            hooks = list(hook_progress_bar()))
tr
plot(tr)

#predictions <- predict(classifier, input_fn = bin_input_fn(testData))
evaluation <- evaluate(classifier, input_fn = bin_input_fn(testData))

for (c in 1:ncol(evaluation))
  print(paste(colnames(evaluation)[c]," = ",evaluation[c],sep=""))

#model_dir(classifier)
# dnn_classifier has a model_dir parameter to load an existing model
#?dnn_classifier

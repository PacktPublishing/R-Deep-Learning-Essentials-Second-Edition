library(keras)
#devtools::install_github("rstudio/tfdeploy")
library(tfdeploy)

# load data
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset_mnist()

# reshape and rescale
x_train <- array_reshape(x_train, dim=c(nrow(x_train), 784)) / 255
x_test <- array_reshape(x_test, dim=c(nrow(x_test), 784)) / 255

# one-hot encode response
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# define and compile model
model <- keras_model_sequential()
model %>%
  layer_dense(units=256, activation='relu', input_shape=c(784),name="image") %>%
  layer_dense(units=128, activation='relu') %>%
  layer_dense(units=10, activation='softmax',name="prediction") %>%
  compile(
    loss='categorical_crossentropy',
    optimizer=optimizer_rmsprop(),
    metrics=c('accuracy')
  )

# train model
history <- model %>% fit(
  x_train, y_train,
  epochs=10, batch_size=128,
  validation_split=0.2
)
preds <- round(predict(model, x_test[1:5,]),0)
head(preds)



# create a json file for an image from the test set
json <- "{\"instances\": [{\"image_input\": ["
json <- paste(json,paste(x_test[1,],collapse=","),sep="")
json <- paste(json,"]}]}",sep="")
write.table(json,"json_image.json",row.names=FALSE,col.names=FALSE,quote=FALSE)

export_savedmodel(model, "savedmodel")
serve_savedmodel('savedmodel', browse=TRUE)

# open up a command prompt and enter the following (remove the # at start of the string)
#curl -X POST -H "Content-Type: application/json" -d @json_image.json http://localhost:8089/serving_default/predict



# we can also load the CIFAR10 model we created and deploy that
model <- load_model_hdf5("cifar_model.h5")
export_savedmodel(model, "savedmodel")
serve_savedmodel(model, browse = TRUE)

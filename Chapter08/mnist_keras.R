library(keras)

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3), activation='relu',
                input_shape=c(28, 28, 1)) %>% 
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation='relu') %>% 
  layer_max_pooling_2d(pool_size=c(2, 2)) %>% 
  layer_dropout(rate=0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units=128, activation='relu') %>% 
  layer_dropout(rate=0.5) %>% 
  layer_dense(units=10, activation='softmax')

model %>% compile(
  loss=loss_categorical_crossentropy,
  optimizer="rmsprop",
  metrics=c('accuracy')
)

batch_size <- 128
epochs <- 5
model %>% fit(
  x_train, y_train,
  batch_size=batch_size,
  epochs=epochs,
  verbose=1,
  callbacks = callback_tensorboard("/tensorflow_logs",
                                   histogram_freq=1,write_images=0),
  validation_split = 0.2
)
# from cmd line, run 'tensorboard --logdir /tensorflow_logs'

scores <- model %>% evaluate(x_test, y_test,verbose=0)
print(sprintf("test accuracy = %1.4f",scores[[2]]))


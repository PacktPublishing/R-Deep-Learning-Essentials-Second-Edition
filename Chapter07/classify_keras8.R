library(keras)

set.seed(42)
word_index <- dataset_reuters_word_index()
max_features <- length(word_index)
maxlen <- 250
skip_top = 0

reuters <- dataset_reuters(num_words = max_features,skip_top = skip_top)
c(c(x_train, y_train), c(x_test, y_test)) %<-% reuters
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
x_train <- rbind(x_train,x_test)
y_train <- c(y_train,y_test)
table(y_train)

#x_train <- x_train[y_train %in% c(3,4),]
#y_train <- y_train[y_train %in% c(3,4)]
#y_train <- y_train-3
y_train[y_train!=3] <- 0
y_train[y_train==3] <- 1
table(y_train)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32,input_length = maxlen) %>%
  layer_spatial_dropout_1d(rate = 0.25) %>%
  layer_conv_1d(64,3, activation = "relu") %>%
  layer_max_pooling_1d() %>%
  bidirectional(layer_gru(units=64,dropout=0.2)) %>%
  layer_dense(units = 1, activation = "sigmoid")
  
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
summary(model)
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

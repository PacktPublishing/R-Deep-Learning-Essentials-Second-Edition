library(keras)

# train a model from a set of images
# note: you need to run gen_cifar10_data.R first to create the images!
model <- keras_model_sequential()
model %>%
  layer_conv_2d(name="conv1", input_shape=c(32, 32, 3),
    filter=32, kernel_size=c(3,3), padding="same"
  ) %>%
  layer_activation("relu") %>%
  layer_conv_2d(name="conv2",filter=32, kernel_size=c(3,3),
                padding="same") %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25,name="drop1") %>%
  
  layer_conv_2d(name="conv3",filter=64, kernel_size=c(3,3),
                padding="same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(name="conv4",filter=64, kernel_size=c(3,3),
                padding="same") %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25,name="drop2") %>%
  
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(8) %>%
  layer_activation("softmax")

model %>% compile(
  loss="categorical_crossentropy",
  optimizer="adam",
  metrics="accuracy"
)

# set up data generators to stream images to the train function
data_dir <- "../data/cifar_10_images/"
train_dir <- paste(data_dir,"data1/train/",sep="")
valid_dir <- paste(data_dir,"data1/valid/",sep="")

# in  CIFAR10
#  there are 50000 images in training set
#  and 10000 images in test set
#  but we are only using 8/10 classes,
#  so its 40000 train and 8000 validation
num_train <- 40000
num_valid <- 8000
flow_batch_size <- 50
# data augmentation
train_gen <- image_data_generator(
  rotation_range=15,
  width_shift_range=0.2,
  height_shift_range=0.2,
  horizontal_flip=TRUE,
  rescale=1.0/255)
# get images from directory
train_flow <- flow_images_from_directory(
  train_dir,
  train_gen,
  target_size=c(32,32),
  batch_size=flow_batch_size,
  class_mode="categorical"
)

# no augmentation on validation data
valid_gen <- image_data_generator(rescale=1.0/255)
valid_flow <- flow_images_from_directory(
  valid_dir,
  valid_gen,
  target_size=c(32,32),
  batch_size=flow_batch_size,
  class_mode="categorical"
)

# call-backs
callbacks <- list(
  callback_early_stopping(monitor="val_acc",patience=10,mode="auto"),
  callback_model_checkpoint(filepath="cifar_model.h5",mode="auto",
                            monitor="val_loss",save_best_only=TRUE)
)

# train the model using the data generators and call-backs defined above
history <- model %>% fit_generator(
  train_flow,
  steps_per_epoch=as.integer(num_train/flow_batch_size),
  epochs=100,
  callbacks=callbacks,
  validation_data=valid_flow,
  validation_steps=as.integer(num_valid/flow_batch_size)
)

library(keras)

# load model trained in build_cifar10_model.R
model <- load_model_hdf5("cifar_model.h5")

length(model$trainable_weights)
# freeze convolutional layers
freeze_weights(model,from="conv1", to="drop2")
length(model$trainable_weights)
# remove the softmax layer
pop_layer(model)
pop_layer(model)

# add a new layer that has the correct number of nodes for the new task
model %>%
  layer_dense(name="new_dense",units=2, activation='softmax')
summary(model)

# compile the model again
model %>% compile(
  loss = "binary_crossentropy",
  optimizer="adam",
  metrics=c('accuracy')
)

# set up data generators to stream images to the train function
data_dir <- "../data/cifar_10_images/"
train_dir <- paste(data_dir,"data2/train/",sep="")
valid_dir <- paste(data_dir,"data2/valid/",sep="")

# in  CIFAR10, there are 50000 images in training set
#  and 10000 images in test set
#  but we are only using 2/10 classes,
#  so its 10000 train and 2000 validation
num_train <- 10000
num_valid <- 2000
flow_batch_size <- 50
# no data augmentation
train_gen <- image_data_generator(rescale=1.0/255)
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

# train the model using the data generators
history <- model %>% fit_generator(
  train_flow,
  steps_per_epoch=as.integer(num_train/flow_batch_size),
  epochs=10,
  validation_data=valid_flow,
  validation_steps=as.integer(num_valid/flow_batch_size)
)

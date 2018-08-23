library(keras)

# load model trained in build_cifar10_model.R
model <- load_model_hdf5("cifar_model.h5")

#######################

# load an image from the first category in the validation set
#  and generate its prediction
valid_dir <-"../data/cifar_10_images/data1/valid/"
first_dir <- list.dirs(valid_dir, full.names=FALSE, recursive=FALSE)[1]
valid_dir <- paste(valid_dir,first_dir,sep="")
img_path <- paste(valid_dir,list.files(valid_dir)[7],sep="/")

# load image and convert to shape we can use for prediction
img <- image_load(img_path, target_size = c(32,32))
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))
x <- x / 255.0
preds <- model %>% predict(x)
preds <- round(preds,3)
preds

#######################

valid_dir <-"../data/cifar_10_images/data1/valid/"
flow_batch_size <- 50
num_valid <- 8000

valid_gen <- image_data_generator(rescale=1.0/255)
valid_flow <- flow_images_from_directory(
  valid_dir,
  valid_gen,
  target_size=c(32,32),
  batch_size=flow_batch_size,
  class_mode="categorical"
)

evaluate_generator(model,valid_flow,
                   steps=as.integer(num_valid/flow_batch_size))

preds <- predict_generator(model,valid_flow,
                           steps=as.integer(num_valid/flow_batch_size))
dim(preds)

# view the predictions
preds <- round(preds,3)
head(preds)


library(keras)
model <- application_vgg16(weights = 'imagenet', include_top = TRUE)
summary(model)

img_path <- "image1.jpg"
img <- image_load(img_path, target_size = c(224,224))
x <- image_to_array(img)
x <- array_reshape(x, c(1, dim(x)))
x <- imagenet_preprocess_input(x)
preds <- model %>% predict(x)

imagenet_decode_predictions(preds, top = 5)


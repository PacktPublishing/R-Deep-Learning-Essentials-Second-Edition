library(tensorflow)
input_dataset <- tf$examples$tutorials$mnist$input_data
mnist <- input_dataset$read_data_sets("MNIST-data", one_hot = TRUE)

#placeholders
x <- tf$placeholder(tf$float32, shape(NULL,784L))
y_ <- tf$placeholder(tf$float32, shape(NULL,10L))

#init functions
weight_variable <- function(shape)
{
  initial <- tf$truncated_normal(shape, stddev=0.1)
  tf$Variable(initial)
}

bias_variable <- function(shape)
{
  initial <- tf$constant(0.1, shape=shape)
  tf$Variable(initial)
}

#convolution and pooling
conv2d <- function(x, W)
{
  tf$nn$conv2d(x, W, strides=c(1L,1L,1L,1L), padding='SAME')
}

max_pool_2x2 <- function(x)
{
  tf$nn$max_pool(x, ksize=c(1L,2L,2L,1L),strides=c(1L,2L,2L,1L), padding='SAME')
}

x_image <- tf$reshape(x, shape(-1L,28L,28L,1L))

#first convolution layer
W_conv1 <- weight_variable(shape(5L,5L,1L,20L))
b_conv1 <- bias_variable(shape(20L))

h_conv1 <- tf$nn$tanh(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 <- max_pool_2x2(h_conv1)

#second convolution layer
W_conv2 <- weight_variable(shape = shape(5L,5L,20L,50L))
b_conv2 <- bias_variable(shape = shape(50L))

h_conv2 <- tf$nn$relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 <- max_pool_2x2(h_conv2)

#densely connected layer
W_fc1 <- weight_variable(shape(7L*7L*50L,500L))
b_fc1 <- bias_variable(shape(500L))

h_pool2_flat <- tf$reshape(h_pool2, shape(-1L,7L*7L*50L))
h_fc1 <- tf$nn$relu(tf$matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob <- tf$placeholder(tf$float32)
h_fc1_drop <- tf$nn$dropout(h_fc1, keep_prob)

#softmax layer
W_fc2 <- weight_variable(shape(500L,10L))
b_fc2 <- bias_variable(shape(10L))

y_conv <- tf$nn$softmax(tf$matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L))
train_step <- tf$train$AdamOptimizer(0.0001)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())

for (i in 1:10000)
{
  batch <- mnist$train$next_batch(128L)
  batch_xs <- batch[[1]]
  batch_ys <- batch[[2]]
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch_xs, y_ = batch_ys, keep_prob = 1.0))
    print(sprintf("Step %1.0f:accuracy on training data = %1.4f",i, train_accuracy))
  }
  sess$run(train_step,feed_dict = dict(x = batch_xs, y_ = batch_ys, keep_prob = 0.5))
}

test_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
print(sprintf("test accuracy = %1.4f",test_accuracy))

library(tensorflow)

# confirm that TensorFlow runs
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)

set.seed(42)
# create fake data, y = x * 0.1 + 0.3
x <- runif(1000, min=0, max=1)
y <- x * rnorm(1000,1.3,0.05) + rnorm(1000,0.8,0.04)

# y = Wx + b
W <- tf$Variable(tf$random_uniform(shape(1L), -1.0, 1.0))
b <- tf$Variable(tf$zeros(shape(1L)))
y_pred <- W * x + b

# Minimize MSE
loss <- tf$reduce_mean((y_pred - y) ^ 2)
# set-up optimizer
optimizer <- tf$train$GradientDescentOptimizer(0.4)
# what are we trying to optimize?
train <- optimizer$minimize(loss)

# initialize the variables and run the graph
sess = tf$Session()
sess$run(tf$global_variables_initializer())

# Fit the line
for (step in 0:100) {
  if (step %% 10 == 0)
    print(sprintf("Epoch %1.0f:W=%1.4f, b=%1.4f",step,sess$run(W), sess$run(b)))
  sess$run(train)
}

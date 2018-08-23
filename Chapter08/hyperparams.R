
#devtools::install_github("rstudio/tfruns")

library(tfruns)
# FLAGS <- flags(
#   flag_numeric("layer1", 256),
#   flag_numeric("layer2", 128),
#   flag_numeric("layer3", 64),
#   flag_numeric("layer4", 32),
#   flag_numeric("dropout", 0.2),
#   flag_string("activ","relu")
# )

training_run('tf_estimators.R')
training_run('tf_estimators.R', flags = list(layer1=128,layer2=64,layer3=32,layer4=16))
training_run('tf_estimators.R', flags = list(dropout=0.1,activ="tanh"))

ls_runs(order=eval_accuracy)
ls_runs(order=eval_accuracy)[,1:5]

# view best run
dir1 <- ls_runs(order=eval_accuracy)[1,1]
view_run(dir1)

# compare best two runs
dir1 <- ls_runs(order=eval_accuracy)[1,1]
dir2 <- ls_runs(order=eval_accuracy)[2,1]
compare_runs(runs=c(dir1,dir2))


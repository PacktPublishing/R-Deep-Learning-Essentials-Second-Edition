
data_opts<-c("bulls_eye","worms","moon","blocks")
activation_opts<-c("sigmoid","tanh","relu")


shinyUI(fluidPage(
  fluidRow(headerPanel('Neural network example')),
  fluidRow(
    column(3,
           selectInput("data_sel", "Select data", data_opts),
           sliderInput("hidden", "Nodes in hidden layer",min=2, max=9, value=3, step=1),
           sliderInput("epochs","# Epochs:",min=1000, max=10000, value=3000, step=500),
           sliderInput("lr","Learning rate:",min=0.1, max=20.0, value=0.5, step=0.1),
           selectInput("activation_ftn", "Activation Function", activation_opts),
           actionButton("submit", "Run NN Model"),
           actionButton("reset", "Reset")
    ),
    column(9,
           plotOutput('plot1'),
           fluidRow(
             column(6,plotOutput('plot2')),
             column(6,verbatimTextOutput('nn_output'))
           )
    )
  )
))
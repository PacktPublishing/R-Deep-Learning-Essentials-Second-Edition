library(clusterSim)
source("nnet_functions.R")

function(input, output, session) {
  
  values <- reactiveValues(
    txt_output = "",
    dfCost = data.frame(epoch=integer(),cost=double())
  )
  
  observeEvent(input$reset, {
    data_opts<-c("bulls_eye","worms","moon","blocks")
    updateSelectInput(session, "data_sel", choices=data_opts, selected=data_opts[1])
    updateSliderInput(session, "hidden", value=3)
    updateSliderInput(session, "epochs", value=3000)
    updateSliderInput(session, "lr", value=0.5)
    activation_opts<-c("sigmoid","tanh","relu")
    updateSelectInput(session, "activation_ftn", choices=activation_opts, selected=activation_opts[1])
    values$txt_output<-""
    values$plot2<-NULL
  })

  observeEvent(input$data_sel, {
    values$txt_output<-""
  })
  
  output$plot1 <- renderPlot({
    df<-getData(input$data_sel)

    model <- glm(Y ~.,family=binomial(link='logit'),data=df)
    res <- predict(model,newdata=df,type='response')
    res <- ifelse(res > 0.5,1,0)
    df$Y2<-res
    logreg_accuracy <- sum(df$Y==df$Y2) / nrow(df)

    plot(df[,1:2],col=rainbow(2)[(1+df$Y)],main=paste("Logistic regression accuracy on",input$data_sel,"=",logreg_accuracy))
  })
  
  observeEvent(input$submit,{

    df<-getData(input$data_sel)
    X<-as.matrix(df[,1:2])
    Y<-as.matrix(df$Y)

    n_x=ncol(X)
    n_h=input$hidden
    n_y=1
    m <- nrow(X)
    
    # initialise weights
    weights1 <- matrix(0.01*runif(n_h*n_x)-0.005, ncol=n_x, nrow=n_h)
    weights2 <- matrix(0.01*runif(n_y*n_h)-0.005, ncol=n_h, nrow=n_y)
    bias1 <- matrix(rep(0,n_h),nrow=n_h,ncol=1)
    bias2 <- matrix(rep(0,n_y),nrow=n_y,ncol=1)
    
    str1 <-""
    str1 <-"--BEGIN TRAINING MODEL--\n"
    lr <- input$lr
    activation_ftn <- input$activation_ftn
    values$dfCost <- values$dfCost[FALSE,]
    for (i in 0:input$epochs)
    {
      activation2 <- forward_prop(t(X),activation_ftn,weights1,bias1,weights2,bias2)
      cost <- cost_f(activation2,t(Y))
      backward_prop(t(X),t(Y),activation_ftn,weights1,weights2,activation1,activation2)
      weights1 <- weights1 - (lr * dweights1)
      bias1 <- bias1 - (lr * dbias1)
      weights2 <- weights2 - (lr * dweights2)
      bias2 <- bias2 - (lr * dbias2)
      
      if ((i %% 500) == 0)
        str1 <-paste(str1," Cost after",i,"epochs=",round(cost,3),"\n")
      values$dfCost <- rbind(values$dfCost,c(i,cost))
    }
    str1 <-paste(str1,"--END TRAINING MODEL--\n")
    
    # predict is the same as a single forward propogation
    res <- forward_prop(t(X),activation_ftn,weights1,bias1,weights2,bias2)
    res <- ifelse(res > 0.5,1,0)
    df$Y2<-res[1,]
    nn_accuracy <- sum(df$Y==df$Y2) / nrow(df)
    str1 <-paste(str1,"\n\nNeural network Accuracy=",nn_accuracy)

    values$txt_output <-str1
  })
 
  output$nn_output <- renderText({
    values$txt_output
  })
 
  output$plot2 <- renderPlot({
    if (nrow(values$dfCost)>0)
    {
      colnames(values$dfCost) <- c("epoch","cost")
      plot(values$dfCost, main="Cost function by epochs", col="blue", cex=.5)
    }
  })
  
}


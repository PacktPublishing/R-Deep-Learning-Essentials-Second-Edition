source("globals.R")

function(input, output, session) {

  output$imageTable <- renderTable({

    row <- as.numeric(train.x[input$i,])
    df <- data.frame(matrix(row,nrow=28,byrow=TRUE))
    df
  }, digits = 0)
  
  output$imagePlot <- renderPlot({
    row <- as.numeric(train.x[input$i,])
    df <- data.frame(matrix(row,nrow=28,byrow=TRUE))
    plotInstance(df, paste("index:",input$i,", label =",train.y[input$i]))
  })

  output$convPlot <- renderPlot({
    row <- as.numeric(train.x[input$i,])
    df <- data.frame(matrix(row,nrow=28,byrow=TRUE))
    
    x <- input$radio
    if (x==1)
      m2<-matrix(c(1,0,0,1,0,0,1,0,0),nrow=3)
    else if (x==2)
      m2<-matrix(c(1,1,1,0,0,0,0,0,0),nrow=3) 
    else if (x==3)
      m2<-matrix(c(1,0,0,0,1,0,0,0,1),nrow=3)
    else if (x==4)
      m2<-matrix(c(0,0,1,0,1,0,1,0,0),nrow=3)
    
    dfOut<-df[1:(-2+nrow(df)),1:(-2+ncol(df))]
    dfOut[]<-0
    
    for (r in 1:(-2+nrow(df)))
    {
      for (c in 1:(-2+ncol(df)))
      {
        m1<-as.matrix(df[r:(r+2),c:(c+2)])
        res<-m1*m2
        dfOut[r,c]<-sum(res)
      }
    }
    max_dfOut<-max(dfOut)
    dfOut <- 255* dfOut / max_dfOut
    plotInstance(dfOut, paste("index:",input$i,", label =",train.y[input$i]))
  })
  
  observe({
    # We'll use the input$controller variable multiple times, so save it as x
    # for convenience.
    x <- input$radio
    if (x==1)
      strVal<-"111\n000\n000"
    else if (x==2)
      strVal<-"100\n100\n100"
    else if (x==3)
      strVal<-"100\n010\n001"
    else if (x==4)
      strVal<-"001\n010\n100"
    
    # This will change the value of input$inText, based on x
    updateTextAreaInput(session, "conv_layer",value=strVal)
  })
  
}

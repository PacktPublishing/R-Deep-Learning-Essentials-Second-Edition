library(shiny)
library(shinydashboard)

dashboardPage(
  dashboardHeader(title = "R Deep Learning Essentials, second edition"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Home", tabName = "tab_home"),
      menuItem("Convolutional Layers", tabName = "tab_conv_layers")
    )
  ),
  dashboardBody(
    tabItems(
      
      tabItem("tab_home",
              fluidRow(
                box(
                  width = 12, status = "primary", solidHeader = TRUE,
                  title = "Description",
                  includeMarkdown("description.md")
                )
              )
      ),

      tabItem("tab_conv_layers",
        fluidRow(
          box(
            width = 12, status = "info", solidHeader = TRUE,
            title = "Convolutional Layers",
            fluidRow(
              column(7,
                     sliderInput("i", "Select an image:",
                          min = 1, max = 100, value = 1)
              ),
              column(3,
                     radioButtons("radio", label = "Select convolutional layer",
                                  choices=list("Horizontal Line"=1,"Vertical Line"=2,"Diagonal1"=3,"Diagonal2"=4),
                                  selected = 1)
              ),
              column(2,
                     textAreaInput("conv_layer", "Conv Layer", value = "111\n000\n000",cols=3, rows=3)
              )
              
            )
          )
        ),
        fluidRow(
          box(
            width = 6, status = "info", solidHeader = TRUE,
            title = "Original image",
            imageOutput("imagePlot")
          ),
          box(
            width = 6, status = "info", solidHeader = TRUE,
            title = "Image after Convolutional applied",
            imageOutput("convPlot")
          )
        )
        
      )

    )
  )
)

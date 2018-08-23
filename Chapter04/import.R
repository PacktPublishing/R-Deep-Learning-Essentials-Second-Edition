library(readr)
data_directory <- "../dunnhumby/"
?list.files
import_data <- function (data_directory,bExploreData=1)
{
  if (bExploreData==1)
  {
    file_list <- list.files(paste(data_directory,"in/",sep=""), "transactions_", recursive =TRUE)
    out_file <- paste(data_directory,"explore.csv",sep="")
    column_list <- c("BASKET_ID","CUST_CODE","SHOP_WEEK","SHOP_DATE","STORE_CODE",
                     "QUANTITY","SPEND","PROD_CODE",
                     "PROD_CODE_10","PROD_CODE_20","PROD_CODE_30","PROD_CODE_40")
    if (file.exists(out_file)) return(out_file)
  }
  else
  {
    file_list <- list.files(paste(data_directory,"in/",sep=""), "transactions_2008..\\.csv", recursive =TRUE)
    out_file <- paste(data_directory,"temp.csv",sep="")
    column_list <- c("CUST_CODE","SHOP_WEEK","SHOP_DATE","SPEND",
                     "PROD_CODE_10","PROD_CODE_20","PROD_CODE_30","PROD_CODE_40")
    if (file.exists(out_file)) file.remove(out_file)
  }

  options(readr.show_progress=FALSE)
  i <- 1
  for (file_name in file_list)
  {
    file_name<-paste(data_directory,"in/",file_name,sep="")
    df<-suppressMessages(read_csv(file_name))
    df <- df[!is.na(df$CUST_CODE),column_list]
    
    if (i ==1)
      write_csv(df,out_file)
    else
      write_csv(df,out_file,append=TRUE)
    print (paste("File",i,"/",length(file_list),"processed"))
    i <- i+1
  }
  return(out_file)
}

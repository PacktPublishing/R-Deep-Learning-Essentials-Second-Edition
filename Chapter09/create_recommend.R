library(magrittr)
library(dplyr)
library(readr)
library(broom)

set.seed(42)
file_list <- list.files("../dunnhumby/in/", "trans*")

temp_file <- "../dunnhumby/temp.csv"
out_file <- "../dunnhumby/recommend.csv"
if (file.exists(temp_file)) file.remove(temp_file)
if (file.exists(out_file)) file.remove(out_file)

options(readr.show_progress=FALSE)
i <- 1
for (file_name in file_list)
{
  file_name<-paste("../dunnhumby/in/",file_name,sep="")
  df<-suppressMessages(read_csv(file_name))
  
  df2 <- df %>%
    filter(CUST_CODE!="") %>%
    group_by(CUST_CODE,PROD_CODE_40) %>%
    summarise(sales=sum(SPEND))
  
  colnames(df2)<-c("cust_id","prod_id","sales")
  if (i ==1)
    write_csv(df2,temp_file)
  else
    write_csv(df2,temp_file,append=TRUE)
  print (paste("File",i,"/",length(file_list),"processed"))
  i <- i+1
}
rm(df,df2)
df_processed<-read_csv(temp_file)
if (file.exists(temp_file)) file.remove(temp_file)

df2 <- df_processed %>%
  group_by(cust_id,prod_id) %>%
  summarise(sales=sum(sales))

# create quantiles
dfProds <- df2 %>%
  group_by(prod_id) %>%
  do( tidy(t(quantile(.$sales, probs = seq(0, 1, 0.2)))) )
colnames(dfProds)<-c("prod_id","X0","X20","X40","X60","X80","X100")
df2<-merge(df2,dfProds)
df2$rating<-0
df2[df2$sales<=df2$X20,"rating"] <- 1
df2[(df2$sales>df2$X20) & (df2$sales<=df2$X40),"rating"] <- 2
df2[(df2$sales>df2$X40) & (df2$sales<=df2$X60),"rating"] <- 3
df2[(df2$sales>df2$X60) & (df2$sales<=df2$X80),"rating"] <- 4
df2[(df2$sales>df2$X80) & (df2$sales<=df2$X100),"rating"] <- 5

# sanity check, are our ratings spread out relatively evenly
df2 %>%
  group_by(rating) %>%
  summarise(recs=n())
df2 %>%
  filter(prod_id==df2[sample(1:nrow(df2), 1),]$prod_id) %>%
  group_by(prod_id,rating) %>%
  summarise(recs=n())


df2 <- df2[,c("cust_id","prod_id","rating")]
write_csv(df2,out_file)

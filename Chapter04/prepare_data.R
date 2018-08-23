library(readr)
library(reshape2)
library(dplyr)

source("import.R")

# step 1, merge files
fileName <- import_data(data_directory,bExploreData=0)

# step 2, group and pivot data
fileOut <- paste(data_directory,"predict.csv",sep="")
df <- read_csv(fileName,col_types = cols(.default = col_character()))

#remove temp file
if (file.exists(fileName)) file.remove(fileName)

# convert spend to numeric field
df$SPEND<-as.numeric(df$SPEND)

# group sales by date. we have not converted the SHOP_DATE to date
# but since it is in yyyymmdd format,
# then ordering alphabetically will preserve date order
sumSalesByDate<-df %>%
  group_by(SHOP_WEEK,SHOP_DATE) %>%
  summarise(sales = sum(SPEND)
  )

# we want to get the cut-off date to create our data model
# this is the last date and go back 13 days beforehand
# therefore our X data only looks at everything from start to max date - 13 days
#  and our Y data only looks at everything from max date - 13 days to end (i.e. 14 days)
max(sumSalesByDate$SHOP_DATE)
sumSalesByDate2 <- sumSalesByDate[order(sumSalesByDate$SHOP_DATE),]
datCutOff <- as.character(sumSalesByDate2[(nrow(sumSalesByDate2)-13),]$SHOP_DATE)
datCutOff
rm(sumSalesByDate,sumSalesByDate2)

# we are going to limit our data here from year 2008 only
# group data and then pivot it
sumTemp <- df %>%
  filter((SHOP_DATE < datCutOff) & (SHOP_WEEK>="200801")) %>%
  group_by(CUST_CODE,SHOP_WEEK,PROD_CODE_40) %>%
  summarise(sales = sum(SPEND)
  )
sumTemp$fieldName <- paste(sumTemp$PROD_CODE_40,sumTemp$SHOP_WEEK,sep="_")
df_X <- dcast(sumTemp,CUST_CODE ~ fieldName, value.var="sales")
df_X[is.na(df_X)] <- 0

# y data just needs a group to get sales after cut-off date
df_Y <- df %>%
  filter(SHOP_DATE >= datCutOff) %>%
  group_by(CUST_CODE) %>%
  summarise(sales = sum(SPEND)
  )
colnames(df_Y)[2] <- "Y_numeric"

# use left join on X and Y data, need to include all values from X
#  even if there is no Y value
dfModelData <- merge(df_X,df_Y,by="CUST_CODE", all.x=TRUE)
# set binary flag
dfModelData$Y_categ <- 0
dfModelData[!is.na(dfModelData$Y_numeric),]$Y_categ <- 1
dfModelData[is.na(dfModelData$Y_numeric),]$Y_numeric <- 0
rm(df,df_X,df_Y,sumTemp)

nrow(dfModelData)
table(dfModelData$Y_categ)

# shuffle data
dfModelData <- dfModelData[sample(nrow(dfModelData)),]

write_csv(dfModelData,fileOut)


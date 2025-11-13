
# 2095: Statistical and Machine Learning - Project 

# 1) Libraries 

library("tibble")
library("dplyr")

# 2) Reading the data

# Getting location of the data file 

path <- file.path(here::here(""))
data_path <- file.path(path,"02_data","flight_data_2024.csv")
data <- read.csv(data_path)

# 3) Handling big data 
# Since the data is huge 7 million rows (1+ GB of data)
# We reduce the number of observations by removing some carriers

carrier_count <- table(data$op_unique_carrier)
print(carrier_count)

# In total we have 15 unique carriers 
# In this project we will only focus on the top 4 carriers by number of flights

# Sorting the above table 
sort_carrier_count <- sort(carrier_count, decreasing = TRUE)
top4_carriers <- head(sort_carrier_count, 4)
top4_codes <- names(top5_carriers)

# Here are the top 4 carriers: 
# WN = "Southwest Airlines", DL = "Delta Air Lines", 
# AA = "American Airlines",UA = "United Airlines"

# Filtering the data to only include top 4 carriers 

data_filtered <- data %>%
  filter(op_unique_carrier %in% top4_codes)

  



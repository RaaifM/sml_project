# sml_project
Statistical &amp; Machine learning project for the course 2095. 

1) We consider flights that are cancelled, diverted, and arrival daly > 15 as being a delayed flight (1: delayed, 0: not delayed)
2) We drop all columns that are redundent and/or include post event information. So we only use data that can be observed before the flights takes-off. 
3) There are 107 factors for the departure airport and 107 factors for the destination airports. This is creating way too many dummy variables and the model is not able to handle this well. So we need to find a better approach for dealing with them. One way is to use the big dataframe and calculate the probability of delay for each departure and arrival airport and use this in the regression. 

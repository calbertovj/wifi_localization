# libraries --------------

pacman::p_load(readr, caret, ggplot2, reshape2, DataExplorer, MASS, tidyr, mlbench,
               corrplot, Hmisc, PerformanceAnalytics, ggiraphExtra, BBmisc, dplyr, padr,
               imputeTS, xgboost, plyr, h2o, Boruta, plotly, sp, ggmap, RColorBrewer, 
               SuperLearner, kernlab, arm)

# General plot settings ----------------
plot.settings <- theme(
  axis.line.x =       element_line(colour = "black", size = 1),                                                       # Settings x-axis line
  axis.line.y =       element_line(colour = "black", size = 1),                                                       # Settings y-axis line 
  axis.text.x =       element_text(colour = "black", size = 12, lineheight = 0.9, vjust = 1, face = "bold"),        # Font x-axis 
  axis.text.y =       element_text(colour = "black", size = 12, lineheight = 0.9, hjust = 1),                         # Font y-axis
  axis.ticks =        element_line(colour = "black", size = 0.3),                                                     # Color/thickness axis ticks
  axis.title.x =      element_text(size = 20, vjust = 1, face = "bold", margin = margin(10,1,1,1)),                   # Font x-axis title
  axis.title.y =      element_text(size = 20, angle = 90, vjust = 1, face = "bold", margin = margin(1,10,1,1)),       # Font y-axis title
  
  legend.background = element_rect(colour=NA),                                                                        # Background color legend
  legend.key =        element_blank(),                                                                                # Background color legend key
  legend.key.size =   unit(1.2, "lines"),                                                                             # Size legend key
  legend.text =       element_text(size = 18),                                                                        # Font legend text
  legend.title =      element_text(size = 20, face = "bold", hjust = 0),                                              # Font legend title  
  legend.position =   "right",                                                                                        # Legend position
  
  panel.background =  element_blank(),                                                                                # Background color graph
  panel.border =      element_blank(),                                                                                # Border around graph (use element_rect())
  panel.grid.major =  element_blank(),                                                                                # Major gridlines (use element_line())
  panel.grid.minor =  element_blank(),                                                                                # Minor gridlines (use element_line())
  panel.margin =      unit(1, "lines"),                                                                               # Panel margins
  
  strip.background =  element_rect(fill = "grey80", colour = "grey50"),                                               # Background colour strip 
  strip.text.x =      element_text(size = 20),                                                                        # Font strip text x-axis
  strip.text.y =      element_text(size = 20, angle = -90),                                                           # Font strip text y-axis
  
  plot.background =   element_rect(colour = NA),                                                                      # Background color of entire plot
  plot.title =        element_text(size = 20, face = "bold", hjust = 0.5),                                                                        # Font plot title 
  plot.margin =       unit(c(1, 1, 1, 1), "lines")                                                                    # Plot margins
)

# Reading data -------------

traindata <- read.csv("../trainingData.csv")
validationdata  <- read.csv("../validationData.csv")

#Preproccessing  ---------------

traindata1 <- traindata %>% 
  dplyr::select(-c(SPACEID,RELATIVEPOSITION,USERID,PHONEID,TIMESTAMP)) %>% 
  unite("building_floor", BUILDINGID,FLOOR, remove = FALSE) %>% 
  dplyr::select(-c(FLOOR)) %>% 
  mutate(BUILDINGID = factor(BUILDINGID), building_floor = factor(building_floor))

levels(traindata1$BUILDINGID) <- c("B0","B1","B2")
traindata1[ , 1:524 ][ traindata1[ , 1:524 ] == 100 ] <- -1000

zerovariance <- function(data) {
  out <- lapply(data, function(x) length(unique(x)))
  want <- which(!out > 1)
  unlist(want)
}

traindata2 <- traindata1[,-zerovariance(traindata1)]

traindata_ID_1 <-  dplyr::select(traindata1, -c(LONGITUDE, LATITUDE, building_floor))
traindata_bf_1 <-  dplyr::select(traindata1, -c(LONGITUDE, LATITUDE, BUILDINGID))
traindata_lon_1 <-  dplyr::select(traindata1, -c(building_floor, LATITUDE, BUILDINGID))
traindata_lat_1 <-  dplyr::select(traindata1, -c(LONGITUDE, building_floor, BUILDINGID))

#validation data

validationdata1 <- validationdata %>% 
  dplyr::select(-c(SPACEID,RELATIVEPOSITION,USERID,PHONEID,TIMESTAMP)) %>% 
  unite("building_floor", BUILDINGID,FLOOR, remove = FALSE) %>% 
  dplyr::select(-c(FLOOR)) %>% 
  mutate(BUILDINGID = factor(BUILDINGID), building_floor = factor(building_floor)) 

levels(validationdata1$BUILDINGID) <- c("B0","B1","B2")
validationdata1[ , 1:524 ][ validationdata1[ , 1:524 ] == 100 ] <- -1000

validationdata_all_1 <-  dplyr::select(validationdata1, -c(LONGITUDE, LATITUDE, building_floor,BUILDINGID))

# PCA - Dimension reduction -------

trainndata_pca <-  dplyr::select(traindata2, -c(LONGITUDE, LATITUDE, building_floor, BUILDINGID))

#principal component analysis
prin_comp <- prcomp(trainndata_pca, scale. = T)
names(prin_comp)

biplot(prin_comp, scale = 0)

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]

#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained", type = "b")

#add a training set with principal components
train.pca.ID <- data.frame(BUILDINGID = traindata2$BUILDINGID, prin_comp$x)
train.pca.bg <- data.frame(building_floor = traindata2$building_floor, prin_comp$x)
train.pca.lon <- data.frame(LONGITUDE = traindata2$LONGITUDE, prin_comp$x)
train.pca.lat <- data.frame(LATITUDE = traindata2$LATITUDE, prin_comp$x)

#we are interested in first 310 PCAs
train.pca.ID <- train.pca.ID[,1:311]
train.pca.bg <- train.pca.bg[,1:311]
train.pca.lon <- train.pca.lon[,1:311]
train.pca.lat <- train.pca.lat[,1:311]

#transform test into PCA
validationdata_pca <-  dplyr::select(validationdata1, -c(LONGITUDE, LATITUDE, building_floor, BUILDINGID))
validation.data_pca <- predict(prin_comp, newdata = validationdata_pca)
validation.data_pca <- as.data.frame(validation.data_pca)

#select the first 310 components
validation.data_pca <- validation.data_pca[,1:310]


#Model for PCA's in Ranger --------
BUILDINGID_ranger_pca <- ranger(formula= BUILDINGID ~ ., data= train.pca.ID, seed = 12345)

# prediction BuildingID
pred_ID_ranger_pca <- predict(BUILDINGID_ranger_pca, validation.data_pca)
confusionMatrix(table(validationdata_ID$BUILDINGID, pred_ID_ranger_pca$predictions))

#Model for building_floor 
bf_ranger_pca <- ranger(formula= building_floor ~ .,data= train.pca.bg, seed = 12345)

# prediction building_floor
pred_bf_ranger_pca <- predict(bf_ranger_pca, validation.data_pca)
confusionMatrix(table(validationdata_bf$building_floor, pred_bf_ranger_pca$predictions))

#Model for longitude
lon_ranger_pca <- ranger(formula= LONGITUDE ~ ., data= train.pca.lon, seed = 12345)

# prediction longitude
pred_lon_ranger_pca <- predict(lon_ranger_pca, validation.data_pca)
postResample(validationdata_lon$LONGITUDE, pred_lon_ranger_pca$predictions)

#Model for latitude
lat_ranger_pca <- ranger(formula= LATITUDE ~ ., data= train.pca.lat, seed = 12345)

# prediction latitude
pred_lat_ranger_pca <- predict(lat_ranger_pca, validation.data_pca)
postResample(validationdata_lat$LATITUDE, pred_lat_ranger_pca$predictions)


# building_floor in Ranger -----

#Model for building_floor
set.seed(12345)

bf_ranger_all <- ranger(
  formula   = building_floor ~ ., 
  data      = traindata_bf_1, 
)

# prediction building_floor
pred_bf_ranger_all <- predict(bf_ranger_all, validationdata_all_1)
confusionMatrix(table(validationdata2$building_floor, pred_bf_ranger_all$predictions))



# Boruta - Dimension reduction -------

traindata2 <- traindata1[,-zerovariance(traindata1)]

traindata_ID_2 <-  dplyr::select(traindata2, -c(LONGITUDE, LATITUDE, building_floor))

boruta.train <- Boruta(BUILDINGID ~ . , data = traindata_ID_2, doTrace = 3, maxRuns = 150)

boruta.final <- TentativeRoughFix(boruta.train)

best.waps <- getSelectedAttributes(boruta.final)

traindata3 <- traindata2 %>% 
  select(c(best.waps, "LONGITUDE", "LATITUDE", "BUILDINGID","building_floor"))

traindata_ID <-  dplyr::select(traindata3, -c(LONGITUDE, LATITUDE, building_floor))
traindata_bf <-  dplyr::select(traindata3, -c(LONGITUDE, LATITUDE, BUILDINGID))
traindata_lon <-  dplyr::select(traindata3, -c(building_floor, LATITUDE, BUILDINGID))
traindata_lat <-  dplyr::select(traindata3, -c(LONGITUDE, building_floor, BUILDINGID))

validationdata2 <- validationdata1 %>% 
  select(c(best.waps, "LONGITUDE", "LATITUDE", "BUILDINGID","building_floor"))


validationdata_all <-  dplyr::select(validationdata2, -c(LONGITUDE, LATITUDE, building_floor,BUILDINGID))

#Model for Boruta in Ranger --------
BUILDINGID_ranger <- ranger(
  formula   = BUILDINGID ~ ., 
  data      = traindata_ID, 
  )

# prediction BuildingID
pred_ID_ranger <- predict(BUILDINGID_ranger, validationdata_all)
confusionMatrix(table(validationdata2$BUILDINGID, pred_ID_ranger$predictions))

#Model for building_floor
bf_ranger <- ranger(
  formula   = building_floor ~ ., 
  data      = traindata_bf, 
)

# prediction building_floor
pred_bf_ranger <- predict(bf_ranger, validationdata_all)
confusionMatrix(table(validationdata2$building_floor, pred_bf_ranger$predictions))

#Model for longitude
lon_ranger <- ranger(
  formula   = LONGITUDE ~ ., 
  data      = traindata_lon, 
)

# prediction longitude
pred_lon_ranger <- predict(lon_ranger, validationdata_all)
postResample(validationdata2$LONGITUDE, pred_lon_ranger$predictions)

#Model for latitude
lat_ranger <- ranger(
  formula   = LATITUDE ~ ., 
  data      = traindata_lat, 
)

# prediction latitude
pred_lat_ranger <- predict(lat_ranger, validationdata_all)
postResample(validationdata2$LATITUDE, pred_lat_ranger$predictions)



# h2o model for Boruta --------

h2o.init()

train_ID_h2o <- as.h2o(traindata_ID)
train_bf_h2o <- as.h2o(traindata_bf)
train_lon_h2o <- as.h2o(traindata_lon)
train_lat_h2o <- as.h2o(traindata_lat)

ID <- "BUILDINGID"
bf <- "building_floor"
lon <- "LONGITUDE"
lat <- "LATITUDE"

fit_ID_h2o <- h2o.automl(y = ID, training_frame = train_ID_h2o,
                         max_runtime_secs = 300,
                         nfolds = 10, seed = 12345,
                         project_name = "ID_h2o")

fit_bf_h2o <- h2o.automl(y = bf, training_frame = train_bf_h2o,
                         max_runtime_secs = 300,
                         nfolds = 10, seed = 12345,
                         project_name = "bf_h2o")

fit_lon_h2o <- h2o.automl(y = lon, training_frame = train_lon_h2o,
                         max_runtime_secs = 300,
                         nfolds = 10, seed = 12345,
                         project_name = "lon_h2o")

fit_lat_h2o <- h2o.automl(y = lat, training_frame = train_lat_h2o,
                         max_runtime_secs = 300,
                         nfolds = 10, seed = 12345,
                         project_name = "lat_h2o")


validation_h2o <- as.h2o(validationdata_all)

pred_ID_h2o <- h2o.predict(fit_ID_h2o, validation_h2o)
pred_bf_h2o <- h2o.predict(fit_bf_h2o, validation_h2o)
pred_lon_h2o <- h2o.predict(fit_lon_h2o, validation_h2o)
pred_lat_h2o <- h2o.predict(fit_lat_h2o, validation_h2o)

# results
id.pred.h2o <- as.data.frame(pred_ID_h2o$predict)
bf.pred.h2o <- as.data.frame(pred_bf_h2o$predict)
lon.pred.h2o <- as.data.frame(pred_lon_h2o$predict)
lat.pred.h2o <- as.data.frame(pred_lat_h2o$predict)

# error metrics

confusionMatrix(table(validationdata2$BUILDINGID, id.pred.h2o$predict))
confusionMatrix(table(validationdata2$building_floor, bf.pred.h2o$predict))
postResample(validationdata2$LONGITUDE, lon.pred.h2o$predict)
postResample(validationdata2$LATITUDE, lat.pred.h2o$predict)

h2o.shutdown()


# Parallel working -------------
library(doParallel)
cl <- makePSOCKcluster(6)
registerDoParallel(cl)
stopCluster(cl)

# KNN model for Boruta -------

# subsetting data for training model
train_ID_knn <- createDataPartition(traindata_ID$BUILDINGID,
                                p = 0.25, list = F)
train_ID_knn1 <- traindata_ID[train_ID_knn,]

ctrl <- trainControl(method = "repeatedcv", repeats = 3)

# BUILDINGID
fit_id_knn <- train(BUILDINGID ~ ., data = traindata_ID, 
                  method = "knn", trControl = ctrl, 
                  preProcess = c("center", "scale"),
                  tuneLength = 20)

pred_ID_knn <- predict(fit_id_knn, newdata = validationdata_all)
postResample(pred_ID_knn, validationdata2$BUILDINGID)

# building_floor
fit_bf_knn <- train(building_floor ~ ., data = traindata_bf, 
                    method = "knn", trControl = ctrl, 
                    preProcess = c("center", "scale"),
                    tuneLength = 20)

pred_bf_knn <- predict(fit_bf_knn, newdata = validationdata_all)
postResample(pred_bf_knn, validationdata2$building_floor)

# LONGITUDE
fit_lon_knn <- train(LONGITUDE ~ ., data = traindata_lon, 
                    method = "knn", trControl = ctrl, 
                    preProcess = c("center", "scale"),
                    tuneLength = 20)

pred_lon_knn <- predict(fit_lon_knn, newdata = validationdata_all)
postResample(pred_lon_knn, validationdata2$LONGITUDE)

# LATITUDE
fit_lat_knn <- train(LATITUDE ~ ., data = traindata_lat, 
                    method = "knn", trControl = ctrl, 
                    preProcess = c("center", "scale"),
                    tuneLength = 20)

pred_lat_knn <- predict(fit_lat_knn, newdata = validationdata_all)
postResample(pred_lat_knn, validationdata2$LATITUDE)


# Calculating errors-------

# errors for longitude
errors_lon <- as.data.frame(validationdata2$LONGITUDE - pred_lon_ranger$predictions)
colnames(errors_lon) <- "error"
errors_lon$abs.error <- abs(errors_lon$error)
errors_lon$error.squared <- (errors_lon$error * errors_lon$error)
errors_lon$BUILDINGID <- validationdata2$BUILDINGID
errors_lon$building_floor <- validationdata2$building_floor

# errors for latitude
errors_lat <- as.data.frame(validationdata2$LATITUDE - pred_lat_ranger$predictions)
colnames(errors_lat) <- "error"
errors_lat$abs.error <- abs(errors_lat$error)
errors_lat$error.squared <- (errors_lat$error * errors_lat$error)
errors_lat$BUILDINGID <- validationdata2$BUILDINGID
errors_lat$building_floor <- validationdata2$building_floor

# plotting errors for Longitude per  building ---------

plot_errors_lon_ID_0 <- ggplot(subset(errors_lon,BUILDINGID %in% "B0"),
                               aes(x=error, color = building_floor, fill = building_floor)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 0, color = "red", size=1) +
  ggtitle('Distribution of errors in Building B0')

plot_errors_lon_ID_0  <- plot_errors_lon_ID_0 + plot.settings
plot_errors_lon_ID_0 <-  ggplotly(plot_errors_lon_ID_0)

plot_errors_lon_ID_1 <- ggplot(subset(errors_lon,BUILDINGID %in% "B1"),
                               aes(x=error, color = building_floor, fill = building_floor)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 0, color = "red", size=1) +
  ggtitle('Distribution of errors in Building B1')

plot_errors_lon_ID_1  <- plot_errors_lon_ID_1 + plot.settings
plot_errors_lon_ID_1 <-  ggplotly(plot_errors_lon_ID_1)

plot_errors_lon_ID_2 <- ggplot(subset(errors_lon,BUILDINGID %in% "B2"),
                               aes(x=error, color = building_floor, fill = building_floor)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 0, color = "red", size=1) +
  ggtitle('Distribution of errors in Building B2')

plot_errors_lon_ID_2  <- plot_errors_lon_ID_2 + plot.settings
plot_errors_lon_ID_2 <-  ggplotly(plot_errors_lon_ID_2)

# plotting errors for Latitude per building ---------

plot_errors_lat_ID_0 <- ggplot(subset(errors_lat,BUILDINGID %in% "B0"),
                               aes(x=error, color = building_floor, fill = building_floor)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 0, color = "red", size=1) +
  ggtitle('Distribution of errors in Building B0')

plot_errors_lat_ID_0  <- plot_errors_lat_ID_0 + plot.settings
plot_errors_lat_ID_0 <-  ggplotly(plot_errors_lat_ID_0)

plot_errors_lat_ID_1 <- ggplot(subset(errors_lat,BUILDINGID %in% "B1"),
                               aes(x=error, color = building_floor, fill = building_floor)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 0, color = "red", size=1) +
  ggtitle('Distribution of errors in Building B1')

plot_errors_lat_ID_1  <- plot_errors_lat_ID_1 + plot.settings
plot_errors_lat_ID_1 <-  ggplotly(plot_errors_lat_ID_1)

plot_errors_lat_ID_2 <- ggplot(subset(errors_lat,BUILDINGID %in% "B2"),
                               aes(x=error, color = building_floor, fill = building_floor)) + 
  geom_density(alpha = 0.2) +
  geom_vline(xintercept = 0, color = "red", size=1) +
  ggtitle('Distribution of errors in Building B2')

plot_errors_lat_ID_2  <- plot_errors_lat_ID_2 + plot.settings
plot_errors_lat_ID_2 <-  ggplotly(plot_errors_lat_ID_2)



plot(traindata$LONGITUDE,traindata$LATITUDE)

# put coordinates on a map -----------

coordinates_train <- read.csv("../Claudin/coor_train.csv")
coordinates_val  <- read.csv("../Claudin/coor_validation.csv")

coordinates_train$BUILDINGID <- traindata3$BUILDINGID
coordinates_val$BUILDINGID <- validationdata2$BUILDINGID

# creating a googlemaps plot

register_google(key = "AIzaSyBuuXXWaHv37guat1Q3adREvN6jVO-b4Zo")

# use qmplot to make a scatterplot on a map
qmplot(LONGITUDE, LATITUDE, data = coordinates_train, color = I("red"))

revgeocode(c(lon = -0.06774432, lat = 39.99297))

map <- get_googlemap("Escuela Superior de Tecnologia I Ciencias Experimentales",
                     zoom = 17, size=c(640,640), scale = 4, language = "en-EN",
                     maptype = "hybrid") 

ggmap(map)

ggmap(map, extent = "device") +
  geom_point(aes(x = LONGITUDE, y = LATITUDE, colour = BUILDINGID), data = coordinates_train,
             size = 3, alpha = 0.02)

ggmap(map, extent = "device") +
  geom_point(aes(x = LONGITUDE, y = LATITUDE, colour = BUILDINGID), data = coordinates_val,
             size = 3, alpha = 0.1)


ggmap(map) +
  stat_density_2d(data = coordinates_train,
                  aes(x = LONGITUDE,
                      y = LATITUDE,
                      fill = stat(level)),
                  alpha = .2,
                  bins = 25,
                  geom = "polygon") +
  scale_fill_gradientn(colors = brewer.pal(7, "YlOrRd"))

ggmap(map) +
  stat_density_2d(data = coordinates_val,
                  aes(x = LONGITUDE,
                      y = LATITUDE,
                      fill = stat(level)),
                  alpha = .2,
                  bins = 25,
                  geom = "polygon") +
  scale_fill_gradientn(colors = brewer.pal(7, "YlOrRd"))

# SuperLearner -----
SL_train_lon <- data.frame(traindata_lon[,1:197])
SL_train_lon_y <-as.numeric(traindata_lon[,198])


# Fit the ensemble model for Longitude
SL.lon <- SuperLearner(SL_train_lon_y,
                      SL_train_lon,
                      family=gaussian(),
                      SL.library=list("SL.ranger",
                                      "SL.ksvm",
                                      "SL.bayesglm"))

SL_train_lat <- data.frame(traindata_lat[,1:197])
SL_train_lat_y <-as.numeric(traindata_lat[,198])


# Fit the ensemble model for Latitude
SL.lat <- SuperLearner(SL_train_lat_y,
                       SL_train_lat,
                       family=gaussian(),
                       SL.library=list("SL.ranger",
                                       "SL.ksvm",
                                       "SL.bayesglm"))


# prediction Longitude
pred_lon_SL <- predict.SuperLearner(SL.lon, validationdata_all)
postResample(validationdata2$LONGITUDE, pred_lon_SL$pred)

# prediction Latitude
pred_lat_SL <- predict.SuperLearner(SL.lat, validationdata_all)
postResample(validationdata2$LATITUDE, pred_lat_SL$pred)





# combine data ---------

combined_data <- rbind(traindata,validationdata) 

combined_data1 <- combined_data %>% 
  dplyr::select(-c(SPACEID,RELATIVEPOSITION,USERID,PHONEID,TIMESTAMP)) %>% 
  unite("building_floor", BUILDINGID,FLOOR, remove = FALSE) %>% 
  dplyr::select(-c(FLOOR)) %>% 
  mutate(BUILDINGID = factor(BUILDINGID), building_floor = factor(building_floor))

levels(combined_data1$BUILDINGID) <- c("B0","B1","B2")
combined_data1[ , 1:524 ][ combined_data1[ , 1:524 ] == 100 ] <- -1000

combined_data_ID <-  dplyr::select(combined_data1, -c(LONGITUDE, LATITUDE, building_floor))
combined_data_bf <-  dplyr::select(combined_data1, -c(LONGITUDE, LATITUDE, BUILDINGID))
combined_data_lon <-  dplyr::select(combined_data1, -c(building_floor, LATITUDE, BUILDINGID))
combined_data_lat <-  dplyr::select(combined_data1, -c(LONGITUDE, building_floor, BUILDINGID))

boruta.combined <- Boruta(BUILDINGID ~ . , data = combined_data_ID, doTrace = 3, maxRuns = 150)

boruta.combined.final <- TentativeRoughFix(boruta.combined)

best.waps.combined <- getSelectedAttributes(boruta.combined.final)

combined_data2 <- combined_data1 %>% 
  select(c(best.waps.combined, "LONGITUDE", "LATITUDE", "BUILDINGID","building_floor"))

combined_data_ID_1 <-  dplyr::select(combined_data2, -c(LONGITUDE, LATITUDE, building_floor))
combined_data_bf_1 <-  dplyr::select(combined_data2, -c(LONGITUDE, LATITUDE, BUILDINGID))
combined_data_lon_1 <-  dplyr::select(combined_data2, -c(building_floor, LATITUDE, BUILDINGID))
combined_data_lat_1 <-  dplyr::select(combined_data2, -c(LONGITUDE, building_floor, BUILDINGID))

# loading real data ---------

realdata <- read.csv("../real_testData.csv")

realdata1 <- realdata %>% 
  dplyr::select(-c(SPACEID,RELATIVEPOSITION,USERID,PHONEID,TIMESTAMP)) %>% 
  unite("building_floor", BUILDINGID,FLOOR, remove = FALSE) %>% 
  dplyr::select(-c(FLOOR)) %>% 
  mutate(BUILDINGID = factor(BUILDINGID), building_floor = factor(building_floor))

levels(realdata1$BUILDINGID) <- c("B0","B1","B2")
realdata1[ , 1:524 ][ realdata1[ , 1:524 ] == 100 ] <- -1000

# Models on combinated data only -------
#Model for building
BUILDINGID_ranger_real <- ranger(
  formula   = BUILDINGID ~ ., 
  data      = combined_data_ID, 
)

#Model for building_floor
bf_ranger_real <- ranger(
  formula   = building_floor ~ ., 
  data      = combined_data_bf, 
)

#Model for longitude
lon_ranger_real <- ranger(
  formula   = LONGITUDE ~ ., 
  data      = combined_data_lon, 
)

#Model for latitude
lat_ranger_real <- ranger(
  formula   = LATITUDE ~ ., 
  data      = combined_data_lat, 
)
# Combined data with PCA ---------

combined_pca <-  dplyr::select(combined_data1, -c(LONGITUDE, LATITUDE, building_floor, BUILDINGID))

#principal component analysis
prin_comp_com <- prcomp(combined_pca, scale. = T)
names(prin_comp_com)

biplot(prin_comp_com, scale = 0)

#compute standard deviation of each principal component
std_dev_com <- prin_comp_com$sdev

#compute variance
pr_var_com <- std_dev_com^2

#check variance of first 10 components
pr_var_com[1:10]

#proportion of variance explained
prop_varex_com <- pr_var_com/sum(pr_var_com)
prop_varex_com[1:20]

#scree plot
plot(prop_varex_com, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", type = "b")

#add a training set with principal components
combined.pca.ID <- data.frame(BUILDINGID = combined_data1$BUILDINGID, prin_comp_com$x)
combined.pca.bg <- data.frame(building_floor = combined_data1$building_floor, prin_comp_com$x)
combined.pca.lon <- data.frame(LONGITUDE = combined_data1$LONGITUDE, prin_comp_com$x)
combined.pca.lat <- data.frame(LATITUDE = combined_data1$LATITUDE, prin_comp_com$x)

#we are interested in first 455 PCAs
combined.pca.ID <- combined.pca.ID[,1:456]
combined.pca.bg <- combined.pca.bg[,1:456]
combined.pca.lon <- combined.pca.lon[,1:456]
combined.pca.lat <- combined.pca.lat[,1:456]

#transform realdata into PCA
realdata_pca <-  dplyr::select(realdata1, -c(LONGITUDE, LATITUDE, building_floor, BUILDINGID))
realdata_pca <- predict(prin_comp_com, newdata = realdata_pca)
realdata_pca <- as.data.frame(realdata_pca)

#select the first 455 components
realdata_pca <- realdata_pca[,1:455]



#Model for PCA's combined in Ranger --------

#Model for building_floor 
bf_ranger_pca_com <- ranger(formula= building_floor ~ .,data= combined.pca.bg, seed = 12345)

#Model for longitude
lon_ranger_pca_com <- ranger(formula= LONGITUDE ~ ., data= combined.pca.lon, seed = 12345)

#Model for latitude
lat_ranger_pca_com <- ranger(formula= LATITUDE ~ ., data= combined.pca.lat, seed = 12345)




# Models on combinated data and boruta -------
#Model for building
BUILDINGID_ranger_com <- ranger(
  formula   = BUILDINGID ~ ., 
  data      = combined_data_ID_1, 
)

#Model for building_floor
bf_ranger_com <- ranger(
  formula   = building_floor ~ ., 
  data      = combined_data_bf_1, 
)

#Model for longitude
lon_ranger_com <- ranger(
  formula   = LONGITUDE ~ ., 
  data      = combined_data_lon_1, 
)

#Model for latitude
lat_ranger_com <- ranger(
  formula   = LATITUDE ~ ., 
  data      = combined_data_lat_1, 
)
# Models on combinated data and h2o ------

combined_data_ID_h2o_only <- as.h2o(combined_data_ID)
combined_data_bf_h2o_only <- as.h2o(combined_data_bf)
combined_data_lon_h2o_only <- as.h2o(combined_data_lon)
combined_data_lat_h2o_only <- as.h2o(combined_data_lat)

ID <- "BUILDINGID"
bf <- "building_floor"
lon <- "LONGITUDE"
lat <- "LATITUDE"

fit_bf_h2o_com <- h2o.automl(y = bf, training_frame = combined_data_bf_h2o_only,
                                    max_runtime_secs = 300,
                                    nfolds = 10, seed = 12345,
                                    project_name = "bf_h2o_com")

fit_lon_h2o_com <- h2o.automl(y = lon, training_frame = combined_data_lon_h2o_only,
                                     max_runtime_secs = 300,
                                     nfolds = 10, seed = 12345,
                                     project_name = "lon_h2o_com")

fit_lat_h2o_com <- h2o.automl(y = lat, training_frame = combined_data_lat_h2o_only,
                                     max_runtime_secs = 300,
                                     nfolds = 10, seed = 12345,
                                     project_name = "lat_h2o_com")

# Models on combinated data and boruta and h2o ------

combined_data_ID_h2o <- as.h2o(combined_data_ID_1)
combined_data_bf_h2o <- as.h2o(combined_data_bf_1)
combined_data_lon_h2o <- as.h2o(combined_data_lon_1)
combined_data_lat_h2o <- as.h2o(combined_data_lat_1)

ID <- "BUILDINGID"
bf <- "building_floor"
lon <- "LONGITUDE"
lat <- "LATITUDE"

fit_bf_h2o_com_boruta <- h2o.automl(y = bf, training_frame = combined_data_bf_h2o,
                                    max_runtime_secs = 300,
                                    nfolds = 10, seed = 12345,
                                    project_name = "bf_h2o_com")

fit_lon_h2o_com_boruta <- h2o.automl(y = lon, training_frame = combined_data_lon_h2o,
                                    max_runtime_secs = 300,
                                    nfolds = 10, seed = 12345,
                                    project_name = "lon_h2o_com")

fit_lat_h2o_com_boruta <- h2o.automl(y = lat, training_frame = combined_data_lat_h2o,
                                    max_runtime_secs = 300,
                                    nfolds = 10, seed = 12345,
                                    project_name = "lat_h2o_com")

# predictions real data ------
# predictions with only combined data
pred_ID_ranger_com_real <- predict(BUILDINGID_ranger_real, realdata1)
pred_bf_ranger_com_real <- predict(bf_ranger_real, realdata1)
pred_lon_ranger_com_real <- predict(lon_ranger_real, realdata1)
pred_lat_ranger_com_real <- predict(lat_ranger_real, realdata1)

# predictions with boruta and combined data
pred_ID_ranger_com <- predict(BUILDINGID_ranger_com, realdata1)
pred_bf_ranger_com <- predict(bf_ranger_com, realdata1)
pred_lon_ranger_com <- predict(lon_ranger_com, realdata1)
pred_lat_ranger_com <- predict(lat_ranger_com, realdata1)

# predictions with combined data and h2o
realdata_h2o <- as.h2o(realdata1)
pred_bf_com_h2o_real <- h2o.predict(fit_bf_h2o_com, realdata_h2o)
pred_lon_com_h2o_real <- h2o.predict(fit_lon_h2o_com, realdata_h2o)
pred_lat_com_h2o_real <- h2o.predict(fit_lat_h2o_com, realdata_h2o)

# predictions with boruta and combined data and h2o
realdata_h2o <- as.h2o(realdata1)
pred_bf_com_h2o <- h2o.predict(fit_bf_h2o_com_boruta, realdata_h2o)
pred_lon_com_h2o <- h2o.predict(fit_lon_h2o_com_boruta, realdata_h2o)
pred_lat_com_h2o <- h2o.predict(fit_lat_h2o_com_boruta, realdata_h2o)

# predictions with PCA and combined data
pred_bf_pca_com <- predict(bf_ranger_pca_com, realdata_pca)
pred_lon_pca_com <- predict(lon_ranger_pca_com, realdata_pca)
pred_lat_pca_com <- predict(lat_ranger_pca_com, realdata_pca)

#results ------
#submission_1 combined data only
submission_combined <- as.data.frame(pred_bf_ranger_com_real$predictions)
submission_combined$LATITUDE <- pred_lat_ranger_com_real$predictions
submission_combined$LONGITUDE <- pred_lon_ranger_com_real$predictions
write.csv(submission_combined, "../CVJ_submission_combined.csv")

#submission_2 combined + boruta
submission_combined_boruta <- as.data.frame(pred_bf_ranger_com$predictions)
submission_combined_boruta$LATITUDE <- pred_lat_ranger_com$predictions
submission_combined_boruta$LONGITUDE <- pred_lon_ranger_com$predictions
write.csv(submission_combined_boruta, "../CVJ_submission_combined_boruta.csv")

#submission_3 combined h2o + boruta
submission_h2o <- pred_bf_com_h2o$pred
submission_h2o$LATITUDE <- pred_lat_com_h2o$pred
submission_h2o$LONGITUDE <- pred_lon_com_h2o$pred
h2o.exportFile(submission_h2o, "../CVJ_submission_combined_boruta_h2o.csv")

#submission_4 combined h2o
submission_h2o_only <- pred_bf_com_h2o_real$pred
submission_h2o_only$LATITUDE <- pred_lat_com_h2o_real$pred
submission_h2o_only$LONGITUDE <- pred_lon_com_h2o_real$pred
h2o.exportFile(submission_h2o_only, "../CVJ_submission_combined_h2o.csv")

#submission_5 combined PCA
submission_combined_pca <- as.data.frame(pred_bf_pca_com$predictions)
submission_combined_pca$LATITUDE <- pred_lat_pca_com$predictions
submission_combined_pca$LONGITUDE <- pred_lon_pca_com$predictions
write.csv(submission_combined_boruta, "../CVJ_submission_combined_pca.csv")




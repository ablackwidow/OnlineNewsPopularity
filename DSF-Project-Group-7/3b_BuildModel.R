
# --------------- DEFINING PARAMETERS FOR tuning_run FUNCTION (file "NeuralNetTens") ---------------


#THIS FILE WAS ONLY USED TO DEFINE THE RUNS PARAMETERS IN PARAMETERS TUNING


#Define flags
Flags <- flags(flag_integer("dense_units", 32),
              flag_integer("dense_units2", 32),
              flag_numeric("dropout1", 0.1),
              flag_numeric("dropout2", 0.1),
              flag_integer("batch_size", 32),
              flag_integer("epoch", 50))

neurnetTune <- keras_model_sequential()

# add layers and compile the model
neurnetTune %>%
  layer_dense(units = Flags$dense_units , activation = 'relu', input_shape = c(totalCol-1)) %>%
  layer_dropout(rate = Flags$dropout1)%>% #avoid overfitting 
  layer_dense(units = Flags$dense_units2, activation = 'relu', input_shape = c(totalCol-1))%>%
  layer_dropout(rate = Flags$dropout2)%>%
  layer_dense(units = 2, activation = 'softmax')
summary(neurnetTune)

neurnetTune %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = c("accuracy", tf$keras$metrics$AUC()))

#fit model
history <- neurnetTune %>%
  fit(news_train, 
      trainLabels, 
      epoch = Flags$epoch, 
      batch_size = Flags$batch_size,
      validation_split = 0.2)




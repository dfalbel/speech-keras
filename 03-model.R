library(keras)
library(dplyr)
source("02-generator.R")

df <- readRDS("data/df.rds") %>% sample_frac(1)
id_train <- sample(nrow(df), size = 0.7*nrow(df))

ds_train <- data_generator(df[id_train,], 32L)
ds_test <- data_generator(df[-id_train,], 32, shuffle = FALSE)


model <- keras_model_sequential()
model %>%  
  layer_conv_2d(input_shape = c(98, 257, 1), 
                filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 30, activation = 'softmax')

# Compile model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
model %>% fit_generator(
  generator = ds_train,
  steps_per_epoch = 0.7*nrow(df)/32,
  epochs = 10, 
  validation_data = ds_test, 
  validation_steps = 0.3*nrow(df)/32
)

save_model_hdf5(model, filepath = "model.hdf5")

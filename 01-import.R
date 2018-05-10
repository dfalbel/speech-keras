library(stringr)
library(dplyr)

files <- fs::dir_ls(
  path = "data/speech_commands_v0.01/", 
  recursive = TRUE, 
  glob = "*.wav"
)

files <- files[!str_detect(files, "background_noise")]

df <- data_frame(
  fname = files, 
  class = fname %>% str_extract("1/.*/") %>% 
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)

library(tfdatasets)
audio_ops <- tf$contrib$framework$python$ops$audio_ops

ds <- tfdatasets::tensor_slices_dataset(df)
ds <- ds %>%
  dataset_map(function(obs) {
    
    # decoding wav files
    audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
    wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
    
    # create the spectogram
    spectrogram <- audio_ops$audio_spectrogram(
      wav$audio,
      window_size = 30,
      stride = 20,
      magnitude_squared = TRUE
    )
    
    # create the  Mel-Frequency Cepstral Coefficients 
    x <- audio_ops$mfcc(
      spectrogram,
      wav$sample_rate,
      dct_coefficient_count = 40L
    )
    
    # transform the class_id into a one-hot encoded vector
    response <- tf$one_hot(obs$class_id, 29L)
    
    list(x, response)
  }) %>%
  dataset_shuffle(buffer_size = 100) %>%
  dataset_repeat() %>%
  #dataset_batch(32)
  #dataset_padded_batch(32, list(shape(16000), shape(NULL)))
  
  
  
  batch <- next_batch(ds)
sess <- tf$Session()
k <- sess$run(batch)

# Creates a generator from a dataset.

library(tfdatasets)
audio_ops <- tf$contrib$framework$python$ops$audio_ops

data_generator <- function(df, batch_size, shuffle = TRUE, 
                           window_size_ms = 30, window_stride_ms = 10) {
  
  ds <- tensor_slices_dataset(df) %>%
    dataset_map(function(obs) {
      
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
      
      # create the spectrogram
      spectrogram <- audio_ops$audio_spectrogram(
        wav$audio,
        window_size = as.integer(16000*window_size_ms/1000),
        stride = as.integer(16000*window_stride_ms/1000),
        magnitude_squared = TRUE
      )
      
      # create the  Mel-Frequency Cepstral Coefficients
      # spectrogram <- audio_ops$mfcc(
      #   spectrogram,
      #   wav$sample_rate,
      #   dct_coefficient_count = 40L
      # )
      
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 29L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 100)  
  
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(98, 257, NULL), shape(NULL)))
  
  ds
}


create_spectrogram <- function(x) {
  
  audio_binary <- tf$read_file(tf$reshape(x, shape = list()))
  wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
  
  spectrogram <- audio_ops$audio_spectrogram(
    wav$audio,
    window_size = 480,
    stride = 160,
    magnitude_squared = TRUE
  )
  
  spectrogram
}

# s <- create_spectrogram("data/speech_commands_v0.01/bed/fffcabd1_nohash_1.wav")
# x <- audio_ops$mfcc(s, sample_rate =  16000L, dct_coefficient_count = 40)
# 
# sess <- tf$Session()
# a <- sess$run(list(x, s))
# 
# a <- a[[1]]
# a <- (a  - min(a))/(max(a) - min(a))
# plot(as.raster(a[1,,]))




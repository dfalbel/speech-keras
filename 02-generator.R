# Creates a generator from a dataset.

library(tfdatasets)
audio_ops <- tf$contrib$framework$python$ops$audio_ops

data_generator <- function(df, batch_size, shuffle = TRUE) {
  
  ds <- tensor_slices_dataset(df) %>%
    dataset_map(function(obs) {
      
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
      
      # create the spectrogram
      spectrogram <- audio_ops$audio_spectrogram(
        wav$audio,
        window_size = 16000/50,
        stride = 16000/75,
        magnitude_squared = TRUE
      )
      
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
      
      # create the  Mel-Frequency Cepstral Coefficients
      # x <- audio_ops$mfcc(
      #   spectrogram,
      #   wav$sample_rate,
      #   dct_coefficient_count = 40L
      # )
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 29L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 100)  
  
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(74, 257, NULL), shape(NULL)))
  
  ds
}




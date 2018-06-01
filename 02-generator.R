# Creates a generator from a dataset.

library(tfdatasets)

audio_ops <- tf$contrib$framework$python$ops$audio_ops
signal <- tf$contrib$signal 

data_generator <- function(df, batch_size, shuffle = TRUE, 
                           frame_length = 480L, frame_step = 160L) {
  
  ds <- tensor_slices_dataset(df) 
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(1000)  
  
  # fft lenght
  # by default it's the smallest power of 2 enclosing frame_length.
  fft_length <- as.integer(2^(as.integer(log(frame_length, 2)) + 1L))
  
  ds <- ds %>%
    dataset_map(function(obs) {
      
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
      
      # create the spectrogram
      spectrogram <- signal$stft(
        wav$audio[,1], 
        frame_length = frame_length, 
        frame_step = frame_step, 
        pad_end = TRUE, 
        fft_length = fft_length
      )
      
      spectrogram <- tf$real(spectrogram * tf$conj(spectrogram))
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$expand_dims(spectrogram, -1L)
      
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 29L)
      
      list(spectrogram, response)
    }) 
  
  
  ds <- ds %>% 
    dataset_repeat() %>%
    dataset_padded_batch(
      batch_size, 
      list(
        shape(as.integer(16000/frame_step), as.integer(fft_length/2 + 1), NULL), 
        shape(NULL)
      )
    )
  
  ds
}


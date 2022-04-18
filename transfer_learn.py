# Utility function for loading audio files and making sure the sample rate is correct.
@tf.function
def load_wav_16k_mono(filename):
      """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio, and frame it to 15600 samples. """
      file_contents = tf.io.read_file(filename)
      wav, sample_rate = tf.audio.decode_wav(
            file_contents, 
            desired_channels=1
      )
      wav = tf.squeeze(wav, axis=-1)
      sample_rate = tf.cast(sample_rate, dtype=tf.int64)
      wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
      return wav
      # print(wav.shape)
      frames = tf.signal.frame(wav, 15600, 15600)
      # print(frames.shape)
      return frames

@tf.function
def frame_16k_mono(filename):
      wav = load_wav_16k_mono(filename)
      frames = tf.signal.frame(wav, 15600, 15600)
      return frames




def main():
    # read in esc-50 descriptive data
    esc50_csv = './datasets/ESC-50-master/meta/esc50.csv'
    base_data_path = './datasets/ESC-50-master/audio/'

    pd_data = pd.read_csv(esc50_csv)
    pd_data.head()


    # filter descriptive data to cat and dog class
    my_classes = ['dog', 'cat']
    map_class_to_id = {'dog':0, 'cat':1}

    filtered_pd = pd_data[pd_data.category.isin(my_classes)]

    class_id = filtered_pd['category'].apply(lambda name: map_class_to_id[name])
    filtered_pd = filtered_pd.assign(target=class_id)

    full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
    filtered_pd = filtered_pd.assign(filename=full_path)

    filtered_pd.head(10)


    # create dataset from the descriptive data
    filenames = filtered_pd['filename']
    targets = filtered_pd['target']
    folds = filtered_pd['fold']

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))


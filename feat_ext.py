
import numpy as np
import scipy
import librosa
import soundfile


#########################################################################
# Some of these functions have been inspired on the DCASE UTIL framework by Toni Heittola
# https://dcase-repo.github.io/dcase_util/
#########################################################################


def load_audio_file(file_path, input_fixed_length=0, params_extract=None):
    """

    :param file_path:
    :param input_fixed_length:
    :param params_extract:
    :return:
    """
    data, source_fs = soundfile.read(file=file_path)
    data = data.T

    # Resample if the source_fs is different from expected
    if params_extract.get('fs') != source_fs:
        data = librosa.core.resample(data, source_fs, params_extract.get('fs'))
        print('Resampling to %d: %s' % (params_extract.get('fs'), file_path))

    if len(data) > 0:
        data = get_normalized_audio(data)
    else:
        # 3 files are corrupted in the test set. They belong to the padding group (not used for evaluation)
        data = np.ones((input_fixed_length, 1))
        print('File corrupted. Could not open: %s' % file_path)

    # careful with the shape
    data = np.reshape(data, [-1, 1])
    return data


def modify_file_variable_length(data=None, input_fixed_length=0, params_extract=None):
    """

    :param data:
    :param input_fixed_length:
    :param params_extract:
    :return:
    """

    if params_extract.get('load_mode') == 'varup':
        # 获取关键参数
        n_fft = int(params_extract['n_fft'])
        hop_length = int(params_extract['hop_length_samples'])
        num_frames = len(data)
        
        # print(n_fft)
        # print(hop_length)
        # print(num_frames)
        # 计算当前能产生的时间步数
        time_steps = (num_frames - n_fft) // hop_length + 1

        # deal with short sounds
        num_frames = len(data)

        # 计算最小所需音频长度：确保至少能生成1个时间步
        # min_audio_length = params_extract['patch_len'] * params_extract['hop_length_samples']  # 假设n_fft为2048

        target_time_steps = 150

         # 计算目标音频长度
        target_length2 = (target_time_steps - 1) * hop_length + n_fft
        target_length1 = (100 - 1) * hop_length + n_fft
        # print(f'num_frames:{num_frames}, target_length1:{target_length1}, target_length2:{target_length2}')

        if num_frames < target_length1:
            # 计算需要复制的次数
            num_repeats = int(np.ceil(target_length2 / num_frames))
            extended_data = np.tile(data, (num_repeats, 1))
            
            # 截取到目标长度
            data = extended_data[:target_length2]
            # print(f"[Replicate] {num_frames} -> {target_length2} samples ({target_time_steps} steps)")
        
            # if file shorter than input_length, replicate the sound to reach the input_fixed_length
            # nb_replicas = int(np.ceil(input_fixed_length / num_frames))
            # # replicate according to column
            # data_rep = np.tile(data, (nb_replicas, 1))
            # data = data_rep[:2 * input_fixed_length]

            # print(f"Data padded to shape: {data.shape}")  # 打印补齐后的形状

        elif num_frames >= input_fixed_length:
         # 如果帧数超出，裁剪到 input_fixed_length 的整数倍
            num_patches = num_frames // input_fixed_length  # 计算可以生成的完整补丁数量
            data = data[:num_patches * input_fixed_length]  # 裁剪到整数倍长度
            # print(f"Data trimmed to shape: {data.shape}")  # 打印裁剪后的形状

    return data


def get_normalized_audio(y, head_room=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value


def get_mel_spectrogram(audio, params_extract=None):
    """

    :param audio:
    :param params_extract:
    :return:
    """

    # make sure rows are channels and columns the samples
    audio = audio.reshape([1, -1])
    window = scipy.signal.hamming(params_extract.get('win_length_samples'), sym=False)

    mel_basis = librosa.filters.mel(sr=params_extract.get('fs'),
                                    n_fft=params_extract.get('n_fft'),
                                    n_mels=params_extract.get('n_mels'),
                                    fmin=params_extract.get('fmin'),
                                    fmax=params_extract.get('fmax'),
                                    htk=False,
                                    norm=None)

    # init mel_spectrogram expressed as features: row x col = frames x mel_bands = 0 x mel_bands (to concatenate axis=0)
    feature_matrix = np.empty((0, params_extract.get('n_mels')))
    for channel in range(0, audio.shape[0]):
        spectrogram = get_spectrogram(
            y=audio[channel, :],
            n_fft=params_extract.get('n_fft'),
            win_length_samples=params_extract.get('win_length_samples'),
            hop_length_samples=params_extract.get('hop_length_samples'),
            spectrogram_type=params_extract.get('spectrogram_type') if 'spectrogram_type' in params_extract else 'magnitude',
            center=True,
            window=window,
            params_extract=params_extract
        )

        mel_spectrogram = np.dot(mel_basis, spectrogram)
        mel_spectrogram = mel_spectrogram.T

        if params_extract.get('log'):
            mel_spectrogram = np.log10(mel_spectrogram + params_extract.get('eps'))

        feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)

    return feature_matrix


def get_spectrogram(y,
                    n_fft=1024,
                    win_length_samples=0.04,
                    hop_length_samples=0.02,
                    window=scipy.signal.hamming(1024, sym=False),
                    center=True,
                    spectrogram_type='magnitude',
                    params_extract=None):

    if spectrogram_type == 'power':
        return np.abs(librosa.stft(y + params_extract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window)) ** 2

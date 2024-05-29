import numpy as np
import soundfile as sf

# 定义要拼接的文件列表
file_list = ['tests/data/asr_example_zh.wav', 'tests/data/asr_example_粤语.wav', 'tests/data/asr_example_jp.wav', 'tests/data/asr_example_en.wav']

# 初始化音频数据和采样率列表
audio_data_list = []
samplerates = []

# 读取所有文件
for file_path in file_list:
    data, samplerate = sf.read(file_path)
    audio_data_list.append(data)
    samplerates.append(samplerate)

# 检查所有文件的采样率是否相同
if not all(s == samplerates[0] for s in samplerates):
    print("Error: Not all files have the same sample rate.")
    exit()

# 拼接音频数据
concatenated_data = np.concatenate(audio_data_list, axis=0)

# 写入新的WAV文件
output_file = 'tests/data/output_concatenate.wav'
sf.write(output_file, concatenated_data, samplerates[0])
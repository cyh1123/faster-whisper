import time

from faster_whisper import WhisperModel

# model_size = 'models/faster-whisper-medium'
model_size = "models/faster-whisper-large-v3"


# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

start = time.time()
# segments, info = model.transcribe("tests/data/asr_example_zh.wav", beam_size=5)
segments, info = model.transcribe("tests/data/output_concatenate_zh_yue_jp_en.wav", beam_size=5)
end = time.time()

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print(f"Time: {end - start} s")
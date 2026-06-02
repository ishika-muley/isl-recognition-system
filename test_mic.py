import sounddevice as sd
import numpy as np

print("Devices:")
print(sd.query_devices())
print("\nDefault device:", sd.default.device)

print("\nRecording 2 seconds...")
audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype='int16')
sd.wait()
print(f"Max amplitude: {np.max(np.abs(audio))}")

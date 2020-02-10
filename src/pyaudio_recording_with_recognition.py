import pyaudio
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import datetime
from matplotlib.animation import FuncAnimation
import threading
from scipy.io import wavfile
import os
import librosa
from pydub import AudioSegment
from pydub.playback import play

import keyboard

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import signal as scipy_signal

import glob

RATE = 16000
CHUNK = int(RATE/20) # RATE / number of updates per second
voice_activated = False
file_name = None
file_names = list()

song = AudioSegment.from_wav("chime.wav")
song = song - 20

answer_list = [' ', '_hey_liku', 'bed', 'bird', 'cat', 'dog', \
			   'down', 'eight', 'five', 'four', 'go', 'happy', 
			   'house', 'left', 'marvin', 'nine', 'no', 'off', \
			   'on', 'one', 'right', 'seven', 'sheila', 'six', \
			   'stop', 'three', 'tree', 'two', 'up', 'wow', \
			   'yes', 'zero']

class basic_rnn_model(nn.Module):
#     def __init__(self, first_kernel_size, second_kernel_size, dropout_rate, num_labels):
	def __init__(self):
		super(basic_rnn_model, self).__init__()
		
		self.gru_1 = nn.GRU(321, 512, 1, bidirectional=False)
		self.gru_2 = nn.GRU(512, 512, 1, bidirectional=False)
		self.gru_3 = nn.GRU(512, 512, 1, bidirectional=False)
		self.gru_4 = nn.GRU(512, 512, 1, bidirectional=False)

		self.flatten = nn.Flatten()
		self.fc = nn.Linear(512, 32)
		
	def forward(self, input_tensor):
		
		# (B, F, T)
		input_tensor.transpose_(0, 2) # (T, F, B)
		input_tensor.transpose_(1, 2) # (T, B, F)
		tensor, _ = self.gru_1(input_tensor)
		
		tensor = F.relu(tensor)
		tensor = F.dropout(tensor, 0.1, training=True)
		tensor, _ = self.gru_2(tensor)

		tensor = F.relu(tensor)
		tensor = F.dropout(tensor, 0.1, training=True)
		tensor, _ = self.gru_3(tensor)
		
		tensor = F.relu(tensor)
		tensor = F.dropout(tensor, 0.1, training=True)
		tensor, hidden = self.gru_4(tensor) # Hidden (2[Direction], B, H)

		tensor = F.relu(hidden)
		tensor = F.dropout(tensor, 0.1, training=True)
		
		tensor.transpose_(0, 1) # (B, T, F)
		tensor = self.flatten(tensor)
		tensor = self.fc(tensor)
		pred_tensor = F.log_softmax(tensor, dim=-1)
		
		return pred_tensor

def print_pressed_keys(e):
	for code in keyboard._pressed_events:
		line = ', '.join(str(code))
		print(line,e.name)
		
def delete_last_file(e):
	global file_name
	global file_names
	
#     if e.name == 'delete':
	if e.name == '`':
		if file_name is not None:
			os.remove(file_name)
			print('[Deleted: {}]'.format(file_name))
			if len(file_names) > 0:
				file_name = file_names.pop()
			else:
				file_name = None 
#     elif e.name == 'p':
	elif e.name == '=':
		if file_name is not None:
			record = AudioSegment.from_wav(file_name)
			play(record)
			print('[Played: {}]'.format(file_name))
#     else:
#         print(e.name)
		

# keyboard.hook(delete_last_file)
keyboard.on_press(delete_last_file)

def audio_streaming_thread(stream):
# 	for i in range(1000):
	while True:
		global plotdata
# 		t1=time.time()
		data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
# 		data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
		# data = data / 2 ** 16
		# print(data)
		shift = len(data)
		plotdata = np.roll(plotdata, -shift, axis=0)
		plotdata[-shift:] = data
# 		now = datetime.datetime.now()
# 		if i != 0:
# 			print('[Interval: {} Chunk: {}]'.format(now - last, len(data)))
# 		last = datetime.datetime.now()

# 		if i == 50:
# 			print('PyAudio Wave File Saved.')
# 			wavfile.write(os.path.join('record', 'pa_test.wav'), 16000, plotdata)


def update_plot(frame):
	global plotdata
	global voice_activated
	global song
	
	lines[0].set_ydata(plotdata[::160]/2**15)

	S = librosa.feature.melspectrogram(y=plotdata/2**15, sr=16000, n_mels=256)
	S_dB = librosa.power_to_db(S, ref=np.max)
	img.set_data(S_dB)
	
# 	lines[0].set_ydata(plotdata[::10]/2**15)

	lines2[0].set_ydata(np.mean(S, axis=0))

#         lines3[0].set_ydata(np.mean(S, axis=0) > 0.2)

	
	conv_array = np.ones(10)/10
	VAD = np.ceil(np.convolve(np.mean(S, axis=0) > 0.05, conv_array, 'same'))
	
	lines3[0].set_ydata(VAD)
	
	if VAD[-1] > 0:
		voice_activated = True
		
	if voice_activated and VAD[-1] < 1:
		revVAD = VAD[::-1]
		for i, val in enumerate(revVAD):
			if i < 5:
				continue
			if val <= 0 and revVAD[i-1] <= 0 and revVAD[i-2] <= 0 and revVAD[i-3] <= 0 and revVAD[i-4] <= 0:
				idx = i
				voice_activated = False
				break
 
		if idx > 15 and idx < 50:
			global file_name
			global file_names
			
			file_names.append(file_name)
			file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.wav")
			file_name = 'jh_' + file_name         
			print('[VAD: {}/{} | Saved {}]'.format(idx, len(VAD), file_name))
			file_name = os.path.join('record', file_name)
			recording_array = plotdata[-16000-2000:-2000].astype(np.int16)
			wavfile.write(file_name, 16000, recording_array)
			play(song)
			
			sr = 16000
			nsc_in_ms = 40
			nov_in_ms = 0
			nsc_in_sample = int(nsc_in_ms / 1000 * sr)
			nov_in_sample = int(nov_in_ms / 1000 * sr)
			
			f, t, Zxx = scipy_signal.stft(recording_array / 2** 15, fs=sr, 
							  nperseg=nsc_in_sample,
							  noverlap=nov_in_sample)

			Sxx = np.abs(Zxx)
			normalized_spectrogram = (20 * np.log10(np.maximum(Sxx, 1e-8)) + 160) / 160
			normalized_spectrogram = normalized_spectrogram.astype(np.float32)
			single_batch = torch.tensor(np.expand_dims(normalized_spectrogram, 0))
			pred_tensor = model(single_batch)
			pred_array = pred_tensor.detach().numpy().T
			idx = np.argmax(pred_array)
			print('[Predicted: {}]'.format(answer_list[idx]))

		elif idx >= 50:
			print('[VAD: {}/{} | Not saved]'.format(idx, len(VAD)))    

	return [img] + lines + lines2 + lines3

if __name__=="__main__":
	
	model = basic_rnn_model()
	print(glob.glob('notebooks/models/*'))
	
	selection = int(input('[Type in your model number]\n'))
	model_path = glob.glob('notebooks/models/*')[selection]
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	
	p=pyaudio.PyAudio()
	stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
				  frames_per_buffer=CHUNK)
# 	stream=p.open(format=pyaudio.paFloat32,channels=1,rate=RATE,input=True,
# 				  frames_per_buffer=CHUNK)

	plotdata = np.zeros([16000 * 10])


	# fig, ax = plt.subplots()
	# lines = ax.plot(plotdata[::10])
	# ax.axis((0, len(plotdata[::10]), -1, 1))
	# ax.set_yticks([0])
	# ax.yaxis.grid(True)
	# ax.tick_params(bottom=False, top=False, labelbottom=False,
	# 			   right=False, left=False, labelleft=False)
	# fig.tight_layout(pad=0)

	dummy_specgram = librosa.feature.melspectrogram(y=plotdata, sr=16000, n_mels=256)

	fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8))
	img = axes[0].imshow(dummy_specgram, origin='reversed', aspect='auto')
	img.set_clim([-100, 0])
#     fig.colorbar(img, ax=axes[0])
#     lines = ax.plot(plotdata)
#     if len(args.channels) > 1:
#         ax.legend(['channel {}'.format(c) for c in args.channels],
#                   loc='lower left', ncol=len(args.channels))
#     ax.axis((0, len(plotdata), -1, 1))
	axes[0].set_yticks([0])
	axes[0].yaxis.grid(True)
	axes[0].tick_params(bottom=False, top=False, labelbottom=False,
				   right=False, left=False, labelleft=False)
	
	###
	
#     print('[Dummy Specgram: {}]'.format(dummy_specgram.shape))
#     print('[Dummy Specgram Sum: {}]'.format(len(np.sum(dummy_specgram, axis=0))))
	
#     lines = axes[1].plot(np.sum(dummy_specgram))
	waveform = plotdata[::10]
	lines = axes[1].plot(plotdata[::160])
	axes[1].set_ylim([-.5, .5])
	axes[1].set_xlim([0, int(len(plotdata)/160)])
	axes[1].yaxis.grid(True)
	axes[1].tick_params(bottom=False, top=False, labelbottom=False,
				   right=False, left=False, labelleft=False)
	
	lines2 = axes[2].plot(np.sum(dummy_specgram, axis=0))
#     lines2 = axes[2].plot(plotdata[::10])
	axes[2].set_ylim([0, 5])
#     axes[2].set_xlim([0, int(length/10)])
	axes[2].set_xlim([0, dummy_specgram.shape[1]])
	axes[2].yaxis.grid(True)
#     axes[2].tick_params(bottom=False, top=False, labelbottom=False,
#                    right=False, left=False, labelleft=False)

	lines3 = axes[3].plot(np.sum(dummy_specgram, axis=0))
#     lines2 = axes[2].plot(plotdata[::10])
	axes[3].set_ylim([0, 2])
#     axes[2].set_xlim([0, int(length/10)])
	axes[3].set_xlim([0, dummy_specgram.shape[1]])
	axes[3].yaxis.grid(True)

	ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

	t = threading.Thread(target=audio_streaming_thread, args=(stream,))
	t.start()

	# for i in range(sys.maxsize**10):
	#     soundplot(stream)
	#     plt.show()

	plt.show()



	stream.stop_stream()
	stream.close()
	p.terminate()
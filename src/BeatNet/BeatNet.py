# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>


# This is the script handler of the BeatNet. First, it extracts the input embeddings of the current frame or the whole song, depending on the working mode. 
# Then by feeding them into the selected pre-trained model, it calculates the beat/downbeat activation probabilities.
# Finally, it infers beats and downbeats of the current frame/song based on one of the four performance modes and selected inference method.

import os
import torch
import numpy as np
from madmom.features import DBNDownBeatTrackingProcessor
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from BeatNet.log_spect import LOG_SPECT
import librosa
import sys
from BeatNet.model import BDA
import pyaudio
import matplotlib.pyplot as plt
import time
import threading


class BeatNet:

    '''
    The main BeatNet handler class including different trained models, different modes for extracting the activation and causal and non-causal inferences

        Parameters
        ----------
        Inputs: 
            model: An scalar in the range [1,3] to select which pre-trained CRNN models to utilize. 
            mode: An string to determine the working mode. i.e. 'stream', 'realtime', 'online' and ''offline.
                'stream' mode: Uses the system microphone to capture sound and does the process in real-time. Due to training the model on standard mastered songs, it is highly recommended to make sure the microphone sound is as loud as possible. Less reverbrations leads to the better results.  
                'Realtime' mode: Reads an audio file chunk by chunk, and processes each chunck at the time.
                'Online' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then infers the parameters on interest using particle filtering.
                'offline' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then inferes the parameters on interest using madmom dynamic Bayesian network. This method is quicker that madmom beat/downbeat tracking.
            inference model: A string to choose the inference approach. i.e. 'PF' standing for Particle Filtering for causal inferences and 'DBN' standing for Dynamic Bayesian Network for non-causal usages.
            plot: A list of strings to plot. 
                'activations': Plots the neural network activations for beats and downbeats of each time frame. 
                'beat_particles': Plots beat/tempo tracking state space and current particle states at each time frame.
                'downbeat_particles': Plots the downbeat/meter tracking state space and current particle states at each time frame.
                Note that to speedup plotting the figures, rather than new plots per frame, the previous plots get updated. However, to secure realtime results, it is recommended to not plot or have as less number of plots as possible at the time.   
            threading: To decide whether accomplish the inference at the main thread or another thread. 
            device: type of dvice. cpu or cuda:i

        Outputs:
            A vector including beat times and downbeat identifier columns, respectively with the following shape: numpy_array(num_beats, 2).
    '''
    
    
    def __init__(self, model, mode='online', inference_model='PF', plot=[], thread=False, device='cpu', input_device_index=None):
        self.model = model
        self.mode = mode
        self.inference_model = inference_model
        self.plot= plot
        self.thread = thread
        self.device = device
        if plot and thread:
            raise RuntimeError('Plotting cannot be accomplished in the threading mode')
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)
        if self.inference_model == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=self.plot, mode=self.mode)
        elif self.inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        else:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')
        script_dir = os.path.dirname(__file__)
        #assiging a BeatNet CRNN instance to extract joint beat and downbeat activations
        self.model = BDA(272, 150, 2, self.device)   #Beat Downbeat Activation detector
        #loading the pre-trained BeatNet CRNN weigths
        if model == 1:  # GTZAN out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_1_weights.pt')), strict=False)
        elif model == 2:  # Ballroom out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_2_weights.pt')), strict=False)
        elif model == 3:  # Rock_corpus out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_3_weights.pt')), strict=False)
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.eval()
        if self.mode == 'stream':
            self.stream_window = np.zeros(self.log_spec_win_length + 2 * self.log_spec_hop_length, dtype=np.float32)
            self._pyaudio = pyaudio.PyAudio()
            self.input_device_index = input_device_index
            # Determine device to use
            if self.input_device_index is not None:
                dev_info = self._pyaudio.get_device_info_by_index(self.input_device_index)
            else:
                dev_info = self._pyaudio.get_default_input_device_info()
                self.input_device_index = int(dev_info.get('index', 0))
            dev_name = dev_info.get('name', 'Unknown')
            dev_sr = int(dev_info.get('defaultSampleRate', self.sample_rate))
            max_in_ch = int(dev_info.get('maxInputChannels', 1))
            # Choose channels and input sample rate
            self.input_channels = 1 if max_in_ch >= 1 else max_in_ch
            self.input_sample_rate = self.sample_rate
            self.frames_per_buffer_in = self.log_spec_hop_length
            # Try to open at model sample rate first; if fails, fall back to device rate
            try:
                self.stream = self._pyaudio.open(format=pyaudio.paFloat32,
                                                 channels=self.input_channels,
                                                 rate=self.sample_rate,
                                                 input=True,
                                                 input_device_index=self.input_device_index,
                                                 frames_per_buffer=self.log_spec_hop_length)
                self.input_sample_rate = self.sample_rate
                self.frames_per_buffer_in = self.log_spec_hop_length
                print(f"BeatNet stream: using device {self.input_device_index} '{dev_name}' at {self.sample_rate} Hz, channels={self.input_channels}")
            except Exception:
                # Fallback to device default sample rate and compute corresponding frames per buffer (~20ms)
                self.input_sample_rate = dev_sr
                self.frames_per_buffer_in = max(1, int(0.02 * self.input_sample_rate))
                self.stream = self._pyaudio.open(format=pyaudio.paFloat32,
                                                 channels=self.input_channels,
                                                 rate=self.input_sample_rate,
                                                 input=True,
                                                 input_device_index=self.input_device_index,
                                                 frames_per_buffer=self.frames_per_buffer_in)
                print(f"BeatNet stream: using device {self.input_device_index} '{dev_name}' at {self.input_sample_rate} Hz (resampling to {self.sample_rate} Hz), channels={self.input_channels}")
            # Initialize streaming counters/state so external callers can use
            # activation_extractor_stream() without calling process() first.
            self.counter = 0
                                             
    def process(self, audio_path=None):   
        if self.mode == "stream":
            if self.inference_model != "PF":
                    raise RuntimeError('The infernece model should be set to "PF" for the streaming mode!')
            self.counter = 0
            while self.stream.is_active():
                self.activation_extractor_stream()  # Using BeatNet causal Neural network streaming mode to extract activations
                if self.thread:
                    x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                    x.start()
                    x.join()    
                else:
                    output = self.estimator.process(self.pred)
                self.counter += 1

                
        elif self.mode == "realtime":
            self.counter = 0
            self.completed = 0
            if self.inference_model != "PF":
                raise RuntimeError('The infernece model for the streaming mode should be set to "PF".')
            if isinstance(audio_path, str) or audio_path.all()!=None:
                while self.completed == 0:
                    self.activation_extractor_realtime(audio_path) # Using BeatNet causal Neural network realtime mode to extract activations
                    if self.thread:
                        x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                        x.start()
                        x.join()    
                    else:
                        output = self.estimator.process(self.pred)  # Using particle filtering online inference to infer beat/downbeats
                    self.counter += 1
                return output
            else:
                raise RuntimeError('An audio object or file directory is required for the realtime usage!')
        
        
        elif self.mode == "online":
            if isinstance(audio_path, str) or audio_path.all()!=None:
                preds = self.activation_extractor_online(audio_path)    # Using BeatNet causal Neural network to extract activations
            else:
                raise RuntimeError('An audio object or file directory is required for the online usage!')
            if self.inference_model == "PF":   # Particle filtering inference (causal)
                output = self.estimator.process(preds)  # Using particle filtering online inference to infer beat/downbeats
                return output
            elif self.inference_model == "DBN":    # Dynamic bayesian Network Inference (non-causal)
                output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                return output
        
        
        elif self.mode == "offline":
                if self.inference_model != "DBN":
                    raise RuntimeError('The infernece model should be set to "DBN" for the offline mode!')
                if isinstance(audio_path, str) or audio_path.all()!=None:
                    preds = self.activation_extractor_online(audio_path)    # Using BeatNet causal Neural network to extract activations
                    output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                    return output
        
                else:
                    raise RuntimeError('An audio object or file directory is required for the offline usage!')
                

    def activation_extractor_stream(self):
        # TODO: 
        ''' Streaming window
        Given the training input window's origin set to center, this streaming data formation causes 0.084 (s) delay compared to the trained model that needs to be fixed. 
        '''
        with torch.no_grad():
            # Read from device buffer size
            hop_buf = self.stream.read(self.frames_per_buffer_in, exception_on_overflow=False)
            hop = np.frombuffer(hop_buf, dtype=np.float32)
            # Downmix if multi-channel
            if self.input_channels and self.input_channels > 1 and hop.size >= self.frames_per_buffer_in * self.input_channels:
                hop = hop.reshape(-1, self.input_channels).mean(axis=1)
            # Resample if needed
            if self.input_sample_rate != self.sample_rate:
                # Target number of samples to match model hop length
                target_len = self.log_spec_hop_length
                hop = librosa.resample(hop, orig_sr=self.input_sample_rate, target_sr=self.sample_rate)
                # Pad/trim to expected hop length
                if hop.size < target_len:
                    hop = np.pad(hop, (0, target_len - hop.size), mode='constant')
                elif hop.size > target_len:
                    hop = hop[:target_len]
            # Maintain streaming window at model hop length
            self.stream_window = np.append(self.stream_window[self.log_spec_hop_length:], hop)
            if self.counter < 5:
                self.pred = np.zeros([1,2])
            else:
                feats = self.proc.process_audio(self.stream_window).T[-1]
                feats = torch.from_numpy(feats)
                feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                pred = self.model(feats)[0]
                pred = self.model.final_pred(pred)
                pred = pred.cpu().detach().numpy()
                self.pred = np.transpose(pred[:2, :])


    def activation_extractor_realtime(self, audio_path):
        with torch.no_grad():
            if self.counter==0: #loading the audio
                if isinstance(audio_path, str):
                    self.audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
                elif len(np.shape(audio_path))>1:
                    self.audio = np.mean(audio_path ,axis=1)
                else:
                    self.audio = audio_path
            if self.counter<(round(len(self.audio)/self.log_spec_hop_length)):
                if self.counter<2:
                    self.pred = np.zeros([1,2])
                else:
                    feats = self.proc.process_audio(self.audio[self.log_spec_hop_length * (self.counter-2):self.log_spec_hop_length * (self.counter) + self.log_spec_win_length]).T[-1]
                    feats = torch.from_numpy(feats)
                    feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                    pred = self.model(feats)[0]
                    pred = self.model.final_pred(pred)
                    pred = pred.cpu().detach().numpy()
                    self.pred = np.transpose(pred[:2, :])
            else:
                self.completed = 1


    def activation_extractor_online(self, audio_path):
        with torch.no_grad():
            if isinstance(audio_path, str):
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
            elif len(np.shape(audio_path))>1:
                audio = np.mean(audio_path ,axis=1)
            else:
                audio = audio_path
            feats = self.proc.process_audio(audio).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0).to(self.device)
            preds = self.model(feats)[0]  # extracting the activations by passing the feature through the NN
            preds = self.model.final_pred(preds)
            preds = preds.cpu().detach().numpy()
            preds = np.transpose(preds[:2, :])
        return preds


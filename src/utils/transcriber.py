import os
import argparse
import copy
import time

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import soundfile as sf
import librosa
import torchaudio
from tqdm import tqdm
import pyaudio as pa
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate

import sys
# 상위 폴더 경로
denoiser_directory = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
denoiser_directory = os.path.join(denoiser_directory, 'denoiser')
sys.path.append(denoiser_directory)
print(denoiser_directory)
from denoiser import pretrained

from .audio_dataset import AudioDataset
from .audio_utils import AudioChunkIterator
from .decoder import ChunkBufferDecoder
from .microphone import Microphone

def load_asr_model(model_path, device):
    return nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path, map_location=device)

def load_denoiser_model(args):
    return pretrained.get_model(args).to(args.device)

class DenoiseTranscriber:
    def __init__(self, args):
        # Load denoiser model
        self.device = args.device
        self.num_workers = args.num_workers
        self.inference = args.inference
        self.mode = args.mode
        self.audio_path = args.audio_path
        self.manifest_path = args.manifest_path
        self.use_denoiser = not args.disable_denoiser

        if self.use_denoiser:
            self.denoiser_args = argparse.Namespace(
                device=args.device,
                dry=args.denoiser_dry,
                batch_size=1,
                num_workers=args.num_workers,
                model_path=args.denoiser_model_path,
                denoiser_output_save=args.denoiser_output_save,
                out_dir=args.denoiser_output_dir,
                )
            self.denoiser_model = pretrained.get_model(self.denoiser_args).to(self.device)
            self.denoiser_dry = args.denoiser_dry
            self.denoiser_output_save = args.denoiser_output_save
            self.denoiser_output_dir = args.denoiser_output_dir
        else:
            self.denoiser_model = None
            self.denoiser_dry = None
            self.denoiser_output_save = None
            self.denoiser_output_dir = None

        # Load asr model
        self.asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.asr_model_path, map_location=self.device)
        asr_cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(asr_cfg.preprocessor, False) # Make config overwrite-able
        asr_cfg.preprocessor.dither = 0.0
        asr_cfg.preprocessor.pad_to = 0    
        OmegaConf.set_struct(asr_cfg.preprocessor, True) # Disable config overwriting
        self.asr_model.preprocessor = self.asr_model.from_config_dict(asr_cfg.preprocessor)
        if self.device == 'cuda':
            self.asr_model.cuda()

        # Prepare data loader
        self.sample_rate = self.asr_model.preprocessor._cfg['sample_rate'] 
        self.chunk_len_in_secs = args.chunk_length
        self.context_len_in_secs = args.context_length
        self.buffer_len_in_secs = self.chunk_len_in_secs + self.context_len_in_secs # total length of audio to be fed to the model
        self.chunk_len = self.sample_rate*self.chunk_len_in_secs
        self.stride = 4
        self.buffer_len = self.sample_rate*self.buffer_len_in_secs

        self.sampbuffer = np.zeros([self.buffer_len], dtype=np.float32)

        self.asr_decoder = ChunkBufferDecoder(self.asr_model, 
                                        self.denoiser_model, 
                                        stride=self.stride, 
                                        chunk_len_in_secs=self.chunk_len_in_secs, 
                                        buffer_len_in_secs=self.buffer_len_in_secs, 
                                        denoise_dry=self.denoiser_dry)
        
        if self.mode == 'microphone':
            self.microphone = Microphone(self.sample_rate, self.chunk_len)

    def transcribe(self, audio_path=None, manifest_path=None, inference_result_path='inference_result.txt', callback=None):
        if self.mode == 'file':
            if self.inference:
                assert manifest_path is not None
                self.lines = []
                self.ref_transcriptions = []
                self.transcriptions = []
                dataset = AudioDataset(manifest_path)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                for i, data in enumerate(tqdm(dataloader)):
                    audio_path, audio_len, ref_transcription = data
                    audio_path, audio_len, ref_transcription = audio_path[0], audio_len[0], ref_transcription[0]
                    start = time.time()
                    transcription, length_in_secs = self.transcribe_single_file(audio_path)
                    end = time.time()
                    line = f'{ref_transcription}\t{transcription}\t{end-start}\t{length_in_secs}'
                    self.lines.append(line)
                    self.ref_transcriptions.append(ref_transcription)
                    self.transcriptions.append(transcription)
                wer = word_error_rate(hypotheses=self.transcriptions, references=self.ref_transcriptions)
                
                # with open("customer_result_cuda_no_denoiser.txt", "w") as f:
                with open(inference_result_path, "w") as f:
                    f.write(f"wer: {wer}\n")
                    f.write("\n".join(self.lines))
                print(f"wer: {wer}")
            else:
                assert audio_path is not None
                transcription = self.transcribe_single_file(audio_path, callback=callback)

        elif self.mode == 'microphone':
            self.microphone.start_stream(self.transcribe_microphone_data)
            self.microphone.run()
            self.microphone.close()
                
    def transcribe_microphone_data(self, in_data, frame_count, time_info, status):
        signal = np.frombuffer(in_data, dtype=np.int16)
        self.sampbuffer[:-self.chunk_len] = self.sampbuffer[self.chunk_len:]
        self.sampbuffer[-self.chunk_len:] = signal
        transcription = self.asr_decoder.transcribe_signal(self.sampbuffer)
        os.system('cls' if os.name =='nt' else 'clear')
        print("Press 'q' and Enter to stop the microphone: ")
        print(transcription)
        return (in_data, pa.paContinue)

    def transcribe_single_file(self, audio_path=None, callback=None):
        assert audio_path is not None
        self.asr_decoder.reset()
        samples, length = self.get_samples(audio_path, target_sr=self.sample_rate)
        chunk_reader = AudioChunkIterator(samples, self.chunk_len_in_secs, self.sample_rate)
        for chunk in chunk_reader:
            self.sampbuffer[:-self.chunk_len] = self.sampbuffer[self.chunk_len:]
            self.sampbuffer[-self.chunk_len:] = chunk
            transcription = self.asr_decoder.transcribe_signal(self.sampbuffer)
            if self.inference:
                pass
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                if callback is not None:
                    callback(transcription)
                    print('callback')
                else:
                    print(transcription)
        if self.denoiser_output_save:
            wav = torch.cat(self.asr_decoder.denoise_buffers, dim=-1)
            wav = wav[:, :length]
            self.save_wavs(wav, audio_path, self.denoiser_output_dir, self.sample_rate)
        return transcription, length/self.sample_rate

    def get_samples(self, audio_file, target_sr=16000):
        with sf.SoundFile(audio_file, 'r') as f:
            file_sample_rate = f.samplerate # sample rate of the audio file
            samples = f.read()
            if len(samples.shape) >= 2: # if audio has more than one channel
                samples = samples[:, 1]
            if file_sample_rate != target_sr:
                samples = librosa.core.resample(samples, orig_sr=file_sample_rate, target_sr=target_sr)
            samples = samples.transpose()
            length = len(samples)
            return samples, length

    def save_wavs(self, wav, filename, out_dir, sr=16_000):
        # Normalize audio if it prevents clipping
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0]) + "_enhanced.wav"
        wav = wav / max(wav.abs().max().item(), 1)
        torchaudio.save(filename, wav.cpu(), sr)
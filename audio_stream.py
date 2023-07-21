"""
Example code to infer from multiple files

python audio_stream.py \
    --device cuda\
    --inference\
    --manifest_path "/home/work/audrey2/dataset/고객 응대 음성/test_manifest.json"\
    --disable_denoiser\ # if you don't want to use denoiser
    --inference_result_path customer_result_cuda_no_denoiser.txt

python  audio_stream.py --audio_path "./audio_example/0001.wav" --device cuda
"""

import os
import sys
import time
import multiprocessing
import argparse
import copy
import math
import json
from omegaconf import OmegaConf

import numpy as np
from tqdm import tqdm
import pyaudio as pa
import soundfile as sf
import librosa
import torch
from torch.utils.data import DataLoader
import torchaudio

import nemo
import nemo.collections.asr as nemo_asr
from nemo.core.classes import IterableDataset
from nemo.collections.asr.metrics.wer import word_error_rate

sys.path.append('./denoiser')
import denoiser
from denoiser import pretrained, distrib
from denoiser.audio import Audioset
from denoiser.demucs import DemucsStreamer

from datasets import load_dataset
from jiwer import wer

parser = argparse.ArgumentParser(
        'Denoiser+ASR',
        description="Speech enhancement using Denoiser and Speech recognition using Conformer CTC")

parser.add_argument('--device', default="cpu", choices=['cpu', 'cuda'],)
parser.add_argument('--num_workers', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--inference', action="store_true",
                        help="Whether to use inference mode. Inference mode takes in audio files, split them into batches, and passes them through the model. Finally, it returns a WER.")
parser.add_argument('--inference_result_path', type=str, default='inference_result.txt',
                        help='path to save inference result')
parser.add_argument('--mode', type=str, default='file', choices=['file', 'microphone'],
                        help='mode for input audio')
parser.add_argument('--audio_path', type=str, default='./audio_example/0001.wav',
                        help='path to audio file.')
parser.add_argument('--manifest_path', type=str, default='manifest.json',
                        help='path to manifest json. It is for inference mode')
parser.add_argument('--chunk_length', type=int, default=1,
                        help='chunk length in seconds. chunk length means the length of new audio data that will be passed to the model at once.')
parser.add_argument('--context_length', type=int, default=1,
                        help='context length in seconds. Context length means the length of previous audio chunks to be fed into the model')


parser.add_argument('--disable_denoiser', action="store_true",
                        help="Whether to use denoiser")
parser.add_argument('--denoiser_dry', type=float, default=0.05,
                    help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
parser.add_argument('--denoiser_output_save', action="store_true",
                        help="save the denoised audio file.")
parser.add_argument('--denoiser_output_dir', type=str, default="./enhanced",
                        help="path to save denoised audio file if denoiser_output_save is true")
parser.add_argument('--denoiser_model_path', type=str, default="./checkpoint/denoiser.th",
                    help="path to denoiser model checkpoint")

parser.add_argument('--asr_model_path', type=str, default="./checkpoint/Conformer-CTC-BPE.nemo",
                    help="path to asr model checkpoint")


class AudioChunkIterator():
    def __init__(self, samples, chunk_len_in_secs, sample_rate):
        self._samples = samples
        self._chunk_len = chunk_len_in_secs*sample_rate
        self._start = 0
        self.output=True
   
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False
   
        return chunk

def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    
    audio_signal= []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        
    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths

# simple data layer to pass audio signal
class AudioBuffersDataLayer(IterableDataset):
    def __init__(self):
        super().__init__()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return torch.as_tensor(self.signal, dtype=torch.float32), \
            torch.as_tensor(self.signal_shape[0], dtype=torch.int64)

    def set_single_signal(self, signal):
        self.signal = signal
        self.signal_shape = self.signal.shape

    def __len__(self):
        return 1

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path=None):
        self.manifest_path = manifest_path
        self.data = []
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data['audio_filepath'], data['duration'], data['text']

def timer_decorator(function, is_print=False):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        if is_print:
            print(f"Function {function.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper



class ChunkBufferDecoder:
    def __init__(self,asr_model, denoiser_model=None, stride=4, chunk_len_in_secs=1, buffer_len_in_secs=3, denoise_dry=0.05):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.denoiser_model = denoiser_model
        if denoiser_model is not None:
            self.denoise_dry = denoise_dry
            self.denoiser_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.denoise_buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_secs
        self.buffer_len = buffer_len_in_secs
        self.sample_rate = self.asr_model.preprocessor._cfg['sample_rate'] 
        assert(chunk_len_in_secs<=buffer_len_in_secs)
        
        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_secs = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(self.chunk_len / self.model_stride_in_secs)
        self.blank_id = len(asr_model.decoder.vocabulary)
        self.plot=False

    @timer_decorator
    @torch.no_grad()
    def transcribe_signal(self, signal, merge=True, plot=False):
        self.plot = plot
        self.buffers.append(signal)
        self.data_layer.set_single_signal(signal)
        self._get_single_preds()
        return self.decode_final(merge)

    def _get_single_preds(self):
        device = self.asr_model.device
        batch = next(iter(self.data_loader))

        audio_signal, audio_signal_len = batch
        audio_signal, audio_signal_len = audio_signal.to(device), audio_signal_len.to(device)
        
        if self.denoiser_model is not None:
            audio_signal = self.denoise_estimate(audio_signal)[0]
            self.denoise_buffers.append(audio_signal[:,self.chunk_len*self.sample_rate:])
        
        log_probs, encoded_len, predictions = self.asr_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
        preds = torch.unbind(predictions)
        for pred in preds:
            self.all_preds.append(pred.cpu().numpy())
    
    @timer_decorator
    @torch.no_grad()   
    def denoise_estimate(self, noisy):
        estimate = self.denoiser_model(noisy)
        estimate = (1 - self.denoise_dry) * estimate + self.denoise_dry * noisy
        return estimate

    def decode_final(self, merge=True, extra=0):
        self.unmerged = []
        self.toks_unmerged = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len - self.chunk_len) / 2) / self.model_stride_in_secs)

        decoded_frames = []
        all_toks = []
        for pred in self.all_preds:
            ids, toks = self._greedy_decoder(pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)

        for decoded in decoded_frames:
            self.unmerged += decoded[len(decoded) - 1 - delay:len(decoded) - 1 - delay + self.n_tokens_per_chunk]
        if self.plot:
            for i, tok in enumerate(all_toks):
                plt.plot(self.buffers[i])
                plt.show()
                print("\nGreedy labels collected from this buffer")
                print(tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk])                
                self.toks_unmerged += tok[len(tok) - 1 - delay:len(tok) - 1 - delay + self.n_tokens_per_chunk]
            print("\nTokens collected from succesive buffers before CTC merge")
            print(self.toks_unmerged)


        if not merge:
            return self.unmerged
        return self.greedy_merge(self.unmerged)
    
    
    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i])
        return ids, s
         
    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p.item())
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis

    def reset(self):
        self.buffers = []
        self.all_preds = []
        self.denoise_buffers = []

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

    def transcribe(self, audio_path=None, manifest_path=None, inference_result_path='inference_result.txt'):
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
                transcription = self.transcribe_single_file(audio_path)

        elif self.mode == 'microphone':
            # Create an audio object
            self.microphone = pa.PyAudio()
            self.asr_decoder.reset()

            # Open the stream
            stream = self.microphone.open(format=pa.paInt16,
                            channels=2,
                            rate=self.chunk_len,
                            input=True,
                            frames_per_buffer=self.chunk_len,
                            stream_callback=self.transcribe_microphone_data)

            stream.start_stream()

            # Keep the stream open for 10 seconds
            time.sleep(30)

            # Stop the stream
            stream.stop_stream()
            stream.close()

            # Terminate the PortAudio session
            self.microphone.terminate()
                
    def transcribe_microphone_data(self, in_data, frame_count, time_info, status):
        signal = np.frombuffer(in_data, dtype=np.int16)
        self.sampbuffer[:-self.chunk_len] = self.sampbuffer[self.chunk_len:]
        self.sampbuffer[-self.chunk_len:] = signal
        transcription = self.asr_decoder.transcribe_signal(self.sampbuffer)
        os.system('cls' if os.name =='nt' else 'clear')
        print(transcription)
        return (in_data, pa.paContinue)

    def transcribe_single_file(self, audio_path=None):
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

if __name__ == "__main__":
    args = parser.parse_args()

    transcriber = DenoiseTranscriber(args)


    if args.inference:
        transcriber.transcribe(manifest_path=args.manifest_path, inference_result_path=args.inference_result_path)
    else:
        transcriber.transcribe(audio_path=args.audio_path)
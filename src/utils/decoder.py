import math

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .audio_utils import AudioBuffersDataLayer, speech_collate_fn

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
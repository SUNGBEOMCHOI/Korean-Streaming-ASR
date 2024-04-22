import numpy as np
import torch
from nemo.core.classes import IterableDataset

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
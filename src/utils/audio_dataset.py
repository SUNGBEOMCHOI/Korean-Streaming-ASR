import json
from torch.utils.data import Dataset

class AudioDataset(Dataset):
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
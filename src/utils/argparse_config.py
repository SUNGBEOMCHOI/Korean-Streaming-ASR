import argparse
import multiprocessing

import torch

def setup_arg_parser():
    parser = argparse.ArgumentParser(
            'Denoiser+ASR',
            description="Speech enhancement using Denoiser and Speech recognition using Conformer CTC")

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'],)
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
    
    return parser
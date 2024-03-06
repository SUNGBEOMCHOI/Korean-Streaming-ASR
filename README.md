# Korean Streaming Automatic Speech Recognition
**Real-time streaming Korean speech-to-text model that can run on a CPU**

ASR (Automatic Speech Recognition) is a process that involves two distinct stages:

1. Speech Enhancement: In this stage, the incoming audio or speech signal is processed to reduce noise, improve clarity, and enhance the quality of the speech. Various techniques such as filtering, spectral subtraction, and deep learning-based methods may be employed to achieve speech enhancement. There are two main approaches for processing using deep learning techniques: waveform domain processing and spectrogram domain processing. We process waveform domain.

2. Speech Recognition: Once the speech signal has been enhanced, it is passed through the speech recognition system. In this stage, the system converts the processed audio into text by identifying and transcribing the spoken words. Modern ASR systems typically rely on advanced machine learning algorithms, such as deep neural networks, to accurately recognize and transcribe the speech.

Together, these two stages enable ASR systems to convert spoken language into text, making them valuable tools in various applications such as voice assistants, transcription services, and more.

We used denoiser from @[facebook](https://github.com/facebookresearch/denoiser) and @[Nemo framework](https://github.com/NVIDIA/NeMo) for conformer CTC.

<div align="center">
  <img src="https://github.com/SUNGBEOMCHOI/Korean-Streaming-ASR/assets/92682815/f140e147-74cb-43ee-8daa-c3356492b28a" height=75% width=80% alt="model_overview"/>
</div>


## Requirements
**Clone the Repository**
```bash
git clone https://github.com/SUNGBEOMCHOI/Korean-Streaming-ASR.git
cd Korean-Streaming-ASR
```

**Make Conda Environment**
```bash
conda create -n korean_asr python==3.8.10
conda activate korean_asr
```

**Installing Dependencies on Ubuntu**
```bash
sudo apt-get update
sudo apt-get install -y libsndfile1 ffmpeg libffi-dev portaudio19-dev
```

**Python Dependencies Installation**
1. Install PyTorch, torchvision, torchaudio, and the CUDA version of PyTorch by following the instructions on the official PyTorch website: https://pytorch.org/get-started/locally/.

2. Install the rest of the required Python packages using pip. Open a terminal and execute the following commands:

```bash
pip install Cython
pip install nemo_toolkit['all']==1.21
pip install PyAudio
pip install julius
pip install datasets
pip install ipywidgets
pip install --upgrade nbformat
pip install flask
pip install Flask-SocketIO
```

**Download Denoiser and ASR Models**
1. From the provided Google Drive link, download **denoiser.th**, **Conformer-CTC-BPE.nemo**. If you wish to train the ASR model, also download Conformer-CTC-BPE.ckpt.
2. Create a folder named **checkpoint** and place the downloaded files in it.

Google Drive Folder: [Download Here](https://drive.google.com/drive/folders/1Adv8kYXV1XGGoLY1XA36EI38kfk0r0WZ?usp=drive_link)

## Run
**File mode**

For CPU:
```bash
python  audio_stream.py --audio_path "./audio_example/0001.wav" --device cpu
```

For GPU:
```bash
python  audio_stream.py --audio_path "./audio_example/0001.wav" --device cuda
```

**Microphone mode**
```bash
python audio_stream.py --mode microphone --device cpu
```

**Web**
```
flask run
```

---
### Example
**Raw Wave(Input)**

<div align="center">
  
  https://github.com/SUNGBEOMCHOI/Korean-Streaming-ASR/assets/92682815/7f4b98f4-bda7-49ba-b191-c2a49f1399dd

</div>

**Clean Wave (enhanced by denoiser)**

<div align="center">

  https://github.com/SUNGBEOMCHOI/Korean-Streaming-ASR/assets/92682815/ac41e19e-8323-436b-914d-958b25d68de2

</div>


**Text (output)**
<div align="center">
  
![스트리밍 gif](https://github.com/SUNGBEOMCHOI/Korean-Streaming-ASR/assets/92682815/97175803-72ed-41dd-8baf-443200bc9022)

</div>


---
### Datasets

We collect data from [AI Hub](https://aihub.or.kr/)

**Stage 1**
Speech Enhancement

We initialized denoiser to dns48 (H = 48, trained on DNS dataset, # of Parameters : 18,867,937) and let enhancement module dry output by $\text{dry} \cdot x + (1-\text{dry}) \cdot \hat y$
We also apply STFT Loss for training the Speech Enhancement model.
We train the model on 카페,음식점 소음 & 시장, 쇼핑몰 소음 in 소음환경음성인식데이터

**Stage 2**
Speech to Text
<div align="center">
  
| Name | # of Samples(train/test) |
| --- | --- |
| 고객응대음성 | 2067668/21092 |
| 한국어 음성 | 620000/3000 |
| 한국인 대화 음성 | 2483570/142399 |
| 자유대화음성(일반남녀) | 1886882/263371 |
| 복지 분야 콜센터 상담데이터 | 1096704/206470 |
| 차량내 대화 데이터 | 2624132/332787 |
| 명령어 음성(노인남여) | 137467/237469 |
| Total | 10916423(13946시간)/1206588(1474시간) |

</div>

If you wanna more info, go to [KO STT(in Hunggingface)](https://huggingface.co/SungBeom/stt_kr_conformer_ctc_medium)


---
### References

```bibtex
@inproceedings{defossez2020real,
  title={Real Time Speech Enhancement in the Waveform Domain},
  author={Defossez, Alexandre and Synnaeve, Gabriel and Adi, Yossi},
  booktitle={Interspeech},
  year={2020}
}
```



import os
import json
import csv
import wave
import glob
import copy
import zipfile
import multiprocessing
from functools import partial
from contextlib import contextmanager
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
import struct
import pathlib
import datetime
import time
from pydub import AudioSegment
import datetime

from tqdm import tqdm
import chardet

from utils import extract_compressed_files_with_multiprocessing, extract_compressed_file



def extract_text(hub_name, text):
    """
    Process text with the given hub name and text.

    Args:
        hub_name (str): Name of the selected pretrained model. 
            It can be one of the following: 
            ["고객 응대 음성", "한국어 음성", "008.소음 환경 음성인식 데이터", "소상공인 고객 주문 질의-응답 텍스트",
             "자유대화 음성(일반남녀)", "한국인 대화 음성"]
        text (str): Text to be processed.

    Returns:
        str: Processed text.

    Examples:
        hub_name = "고객 응대 음성", text = "n/ (14개월)/(십 사 개월) 된 아기가 있는데요. 우리 아기랑 해외여행 가기 어디가 좋을까요? 아이한테 좋은 추억 주고 싶어서 고민이 많이 되네요.\n"
             =>  "14개월 된 아기가 있는데요. 우리 아기랑 해외여행 가기 어디가 좋을까요? 아이한테 좋은 추억 주고 싶어서 고민이 많이 되네요."
        hub_name = "한국어 음성", text = "그러니까 뭐 뭐 스쿠버 다이빙 하고 뭐 이렇게 뭐.\n"
             =>  "그러니까 뭐 뭐 스쿠버 다이빙 하고 뭐 이렇게 뭐."    
    """
    text = text.strip().replace('\n', '')
    if hub_name == '고객 응대 음성':
        raw_text = copy.deepcopy(text)
        speech_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\2', text)
        write_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\1', text)


        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')']

        for noise in NOISE:
            speech_text = speech_text.replace(noise+'/', '')
            write_text = write_text.replace(noise+'/', '')

        for specialChar in EXCEPT:
            speech_text = speech_text.replace(specialChar, '')
            write_text = write_text.replace(specialChar, '')
    elif hub_name == '한국어 음성':
        raw_text = copy.deepcopy(text)
        speech_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\2', text)
        write_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\1', text)

        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')']

        for noise in NOISE:
            speech_text = speech_text.replace(noise+'/', '')
            write_text = write_text.replace(noise+'/', '')

        for specialChar in EXCEPT:
            speech_text = speech_text.replace(specialChar, '')
            write_text = write_text.replace(specialChar, '')
        speech_text = speech_text.replace('  ', ' ')
        write_text = write_text.replace('  ', ' ')

    elif hub_name == '자유대화 음성(일반남녀)':
        matches = re.findall(r'\(([^)]*)\)', text)
        replacement = {match: match.split(':')[-1] for match in matches}

        for key, value in replacement.items():
            text = text.replace(f'({key})', value)
        raw_text = copy.deepcopy(text)
        speech_text = copy.deepcopy(text)
        write_text = ''
        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['(NO:)', '?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')', '~']
        for noise in NOISE:
            speech_text = speech_text.replace(noise+'/', '')

        for specialChar in EXCEPT:
            speech_text = speech_text.replace(specialChar, '')
        speech_text = speech_text.replace('  ', ' ')
    elif hub_name == '한국인 대화 음성' or hub_name == '상담 음성' or hub_name == '차량 내 대화 및 명령어 음성' or hub_name == '명령어 음성(소아,유아)' or hub_name == '명령어 음성(노인남녀)':
        if "#" in text:
            return None
        raw_text = copy.deepcopy(text)
        speech_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\2', text)
        write_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\1', text)
        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')', '~']

        for noise in NOISE:
            speech_text = speech_text.replace(noise+'/', '')
            write_text = write_text.replace(noise+'/', '')

        for specialChar in EXCEPT:
            speech_text = speech_text.replace(specialChar, '')
            write_text = write_text.replace(specialChar, '')
        text = text.replace('  ', ' ')

    elif hub_name == '186.복지 분야 콜센터 상담데이터':
        DUMMY = '0123456789' + 'ㅇo'
        for char in DUMMY:
            if char in text:
                return None
        raw_text = copy.deepcopy(text)
        speech_text = copy.deepcopy(text)
        write_text = copy.deepcopy(text)
        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')', '~']

        for noise in NOISE:
            speech_text = speech_text.replace(noise+'/', '')
            write_text = write_text.replace(noise+'/', '')

        for specialChar in EXCEPT:
            speech_text = speech_text.replace(specialChar, '')
            write_text = write_text.replace(specialChar, '')
        text = text.replace('  ', ' ')

    elif hub_name == '008.소음 환경 음성인식 데이터':
        text = text.replace('\r', '')
        raw_text = copy.deepcopy(text)
        speech_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\2', text)
        write_text = re.sub(r'\(([^/]+)\/([^)]+)\)', r'\1', text)

        NOISE = ['o', 'n', 'u', 'b', 'l']
        EXCEPT = ['?', '!', '/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '(', ')', '~']

        for noise in NOISE:
            speech_text = speech_text.replace(noise+'/', '')
            write_text = write_text.replace(noise+'/', '')

        for specialChar in EXCEPT:
            speech_text = speech_text.replace(specialChar, '')
            write_text = write_text.replace(specialChar, '')

    else:
        raise ValueError(f'Not supported hub name {hub_name}')
    speech_text = speech_text.strip()
    write_text = write_text.strip()
    return raw_text, speech_text, write_text

def change_root_directory(csv_path, old_root_dir='/home/work/audrey', new_root_dir='/home/work/audrey2'):
    """
    Change root directory of the file path written in csv file.

    Args:
        csv_path (str): Path of the csv file.
        new_root_dir (str): New root directory.

    Examples:
        csv_path = "/home/work/audrey2/dataset/한국어 음성/test_clean.csv"
        new_root_dir = "/new/path/"
        first file path in csv file = "/home/work/audrey2/한국어 음성/test/eval_clean/KsponSpeech_E00001.pcm"
        => "/new/path/한국어 음성/test/eval_clean/KsponSpeech_E00001.pcm"
    """
    with open(csv_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    with open(csv_path, "w", encoding="utf-8") as outfile:
        for line in lines:
            elements = line.strip().split(',')
            elements[0] = elements[0].replace(old_root_dir, new_root_dir)
            new_line = ','.join(elements) + '\n'
            outfile.write(new_line)

    print("Complete to change root directory of the file path written in csv file.")

def change_sampling_rate(file_path, target_sample_rate=16000):
    # Load audio file
    audio = AudioSegment.from_file(file_path)

    # Set the frame rate to target sampling rate
    audio = audio.set_frame_rate(target_sample_rate)

    # Export audio file
    new_file_path = file_path[:-4] + '_16000' + file_path[-4:]
    
    # if os.path.isfile(old_file_path):
    #     os.remove(old_file_path)
    # new_file_path = file_path
    audio.export(new_file_path, format="wav")
    return new_file_path

def get_audio_length(file_path, sample_rate=None, sample_width=None, num_channels=None):
    """
    Extract audio length from file path. File type can be ['wav', 'pcm'].

    Args:
        file_path (str): Path of the audio file.
        sample_rate (int): Sample rate of the audio file. (e.g., 16000Hz, 44100HZ)
        sample_width (int): Sample width in bytes (e.g., 2 for 16-bit audio)
        num_channels (int): Number of channels (e.g., 1 for mono, 2 for stereo)

    Returns:
        audio_length (float): Audio length in seconds.
    """
    file_type = file_path.split('.')[-1]
    if file_type == 'wav':
        with wave.open(file_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            duration = n_frames / float(frame_rate)
            return duration
    elif file_type == 'pcm':
        if sample_rate is None or sample_width is None or num_channels is None:
            raise ValueError("For PCM files, you must provide sample_rate, sample_width, and num_channels")

        file_size = os.path.getsize(file_path)
        bytes_per_sample = sample_width * num_channels
        num_samples = file_size // bytes_per_sample
        duration = num_samples / float(sample_rate)
        return duration
    else:
        raise ValueError("Unsupported file type. Supported types: 'wav' and 'pcm'")

def extract_meta_single_file(args):
    """
    Extract meta data from single file.

    Args:
        hub_name (str): Name of the selected pretrained model. 
            It can be one of the following: 
            ["고객 응대 음성", "한국어 음성", "008.소음 환경 음성인식 데이터", "소상공인 고객 주문 질의-응답 텍스트",
             "자유대화 음성(일반남녀)", "한국인 대화 음성"]
        file_data (tuple): It contains elements of the file path. (eg. root_folder, folder, subfolder, session)
    
    """
    hub_name, file_data = args
    if hub_name == '고객 응대 음성':
        root_folder, folder, subfolder, session = file_data
        json_path = os.path.join(root_folder, folder, subfolder, session, f"{session}.json")
        txt_paths = glob.glob(os.path.join(root_folder, folder, subfolder, session, "*.txt"))

        if not txt_paths:
            return None
        txt_path = txt_paths[0]

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text, speech_text, write_text = extract_text(hub_name, f.read())

        audio_path = json_data["dataSet"]["dialogs"][0]["audioPath"]
        audio_path = audio_path.replace('KresSpeech', root_folder)

        category = json_data["dataSet"]["typeInfo"]["category"]
        subcategory = json_data["dataSet"]["typeInfo"]["subcategory"]
        speaker_data = json_data["dataSet"]["typeInfo"]["speakers"][0]
        gender = speaker_data["gender"]
        age = speaker_data["age"]
        residence = speaker_data["residence"]
        speaker = speaker_data["id"]

        audio_length = get_audio_length(audio_path, sample_rate=None, sample_width=None, num_channels=None)

        return [audio_path, raw_text, speech_text, write_text, audio_length, category, subcategory, gender, age, residence, speaker]

    elif hub_name == '한국어 음성':
        root_folder, audio_path, text = file_data
        raw_text, speech_text, write_text  = extract_text(hub_name, text)
        if audio_path.startswith('KsponSpeech_eval'):
            audio_path = audio_path.replace(f"{audio_path.split('/')[0]}/{audio_path.split('/')[1]}", root_folder) # for test dataset
        else:
            audio_path = os.path.join(root_folder, audio_path) # for train dataset
        audio_length = get_audio_length(audio_path, sample_rate=16000, sample_width=2, num_channels=1)
        return [audio_path, raw_text, speech_text, write_text, audio_length]

    elif hub_name == '186.복지 분야 콜센터 상담데이터':
        json_path = file_data

        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except:
            try:
                with open(json_path, 'r', encoding='CP949') as file:
                    data = json.load(file)
                print('cp949', json_path)
            except:
                print(json_path)
                return None

        text = data['inputText'][0]['orgtext']
        age = data['info'][0]['metadata']['speaker_age']
        category1 = data['info'][0]['metadata']['category1']
        category2 = data['info'][0]['metadata']['category2']
        category3 = data['info'][0]['metadata']['category3']
        gender = data['info'][0]['metadata']['speaker_sex']
        speaker = data['info'][0]['metadata']['speaker_id']

        try:
            raw_text, speech_text, write_text = extract_text(hub_name, text)
        except:
            return None
        audio_path = json_path.replace('json', 'wav')
        try:
            audio_length = get_audio_length(audio_path)
        except:
            return None

        return [audio_path, raw_text, speech_text, write_text, audio_length, gender, age, speaker]

    elif hub_name == '한국인 대화 음성':
        root_folder, audio_path, text, gender, age, residence, quality = file_data

        gender_dict = {'M':'남', 'F':'여'}
        age_dict = {'C':'유아', 'T':'청소년', 'A':'일반성인', 'S':'고령층', 'Z':'기타'}
        residence_dict = {'1':'서울,경기', '2':'강원', '3':'충청', '4':'경상', '5':'전라', '6':'제주', '9':'기타'}
        quality_dict = {'1':'정상', '2':'노이즈', '3':'잡음', '4':'원거리'}

        try:
            raw_text, speech_text, write_text = extract_text(hub_name, text)
        except:
            return None
        
        try:
            audio_path = audio_path.replace(f'/{audio_path.split("/")[1]}', root_folder)
            audio_length = get_audio_length(audio_path)
            gender = gender_dict[gender]
            age = age_dict[age]
            residence = residence_dict[residence]
            quality = quality_dict[quality]
        except:
            return None
        return [audio_path, raw_text, speech_text, write_text, audio_length, gender, age, residence, quality]

    elif hub_name == '008.소음 환경 음성인식 데이터':
        root_folder, folder, subfolder, session, save_folder, cut_dialogue = file_data # all session ends with .json
        json_path = os.path.join(root_folder, folder, subfolder, session)
        txt_path = json_path.replace('json', 'srt')
        audio_path = json_path.replace('.json', '.wav')
        audio_path_noise = audio_path.replace('SD', 'SN')

        gender_dict = {'남성':'남', '여성':'여'}

        os.makedirs(os.path.join(save_folder, folder, subfolder), exist_ok = True)

        results_list = []

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        type_info_data = json_data["typeInfo"][0]
        category = type_info_data["category"]
        subcategory = type_info_data["subCategory"]
        bgnoisespl = type_info_data["bgnoisespl"]
        avgnoisespl = type_info_data["avgnoisespl"]

        speakers_dict = {speaker_info['speaker']: [gender_dict[speaker_info['gender']], speaker_info['ageGroup']] for speaker_info in json_data["speakers"]}
        json_dialogs = json_data['dialogs']

        text = ''
        audio_length = get_audio_length(audio_path)
        
        if not cut_dialogue:
            for json_dialog in json_dialogs:
                text += json_dialog['speakerText'] + ' '
            speaker, gender, age = None, None, None
            raw_text, speech_text, write_text = extract_text(hub_name, text)
            results_list.append([audio_path, raw_text, speech_text, write_text, audio_length, audio_path_noise, category, subcategory, gender, age, speaker, bgnoisespl, avgnoisespl])
        else:
            with open(txt_path, 'r', encoding='utf-8') as f:
                clean_audio = AudioSegment.from_wav(audio_path)
                noise_audio = AudioSegment.from_wav(audio_path_noise)
                while True:
                    index_line = f.readline()
                    if not index_line: break
                    try:
                        json_dialog = json_dialogs.pop(0)
                    except:
                        print(f"json_dialogs is empty: {json_path}")
                        break

                    speaker = json_dialog['speaker']
                    try:
                        gender, age = speakers_dict[speaker]
                    except:
                        # print(f"speaker {speaker} not found, file {json_path}")
                        speaker, gender, age = None, None, None
                    
                    index_line = index_line.replace('\n', '')
                    try:
                        audio_count = int(index_line)
                    except:
                        print(f"invalid index line {index_line}, file {json_path}")
                        for _ in range(3):
                            line = f.readline().replace('\n', '')
                            if not line: continue

                    timestamp_line = f.readline().replace('\n', '')

                    # separte a file into files
                    if "-->" in timestamp_line:
                        start_time, end_time = timestamp_line.split(' --> ')
                        
                        start_time = time.strptime(start_time.split(',')[0],'%H:%M:%S')
                        start_time = datetime.timedelta(hours=start_time.tm_hour,minutes=start_time.tm_min,seconds=start_time.tm_sec).total_seconds()
                        end_time = time.strptime(end_time.split(',')[0],'%H:%M:%S')
                        end_time = datetime.timedelta(hours=end_time.tm_hour,minutes=end_time.tm_min,seconds=end_time.tm_sec).total_seconds()
                        
                        length = end_time - start_time
                        # for unit : ms
                        start_time = start_time*1000
                        end_time = end_time*1000
                        
                        # process for clean and noisy
                        new_clean_audio = clean_audio[start_time:end_time]
                        new_clean_audio_path = audio_path[:-7] + f'_{audio_count}' + audio_path[-7:]
                        new_clean_audio_path = new_clean_audio_path.replace(root_folder, save_folder)
                        new_clean_audio.export(new_clean_audio_path, format="wav")

                        new_noise_audio = noise_audio[start_time:end_time]
                        new_noise_audio_path = new_clean_audio_path.replace('SD', 'SN')
                        new_noise_audio.export(new_noise_audio_path, format="wav")
                        
                    else:
                        print(f'No "-->" in timestamp line : {timestamp_line}, file {json_path}')
                        while True:
                            line = f.readline().replace('\n', '')
                            if not line: break

                    # process for text
                    text = f.readline().replace('\n', '')
                    while True:
                        text_line = f.readline().replace('\n', '')
                        text += text_line
                        if not text_line: break
                    raw_text, speech_text, write_text = extract_text(hub_name, text)

                    results_list.append([new_clean_audio_path, raw_text, speech_text, write_text, length, new_noise_audio_path, category, subcategory, gender, age, speaker, bgnoisespl, avgnoisespl])

        return results_list


    elif hub_name == '자유대화 음성(일반남녀)':
        root_folder, folder, session = file_data
        json_path = os.path.join(root_folder, folder, session)

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        raw_text, speech_text, write_text = extract_text(hub_name, json_data['발화정보']['stt'])
        audio_path = json_path.replace('json', 'wav')
        residence = json_data["대화정보"]["cityCode"]
        speaker_data = json_data["녹음자정보"]
        gender = speaker_data["gender"]
        age = speaker_data["age"]
        speaker = speaker_data["recorderId"]
        try:
            audio_length = get_audio_length(audio_path)
        except:
            return None

        return [audio_path, raw_text, speech_text, write_text, audio_length, gender, age, residence, speaker]

    elif hub_name == '상담 음성':
        txt_path = file_data

        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                line = file.readline()
        except:
            try:
                with open(txt_path, 'r', encoding='CP949') as file:
                    line = file.readline()
                print('cp949', txt_path)
            except:
                print(txt_path)
                return None

        text = line.replace('\n', '')

        try:
            raw_text, speech_text, write_text = extract_text(hub_name, text)
        except:
            return None
        audio_path = txt_path.replace('txt', 'wav')
        try:
            audio_path = change_sampling_rate(audio_path, target_sample_rate=16000)
            audio_length = get_audio_length(audio_path)
        except:
            return None

        return [audio_path, raw_text, speech_text, write_text, audio_length]

    elif hub_name == '명령어 음성(소아,유아)' or hub_name == '명령어 음성(노인남녀)':
        json_path = file_data

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        try:
            audio_path = json_path.replace('.json', '.wav')
            audio_path = change_sampling_rate(audio_path, target_sample_rate=16000)
            audio_length = get_audio_length(audio_path)
        except:
            try:
                path_parts = json_path.split('/')
                filename = os.path.splitext(path_parts[-1])[0]
                audio_path = '/'.join(path_parts[:-1] + [filename[:filename.index('-')], filename]) + '.wav'
                audio_path = change_sampling_rate(audio_path, target_sample_rate=16000)
                audio_length = get_audio_length(audio_path)
            except:
                return None

        text = json_data['전사정보']['LabelText']

        try:
            raw_text, speech_text, write_text = extract_text(hub_name, text)
        except:
            return None

        return [audio_path, raw_text, speech_text, write_text, audio_length]

    elif hub_name == '차량 내 대화 및 명령어 음성':
        audio_path = file_data
        if audio_path.endswith('16000.wav'):
            os.remove(audio_path)
            return None

        json_path = audio_path.replace('wav', 'json')
        json_path_convert_dict = {'차량/validation/자율주행/1':'self', '차량/validation/자율주행/2':'self', '차량/validation/카투홈/1':'c2h', '차량/validation/홈투카/1':'h2c', '차량/validation/홈투카/2':'h2c', '차량/validation/AI비서/1':'sec',
                             '차량/training/자율주행/1':'self', '차량/training/자율주행/1':'self', '차량/training/자율주행/2':'self', '차량/training/자율주행/3':'self', '차량/training/자율주행/4':'self',
                             '차량/training/카투홈/1':'c2h', '차량/training/카투홈/2':'c2h', '차량/training/카투홈/3':'c2h', '차량/training/카투홈/4':'c2h', '차량/training/카투홈/5':'c2h',
                             '차량/training/홈투카/1':'h2c', '차량/training/홈투카/2':'h2c', '차량/training/홈투카/3':'h2c', '차량/training/홈투카/4':'h2c', '차량/training/홈투카/5':'h2c', '차량/training/홈투카/6':'h2c', '차량/training/홈투카/7':'h2c', '차량/training/홈투카/8':'h2c',
                             '차량/training/AI비서/1':'sec', '차량/training/AI비서/2':'sec', '차량/training/AI비서/3':'sec', '차량/training/AI비서/4':'sec', '차량/training/AI비서/5':'sec', }
        for json_path_convert_key, json_path_convert_value  in json_path_convert_dict.items():
            json_path = json_path.replace(json_path_convert_key, json_path_convert_value)
        
        try:
            audio_path = change_sampling_rate(audio_path, target_sample_rate=16000)
            audio_length = get_audio_length(audio_path)
        except:
            print('error: ', audio_path)
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except:
            try:
                with open(json_path, 'r', encoding='CP949') as file:
                    data = json.load(file)
                print('cp949', json_path)
            except:
                print('error file: ', json_path)
                return None

        text = data['전사정보']['LabelText']

        try:
            raw_text, speech_text, write_text = extract_text(hub_name, text)
        except:
            return None
        

        return [audio_path, raw_text, speech_text, write_text, audio_length]

    else:
        raise ValueError(f'Not supported hub name {hub_name}')

def get_meta_data(hub_name, absolute_data_path="/home/work/audrey2/dataset/", already_unzip=True, num_workers=4):
    """
    Get meta data of the dataset and save it as csv file.

    Args:
        hub_name (str): Name of the selected pretrained model. 
            It can be one of the following: 
            ["고객 응대 음성", "한국어 음성", "008.소음 환경 음성인식 데이터", "소상공인 고객 주문 질의-응답 텍스트",
             "자유대화 음성(일반남녀)", "한국인 대화 음성"]
        absolute_data_path (str): Absolute path of the dataset.
        already_unzip (bool): If True, the dataset is already unzipped. If False, extract the dataset.
    """
    absolute_data_path = absolute_data_path + hub_name
    if hub_name == '고객 응대 음성':        
        zip_folder_list = [f'{absolute_data_path}/Train', f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/train', f'{absolute_data_path}/test']
        output_csv_list = [f'{absolute_data_path}/train.csv', f'{absolute_data_path}/test.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length", "category", "subcategory", "gender", "age", "residence", "speaker"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder)
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers=4)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                file_data_list = []
                for folder in os.listdir(root_folder):
                    for subfolder in os.listdir(os.path.join(root_folder, folder)):
                        for session in os.listdir(os.path.join(root_folder, folder, subfolder)):
                            file_data_list.append((root_folder, folder, subfolder, session))

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in file_data_list]), total=len(file_data_list)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)

    elif hub_name == '한국어 음성':
        # zip_folder_list = [f'{absolute_data_path}/Train']
        zip_folder_list = [f'{absolute_data_path}/Train']
        root_folder_list = [f'{absolute_data_path}/train', f'{absolute_data_path}/test/eval_clean', f'{absolute_data_path}/test/eval_other']
        script_file_list = [f'{absolute_data_path}/scripts/train.trn', f'{absolute_data_path}/scripts/eval_clean.trn', f'{absolute_data_path}/scripts/eval_other.trn']
        output_csv_list = [f'{absolute_data_path}/train.csv', f'{absolute_data_path}/test_clean.csv', f'{absolute_data_path}/test_other.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv, script_file in zip(root_folder_list, output_csv_list, script_file_list):
            with open(output_csv, 'w', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(csv_column)

                file_data_list = []
                with open(script_file, 'r', encoding='utf-8') as scriptfile:
                    for idx, line in enumerate(scriptfile):
                        if not line:
                            break
                        audio_path, text = line.split(" :: ")
                        file_data_list.append((root_folder, audio_path, text))

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in file_data_list]), total=len(file_data_list)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)

    elif hub_name == '자유대화 음성(일반남녀)':
        zip_folder_list = [f'{absolute_data_path}/Validation', f'{absolute_data_path}/Training']
        # zip_folder_list = [f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/test', f'{absolute_data_path}/train']
        # root_folder_list = [f'{absolute_data_path}/test']
        output_csv_list = [f'{absolute_data_path}/test.csv', f'{absolute_data_path}/train.csv']
        # output_csv_list = [f'{absolute_data_path}/test.csv']
        # csv_column = ["file", "text", "category", "subcategory", "gender", "age", "residence", "speaker"]
        csv_column = ["file", "text", "speech_text", "write_text", "length", "gender", "age", "residence", "speaker"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                file_data_list = []
                for folder in os.listdir(root_folder):
                    for session in os.listdir(os.path.join(root_folder, folder)):
                        if session.endswith('.json'):
                            file_data_list.append((root_folder, folder, session))

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in file_data_list]), total=len(file_data_list)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)

    elif hub_name == '상담 음성':
        zip_folder_list = [f'{absolute_data_path}/Validation', f'{absolute_data_path}/Training']
        # zip_folder_list = [f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/test', f'{absolute_data_path}/train']
        # root_folder_list = [f'{absolute_data_path}/test']
        # output_csv_list = [f'{absolute_data_path}/train.csv', f'{absolute_data_path}/test.csv']
        output_csv_list = [f'/home/work/dataset/상담 음성/test.csv', f'/home/work/dataset/상담 음성/train.csv']
        # output_csv_list = [f'{absolute_data_path}/test.csv']
        # csv_column = ["file", "text", "category", "subcategory", "gender", "age", "residence", "speaker"]
        csv_column = ["file", "text", "speech_text", "write_text", "length"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                txt_files = glob.glob(root_folder + '/**/*.txt', recursive=True)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in txt_files]), total=len(txt_files)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)
    
    elif hub_name == '명령어 음성(소아,유아)' or hub_name == '명령어 음성(노인남녀)':
        # zip_folder_list = [f'{absolute_data_path}/Validation']
        zip_folder_list = [f'{absolute_data_path}/Training']
        # root_folder_list = [f'{absolute_data_path}/test']
        root_folder_list = [f'{absolute_data_path}/train']
        # output_csv_list = [f'{absolute_data_path}/test.csv']
        output_csv_list = [f'{absolute_data_path}/train.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                json_files = glob.glob(root_folder + '/**/*.json', recursive=True)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in json_files]), total=len(json_files)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)
    
    elif hub_name == '한국인 대화 음성':
        zip_folder_list = [f'{absolute_data_path}/Train']
        # zip_folder_list = [f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/train']
        # root_folder_list = [f'{absolute_data_path}/test']
        output_csv_list = [f'{absolute_data_path}/train.csv']
        # output_csv_list = [f'{absolute_data_path}/test.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length", "gender", "age", "residence", "quality"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                file_data_list = []
                for folder in os.listdir(root_folder):
                    if folder == '1.라벨링데이터':
                        for subfolder in os.listdir(os.path.join(root_folder, folder)):
                            for subsubfolder in os.listdir(os.path.join(root_folder, folder, subfolder)):
                                for session in os.listdir(os.path.join(root_folder, folder, subfolder, subsubfolder)):
                                    if session.endswith('scripts.txt'):
                                        script_file = os.path.join(root_folder, folder, subfolder, subsubfolder, session)
                                        metadata_file = os.path.join(root_folder, folder, subfolder, subsubfolder, session.replace('scripts', 'metadata'))
                                        with open(script_file, 'r', encoding='utf-8') as scriptfile:
                                            with open(metadata_file, 'r', encoding='utf-8') as metadatafile:
                                                for script_line, metadata_line in zip(scriptfile, metadatafile):
                                                    if not script_line:
                                                        break
                                                    audio_path, text = script_line.split(" :: ")
                                                    try:
                                                        _, _, _, gender, age, _, residence, _, quality = metadata_line.strip().split(" | ")
                                                    except:
                                                        _, _, _, gender, age, _, residence, _, quality = metadata_line.strip().split("|")
                                                    file_data_list.append((root_folder, audio_path, text, gender, age, residence, quality))

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in file_data_list]), total=len(file_data_list)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)

    elif hub_name == '186.복지 분야 콜센터 상담데이터':
        zip_folder_list = [f'{absolute_data_path}/Train']
        # zip_folder_list = [f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/train']
        # root_folder_list = [f'{absolute_data_path}/test']
        output_csv_list = [f'{absolute_data_path}/train.csv']
        # output_csv_list = [f'{absolute_data_path}/test.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length", "gender", "age", "speaker"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                json_files = glob.glob(root_folder + '/**/*.json', recursive=True)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in json_files]), total=len(json_files)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)

    elif hub_name == '차량 내 대화 및 명령어 음성':
        zip_folder_list = [f'{absolute_data_path}/Validation', f'{absolute_data_path}/Train']
        # zip_folder_list = [f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/test', f'{absolute_data_path}/train']
        # root_folder_list = [f'{absolute_data_path}/test']
        output_csv_list = [f'{absolute_data_path}/test.csv', f'{absolute_data_path}/train.csv']
        # output_csv_list = [f'{absolute_data_path}/test.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, output_csv in zip(root_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")
            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                wav_files = glob.glob(root_folder + '/**/*.wav', recursive=True)

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in wav_files]), total=len(wav_files)))

                for result in results:
                    if result:
                        csv_writer.writerow(result)
    
    elif hub_name == '008.소음 환경 음성인식 데이터':
        zip_folder_list = [f'{absolute_data_path}/Train', f'{absolute_data_path}/Validation']
        root_folder_list = [f'{absolute_data_path}/train', f'{absolute_data_path}/test']
        cut_dialogue = True # This data is too long(300 secs), so cut dialogue into small chunks
        if cut_dialogue:
            save_folder_list = [f'{absolute_data_path}/train_cut', f'{absolute_data_path}/test_cut']
            output_csv_list = [f'{absolute_data_path}/train_cut.csv', f'{absolute_data_path}/test_cut.csv']
        else:
            save_folder_list = [f'{absolute_data_path}/train', f'{absolute_data_path}/test']
            output_csv_list = [f'{absolute_data_path}/train.csv', f'{absolute_data_path}/test.csv']
        csv_column = ["file", "text", "speech_text", "write_text", "length", "filenoise", "category", "subcategory", "gender", "age", "speaker", "bgnoisespl", "avgnoisespl"]

        if not already_unzip:
            for zip_folder, root_folder in zip(zip_folder_list, root_folder_list):
                print(f"Extract {zip_folder} to {root_folder}")
                # extract_compressed_files(zip_folder, root_folder) # not using multiprocessing
                extract_compressed_files_with_multiprocessing(zip_folder, root_folder, num_workers)

        for root_folder, save_folder, output_csv in zip(root_folder_list, save_folder_list, output_csv_list):
            print(f"Extract meta data from {root_folder} and save it to {output_csv}")

            with open(output_csv, 'w', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_column)

                file_data_list = []
                for folder in os.listdir(root_folder):
                    for subfolder in os.listdir(os.path.join(root_folder, folder)):
                        for session in os.listdir(os.path.join(root_folder, folder, subfolder)):
                            if session.endswith('.json') and os.path.isfile(os.path.join(root_folder, folder, subfolder, session).replace('.json', '.wav')) :
                                file_data_list.append((root_folder, folder, subfolder, session, save_folder, cut_dialogue))

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(extract_meta_single_file, [(hub_name, file_data) for file_data in file_data_list]), total=len(file_data_list)))

                for sub_result in results:
                    for result in sub_result:
                        if result:
                            csv_writer.writerow(result)

                # for result in results:
                #     if result:
                #         csv_writer.writerow(result)

def convert_pcm_to_wav(pcm_file_path, num_channels=1, sample_rate=16000):
        """ 
        Convert pcm to wav file. 

        Args:
            pcm_file_path (str): pcm file path
            num_channels (int): number of channels
            sample_rate (int): sample rate
        """
        save_path = pcm_file_path.replace('.pcm', '.wav')
        pcm_data = pathlib.Path(pcm_file_path).read_bytes()
        waves = []
        waves.append(struct.pack('<4s', b'RIFF'))
        waves.append(struct.pack('I', 1))  
        waves.append(struct.pack('4s', b'WAVE'))
        waves.append(struct.pack('4s', b'fmt '))
        waves.append(struct.pack('I', 16))
        # audio_format, channel_cnt, sample_rate, bytes_rate(sr*blockalign:bites per sec), block_align, bps
        if num_channels == 2:
            waves.append(struct.pack('HHIIHH', 1, 2, sample_rate, 64000, 4, 16))  
        else:
            waves.append(struct.pack('HHIIHH', 1, 1, sample_rate, 32000, 2, 16))
        waves.append(struct.pack('<4s', b'data'))
        waves.append(struct.pack('I', len(pcm_data)))
        waves.append(pcm_data)
        waves[1] = struct.pack('I', sum(len(w) for w in waves[2:]))
        with open(save_path, 'wb') as file:
            file.write(b''.join(waves))
        # print(f"Save wav file to {save_path}")
        return None


def convert_csv_to_json(csv_path, json_path):
    """
    Convert csv file to json file. (It's for training NEMO ASR model)

    Args:
        csv_path (str): csv file path
        json_path (str): json file path
    """
    # Read the CSV data
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Process the CSV data and create JSON records
        json_records = []
        # pcm_datas = []
        for row in csv_reader:            
            # pcm_datas.append(row['file'])
            json_record = {
                "audio_filepath": row["file"].replace('pcm', 'wav'),
                # "audio_filepath": row["filenoise"],
                "duration": float(row["length"]),
                "text": row["speech_text"]
            }

            if len(str(json_record["duration"])) >= 7:
                print(json_record["duration"])
            else:           
                json_records.append(json_record)

    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     results = list(tqdm(executor.map(convert_pcm_to_wav, [(pcm_data) for pcm_data in pcm_datas]), total=len(pcm_datas)))

    # Write the JSON records to the output.json file
    with open(json_path, "w") as json_file:
        for record in json_records:
            # json_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            json_file.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ### change directory path in csv file
    # csv_path = "/home/work/audrey/speech_recognition/train.csv"
    # # csv_path = "/home/work/audrey/speech_recognition/test.csv"
    # new_root_dir='/home/work/audrey2'
    # change_root_directory(csv_path, new_root_dir) 

    ### extract one zip file
    # compressed_path = '/home/work/audrey3/dataset/명령어 음성(소아,유아)/Validation/[라벨]1.AI비서_라벨링_명령어(유소아)_validation.zip'
    # extracted_path = '/home/work/audrey3/dataset/명령어 음성(소아,유아)/test'
    # extract_compressed_file(compressed_path, extracted_path)

    ### extract all zip file in a directory
    # compressed_path = "/home/work/audrey2/dataset/자유대화 음성(일반남녀)/Training"
    # extracted_path = "/home/work/dataset/자유대화 음성(일반남녀)/train"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=7)

    # compressed_path = "/home/work/audrey2/dataset/상담 음성/Training"
    # extracted_path = "/home/work/audrey2/dataset/상담 음성/train"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=7)

    # compressed_path = "/home/work/audrey2/dataset/상담 음성/validation"
    # extracted_path = "/home/work/audrey2/dataset/상담 음성/train"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=7)

    # compressed_path = "/home/work/audrey3/dataset/명령어 음성(소아,유아)/Training"
    # extracted_path = "/home/work/audrey3/dataset/명령어 음성(소아,유아)/train"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=7)

    # compressed_path = "/home/work/audrey3/dataset/명령어 음성(소아,유아)/Validation"
    # extracted_path = "/home/work/audrey3/dataset/명령어 음성(소아,유아)/test"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=7)

    # compressed_path = "/home/work/audrey3/dataset/명령어 음성(노인남녀)/Training"
    # extracted_path = "/home/work/audrey3/dataset/명령어 음성(노인남녀)/train"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=7)

    # compressed_path = "/home/work/audrey3/dataset/명령어 음성(노인남녀)/Validation"
    # extracted_path = "/home/work/audrey3/dataset/명령어 음성(노인남녀)/test"
    # extract_compressed_files_with_multiprocessing(compressed_path, extracted_path, num_workers=1)

    
    ### Make CSV file which contains meta data
    # get_meta_data(hub_name='한국어 음성', absolute_data_path='/home/work/audrey2/dataset/', already_unzip=True, num_workers=8)
    # get_meta_data(hub_name='고객 응대 음성', absolute_data_path='/home/work/audrey2/dataset/', already_unzip=True, num_workers=12)
    # get_meta_data(hub_name='자유대화 음성(일반남녀)', absolute_data_path='/home/work/audrey3/dataset/', already_unzip=True, num_workers=20)
    get_meta_data(hub_name='상담 음성', absolute_data_path='/home/work/audrey2/dataset/', already_unzip=True, num_workers=20)
    # get_meta_data(hub_name='008.소음 환경 음성인식 데이터', absolute_data_path='/home/work/audrey2/dataset/', already_unzip=True, num_workers=20)
    # get_meta_data(hub_name='한국인 대화 음성', absolute_data_path='/home/work/audrey2/dataset/', already_unzip=True, num_workers=20)
    # get_meta_data(hub_name='186.복지 분야 콜센터 상담데이터', absolute_data_path='/home/work/audrey2/dataset/', already_unzip=True, num_workers=7)
    # get_meta_data(hub_name='차량 내 대화 및 명령어 음성', absolute_data_path='/home/work/audrey3/dataset/', already_unzip=True, num_workers=7)
    # get_meta_data(hub_name='명령어 음성(소아,유아)', absolute_data_path='/home/work/audrey3/dataset/', already_unzip=True, num_workers=7)
    # get_meta_data(hub_name='명령어 음성(노인남녀)', absolute_data_path='/home/work/audrey3/dataset/', already_unzip=True, num_workers=7)
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/한국어 음성/test_clean.csv', json_path='/home/work/audrey2/dataset/한국어 음성/test_clean_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/한국어 음성/train.csv', json_path='/home/work/audrey2/dataset/한국어 음성/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/한국어 음성/test_other.csv', json_path='/home/work/audrey2/dataset/한국어 음성/test_other_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/고객 응대 음성/train.csv', json_path='/home/work/audrey2/dataset/고객 응대 음성/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/고객 응대 음성/test.csv', json_path='/home/work/audrey2/dataset/고객 응대 음성/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/008.소음 환경 음성인식 데이터/test.csv', json_path='/home/work/audrey2/dataset/008.소음 환경 음성인식 데이터/noise_test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/008.소음 환경 음성인식 데이터/train.csv', json_path='/home/work/audrey2/dataset/008.소음 환경 음성인식 데이터/noise_train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/한국인 대화 음성/train.csv', json_path='/home/work/audrey2/dataset/한국인 대화 음성/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/한국인 대화 음성/test.csv', json_path='/home/work/audrey2/dataset/한국인 대화 음성/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/186.복지 분야 콜센터 상담데이터/train.csv', json_path='/home/work/audrey2/dataset/186.복지 분야 콜센터 상담데이터/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey2/dataset/186.복지 분야 콜센터 상담데이터/test.csv', json_path='/home/work/audrey2/dataset/186.복지 분야 콜센터 상담데이터/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/dataset/상담 음성/train.csv', json_path='/home/work/dataset/상담 음성/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/dataset/상담 음성/test.csv', json_path='/home/work/dataset/상담 음성/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/차량 내 대화 및 명령어 음성/test.csv', json_path='/home/work/audrey3/dataset/차량 내 대화 및 명령어 음성/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/차량 내 대화 및 명령어 음성/train.csv', json_path='/home/work/audrey3/dataset/차량 내 대화 및 명령어 음성/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/자유대화 음성(일반남녀)/train.csv', json_path='/home/work/audrey3/dataset/자유대화 음성(일반남녀)/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/자유대화 음성(일반남녀)/test.csv', json_path='/home/work/audrey3/dataset/자유대화 음성(일반남녀)/test_manifest.json')

    # change_root_directory("/home/work/audrey2/dataset/자유대화 음성(일반남녀)/train.csv", "/home/work/dataset", "/home/work/audrey2/dataset")

    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/자유대화 음성(일반남녀)/train.csv', json_path='/home/work/audrey3/dataset/자유대화 음성(일반남녀)/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/명령어 음성(소아,유아)/test.csv', json_path='/home/work/audrey3/dataset/명령어 음성(소아,유아)/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/명령어 음성(소아,유아)/train.csv', json_path='/home/work/audrey3/dataset/명령어 음성(소아,유아)/train_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/명령어 음성(노인남녀)/test.csv', json_path='/home/work/audrey3/dataset/명령어 음성(노인남녀)/test_manifest.json')
    # convert_csv_to_json(csv_path='/home/work/audrey3/dataset/명령어 음성(노인남녀)/train.csv', json_path='/home/work/audrey3/dataset/명령어 음성(노인남녀)/train_manifest.json')
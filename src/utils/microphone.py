import os
import time
import threading
import pyaudio as pa

class Microphone:
    def __init__(self, rate, chunk_size):
        self.pa = pa.PyAudio()
        self.rate = rate
        self.chunk_size = chunk_size
        self.stream = None
        self.device_index = self.select_device()
        self.running = True

    def clear_terminal(self):
        # 운영 체제에 따라 터미널을 지우는 명령을 실행합니다.
        os.system('cls' if os.name == 'nt' else 'clear')

    def select_device(self):
        self.clear_terminal()
        info = self.pa.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        # 사용 가능한 마이크 디바이스 목록을 출력합니다.
        print("Available microphone devices:")
        for i in range(0, num_devices):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")

        # 사용자에게 장치 선택을 요청합니다.
        while True:
            try:
                device_index = int(input("Select the device index to use: "))
                device_info = self.pa.get_device_info_by_index(device_index)
                if device_info['maxInputChannels'] > 0:
                    return device_index
                else:
                    print("Selected device is not a valid input device. Please try again.")
            except ValueError:
                print("Please enter a valid device index.")

    def start_stream(self, callback):
        self.stream = self.pa.open(format=pa.paInt16,
                                   channels=1,
                                   rate=self.rate,
                                   input=True,
                                   input_device_index=self.device_index,
                                   frames_per_buffer=self.chunk_size,
                                   stream_callback=callback)

    def run(self, duration=None):
        def stop_check():
            input("Press 'q' and Enter to stop the microphone: ")
            self.running = False

        if duration is None:
            # 사용자가 'q'를 입력할 때까지 계속 실행합니다.
            stop_thread = threading.Thread(target=stop_check)
            stop_thread.start()
            self.stream.start_stream()
            while self.running:
                time.sleep(0.1)
            self.stream.stop_stream()
            stop_thread.join()
        else:
            self.stream.start_stream()
            time.sleep(duration)
            self.stream.stop_stream()

    def close(self):
        self.stream.close()
        self.pa.terminate()

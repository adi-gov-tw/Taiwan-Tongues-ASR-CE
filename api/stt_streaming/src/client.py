from buffering_strategy.buffering_strategy_factory import BufferingStrategyFactory
import time


class Client:
    """
    Represents a client connected to the VoiceStreamAI server.

    This class maintains the state for each connected client, including their
    unique identifier, audio buffer, configuration, and a counter for processed audio files.

    Attributes:
        client_id (str): A unique identifier for the client.
        buffer (bytearray): A buffer to store incoming audio data.
        config (dict): Configuration settings for the client, like chunk length and offset.
        file_counter (int): Counter for the number of audio files processed.
        total_samples (int): Total number of audio samples received from this client.
        sampling_rate (int): The sampling rate of the audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
    """

    def __init__(
        self,
        client_id,
        sampling_rate,
        samples_width,
        job_id,
        last_start_time,
        transcript,
    ):
        self.client_id = client_id
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()
        # 用於整段連線期間的原始音訊彙整
        self.session_audio_buffer = bytearray()
        self.config = {
            "language": None,
            "processing_strategy": "silence_at_end_of_chunk",
            "processing_args": {
                "chunk_length_seconds": 1.5,
                "chunk_offset_seconds": 0.1,
            },
        }
        self.file_counter = 0
        self.chunk_save_counter = 0
        self.total_samples = 0
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.buffering_strategy = BufferingStrategyFactory.create_buffering_strategy(
            self.config["processing_strategy"], self, **self.config["processing_args"]
        )
        self.connect_time = None
        self.job_id = job_id
        self.last_start_time = last_start_time
        self.start_time = time.time()
        self.transcript = [] if transcript is None else transcript

    def update_config(self, config_data):
        self.config.update(config_data)
        self.buffering_strategy = BufferingStrategyFactory.create_buffering_strategy(
            self.config["processing_strategy"], self, **self.config["processing_args"]
        )

    def append_audio_data(self, audio_data):
        self.buffer.extend(audio_data)
        self.session_audio_buffer.extend(audio_data)
        self.total_samples += len(audio_data) / self.samples_width

    def clear_buffer(self):
        self.buffer.clear()

    def increment_file_counter(self):
        self.file_counter += 1

    def get_file_name(self):
        return f"{self.client_id}_{self.file_counter}.wav"

    def get_chunk_file_name(self):
        return f"{self.client_id}_chunk_{self.chunk_save_counter:06d}.wav"

    def increment_chunk_save_counter(self):
        self.chunk_save_counter += 1

    def get_session_file_name(self):
        return f"{self.client_id}_{self.job_id}.wav"

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        self.buffering_strategy.process_audio(websocket, vad_pipeline, asr_pipeline)

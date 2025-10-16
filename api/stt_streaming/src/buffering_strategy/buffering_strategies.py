import os, logging
import asyncio
import json
import time

from .buffering_strategy_interface import BufferingStrategyInterface


class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with silence detection.

    This class is responsible for handling audio chunks, detecting silence at the end of each chunk,
    and initiating the transcription process for the chunk.

    Attributes:
        client (Client): The client instance associated with this buffering strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the SilenceAtEndOfChunk buffering strategy.

        Args:
            client (Client): The client instance associated with this buffering strategy.
            **kwargs: Additional keyword arguments, including 'chunk_length_seconds' and 'chunk_offset_seconds'.
        """
        self.client = client

        self.chunk_length_seconds = os.environ.get("BUFFERING_CHUNK_LENGTH_SECONDS")
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = kwargs.get("chunk_length_seconds")
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get("BUFFERING_CHUNK_OFFSET_SECONDS")
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = kwargs.get("chunk_offset_seconds")
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.error_if_not_realtime = os.environ.get("ERROR_IF_NOT_REALTIME")
        if not self.error_if_not_realtime:
            self.error_if_not_realtime = kwargs.get("error_if_not_realtime", False)

        self.processing_flag = False
        self.start_time = None

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """
        Process audio chunks by checking their length and scheduling asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk length and, if so,
        it schedules asynchronous processing of the audio.

        Args:
            websocket (Websocket): The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        if self.client.connect_time is None and len(self.client.buffer) > 0:
            self.client.connect_time = time.time()
        if self.start_time is None:
            self.start_time = time.time()

        chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )
        if len(self.client.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                logging.warning(
                    "Error in realtime processing: tried processing a new chunk while the previous one was still being processed"
                )

            self.client.scratch_buffer += self.client.buffer
            self.client.buffer.clear()
            self.processing_flag = True
            # Schedule the processing in a separate task
            asyncio.create_task(
                self.process_audio_async(
                    websocket,
                    vad_pipeline,
                    asr_pipeline,
                    self.start_time,
                    self.client.last_start_time,
                )
            )
            self.start_time = None

    async def process_audio_async(
        self, websocket, vad_pipeline, asr_pipeline, start_time, default_start_time
    ):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity detection and transcription of
        the audio data. It sends the transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        start_process_time = time.time()
        start_transcribe_time = int(start_time - self.client.connect_time) + float(
            default_start_time
        )
        vad_results = await vad_pipeline.detect_activity(self.client)
        # logging.info(f"process_audio_async: vad_results: {vad_results}")
        if len(vad_results) == 0:
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.processing_flag = False
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds
        # logging.info(last_segment_should_end_before)
        if (
            vad_results[-1]["end"] < last_segment_should_end_before
            or last_segment_should_end_before > 2
        ):
            transcription = await asr_pipeline.transcribe(self.client)
            # logging.info(f"process_audio_async: transcription: {transcription}")
            if transcription is not None and "text" in transcription:
                self.start_time = time.time()
                processing_time = time.time() - start_process_time
                start_time_sec = start_transcribe_time
                end_time_sec = start_transcribe_time + transcription["duration"]

                # 轉換為指定輸出格式
                connection_id = getattr(self.client, "connection_id", None)
                if not connection_id:
                    try:
                        import uuid as _uuid

                        connection_id = str(_uuid.uuid4())
                    except Exception:
                        connection_id = ""

                payload = {
                    "id": connection_id,
                    "code": 200,
                    "message": "轉譯成功",
                    "result": [
                        {
                            "segment": 0,
                            "transcript": transcription.get("text", ""),
                            "final": 1,
                            "startTime": round(float(start_time_sec), 3),
                            "endTime": round(float(end_time_sec), 3),
                        }
                    ],
                }
                json_transcription = json.dumps(payload, ensure_ascii=False)
                # Starlette WebSocket 提供 send_text，websockets 提供 send
                try:
                    if hasattr(websocket, "send_text"):
                        await websocket.send_text(json_transcription)
                    else:
                        await websocket.send(json_transcription)
                except TypeError:
                    # 若底層期望 dict，則改為直接送 dict（部分接口支援）
                    try:
                        if hasattr(websocket, "send_json"):
                            await websocket.send_json(transcription)
                        else:
                            await websocket.send(json_transcription)
                    except Exception:
                        raise
                self.client.transcript.append(payload)
                logging.info(
                    f"process_audio: start_time: {start_time}, text: {transcription.get('text', '')}"
                )
            self.client.scratch_buffer.clear()
            self.client.increment_file_counter()

        self.processing_flag = False

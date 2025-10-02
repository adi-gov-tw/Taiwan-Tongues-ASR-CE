import os
import logging
from .vad_interface import VADInterface
from audio_utils import save_audio_to_file


class SimpleVAD(VADInterface):
    """
    Simple VAD implementation that assumes all audio is speech.
    """

    def __init__(self, **kwargs):
        """
        Initializes the simple VAD.

        Args:
            **kwargs: Additional arguments (ignored for simple VAD).
        """
        self.min_duration = kwargs.get("min_duration", 0.1)
        logging.info("SimpleVAD initialized - assuming all audio is speech")

    async def detect_activity(self, client):
        """
        Detects voice activity by assuming all audio is speech.

        Args:
            client: The client object with audio data.

        Returns:
            List: A list with a single segment representing the entire audio duration.
        """
        if len(client.scratch_buffer) == 0:
            return []

        # Calculate audio duration
        sample_rate = client.sampling_rate
        sample_width = client.samples_width
        audio_duration = len(client.scratch_buffer) / (sample_rate * sample_width)

        # If audio duration is too short, return empty
        if audio_duration < self.min_duration:
            return []

        # Return a single segment representing the entire audio
        return [{"start": 0.0, "end": audio_duration, "confidence": 1.0}]

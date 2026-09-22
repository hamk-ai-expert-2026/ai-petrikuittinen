"""Talk to GPT-Live using the default microphone and speaker on Windows."""

import asyncio
import base64
import os
import queue
import signal
import sys
from collections.abc import Callable
from typing import Any

import sounddevice as sd
from openai import AsyncOpenAI
from openai.types.live.session_config_param import SessionConfigParam


SAMPLE_RATE = 24_000
CHANNELS = 1
AUDIO_DTYPE = "int16"
BLOCK_SIZE = 960  # 40 ms at 24 kHz.
INPUT_QUEUE_SIZE = 32
OUTPUT_QUEUE_SIZE = 200


class AudioStreams:
    """Bridge sounddevice callbacks and the asyncio Live connection."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.input_audio: asyncio.Queue[bytes] = asyncio.Queue(INPUT_QUEUE_SIZE)
        self.output_audio: queue.Queue[bytes] = queue.Queue(OUTPUT_QUEUE_SIZE)
        self.pending_output = bytearray()
        self.input_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            callback=self._input_callback,
        )
        self.output_stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            callback=self._output_callback,
        )

    def start(self) -> None:
        self.output_stream.start()
        self.input_stream.start()

    def close(self) -> None:
        self.input_stream.stop()
        self.output_stream.stop()
        self.input_stream.close()
        self.output_stream.close()

    def _input_callback(
        self,
        indata: Any,
        _frames: int,
        _time: Any,
        status: Any,
    ) -> None:
        if status:
            print(f"Microphone status: {status}", file=sys.stderr)

        audio = bytes(indata)
        self.loop.call_soon_threadsafe(self._enqueue_input, audio)

    def _enqueue_input(self, audio: bytes) -> None:
        try:
            self.input_audio.put_nowait(audio)
        except asyncio.QueueFull:
            # Dropping an old microphone block is preferable to growing latency.
            pass

    def _output_callback(
        self,
        outdata: Any,
        frames: int,
        _time: Any,
        _status: Any,
    ) -> None:
        required_bytes = frames * CHANNELS * 2

        while len(self.pending_output) < required_bytes:
            try:
                self.pending_output.extend(self.output_audio.get_nowait())
            except queue.Empty:
                break

        outdata[:] = b"\x00" * required_bytes
        available_bytes = min(required_bytes, len(self.pending_output))
        if available_bytes:
            outdata[:available_bytes] = self.pending_output[:available_bytes]
            del self.pending_output[:available_bytes]

    def enqueue_output(self, audio: bytes) -> None:
        try:
            self.output_audio.put_nowait(audio)
            return
        except queue.Full:
            pass

        # Keep playback near real time if the speaker cannot keep up.
        try:
            self.output_audio.get_nowait()
            self.output_audio.put_nowait(audio)
        except queue.Empty:
            pass


async def send_microphone_audio(connection: Any, streams: AudioStreams) -> None:
    while True:
        audio = await streams.input_audio.get()
        encoded_audio = base64.b64encode(audio).decode("ascii")
        await connection.session.input_audio.append(audio=encoded_audio)


def install_signal_handler(
    loop: asyncio.AbstractEventLoop,
    stop_event: asyncio.Event,
) -> Callable[[], None]:
    def request_stop(_signal_number: int, _frame: Any) -> None:
        print("\nStopping after the current Live session...", file=sys.stderr)
        loop.call_soon_threadsafe(stop_event.set)

    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)

    def restore() -> None:
        signal.signal(signal.SIGINT, previous_handler)

    return restore


async def close_when_requested(connection: Any, stop_event: asyncio.Event) -> None:
    await stop_event.wait()
    await connection.session.close()


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    session: SessionConfigParam = {
        "model": "gpt-live-1",
        "instructions": (
            "You are a helpful voice assistant. Keep spoken answers concise."
        ),
        "audio": {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "output": {"voice": "marin"},
        },
        "delegation": {"type": "client"},
    }

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    restore_signal_handler = install_signal_handler(loop, stop_event)
    streams: AudioStreams | None = None
    sender: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None

    try:
        async with AsyncOpenAI() as client:
            async with client.live.connect() as connection:
                await connection.session.start(session=session)
                close_task = asyncio.create_task(
                    close_when_requested(connection, stop_event)
                )

                async for event in connection:
                    if event.type == "session.started":
                        streams = AudioStreams(loop)
                        streams.start()
                        sender = asyncio.create_task(
                            send_microphone_audio(connection, streams)
                        )
                        print(
                            "Live session ready. Speak normally; press Ctrl+C to stop.",
                            file=sys.stderr,
                        )
                        print(f"Session ID: {event.session.id}", file=sys.stderr)
                    elif event.type == "session.output_audio.delta":
                        if streams is not None:
                            streams.enqueue_output(base64.b64decode(event.delta))
                    elif event.type in {
                        "session.input_transcript.delta",
                        "session.output_transcript.delta",
                    }:
                        print(event.delta, end="", flush=True)
                    elif event.type == "session.closed":
                        print(
                            f"\nFinal audio usage: {event.usage.seconds:.1f} seconds",
                            file=sys.stderr,
                        )
                        break
                    elif event.type == "error":
                        print(f"Live error: {event.error.message}", file=sys.stderr)

    finally:
        if sender is not None:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        if close_task is not None:
            close_task.cancel()
            await asyncio.gather(close_task, return_exceptions=True)
        if streams is not None:
            streams.close()
        restore_signal_handler()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
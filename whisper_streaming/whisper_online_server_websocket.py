import argparse
import asyncio
import io
import json
import logging
import os
import re
import sys

import numpy as np
import soundfile
import websockets

from whisper_online import *

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=43007)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

# setting whisper object by args

SAMPLING_RATE = 16000

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size



# wraps socket and ASR object, and serves one client connection.
# next client should be served by a new instance of this object
class ServerProcessor:

    def __init__(self, online_asr_proc, min_chunk):
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None


    def format_message(self, resp_type, trace_id="some_trace_id", is_final=False, text=None):
        """
        {
            "resp_type": "RESULT",
            "trace_id": "567e8537-a89c-13c3-a882-826321939651",
            "segments": [
                {
                    "start_time": 100,
                    "end_time": 1500,
                    "is_final": False,
                    "result": {
                        "text": "第一句中间结果",
                        "word_info": [
                            {"start_time": 100, "end_time": 800, "word": "第一"},
                            {"start_time": 800, "end_time": 1000, "word": "句"},
                            {"start_time": 1000, "end_time": 1500, "word": "结果"},
                        ],
                        "score": 0.0,
                    },
                },
            ],
        }
        """

        assert resp_type in ["START", "RESULT", "END"]
        msg = {
            "resp_type": resp_type,
            "trace_id": trace_id,  # Replace with actual trace ID
        }


        if (resp_type == "RESULT") and (text is not None):
            segments = []
            if isinstance(text, str):
                text = [text]
            if isinstance(text, list):
                for text_i in text:
                    start_time, end_time, content = re.findall(r"(\d+\.?\d*) (\d+\.?\d*) (.*)", text_i)[0]

                    seg = {
                            "start_time": '{:.2f}'.format(float(start_time)),
                            "end_time": '{:.2f}'.format(float(end_time)),
                            "is_final": is_final,
                            "result": {
                                "text": content,
                                "word_info": [],
                                "score": 0.0,
                            },
                        }
                    segments.append(seg)

            msg["segments"] = segments
                
        return msg

    async def send_result(self, websocket, o, result_type):
        msg = self.format_output_transcript(o)
        if msg is not None:
            response = self.format_message(result_type, text=msg)
            await self.ws_send(websocket, json.dumps(response))

    async def ws_send(self, websocket, msg):
        try:
            await websocket.send(msg)
        except websockets.ConnectionClosed:
            logger.info("Connection closed, waiting for a new connection.")



# WebSocket server implementation
async def asr_server(websocket, path):
    # Initialize ASR processor
    proc = ServerProcessor(online, min_chunk)
    audio_buffer = []
    # Process messages from the client
    async for message in websocket:
        if isinstance(message, str):
            # If the message is a string, it may be a control message
            control_message = json.loads(message)

            if control_message["command"] == "START":
                # This is the start recognition message
                proc.online_asr_proc.init()

                # Send back status update
                start_message = proc.format_message('START')
                await proc.ws_send(websocket, json.dumps(start_message))

            elif control_message["command"] == "END":
                # This is the end recognition message
                o = proc.online_asr_proc.finish()
                await proc.send_result(websocket, o, result_type="RESULT")

                end_message = proc.format_message('END', is_final=True)
                await proc.ws_send(websocket, json.dumps(end_message))
        elif isinstance(message, bytes):
            # If the message is binary data, it may be audio data
            sf = soundfile.SoundFile(
                io.BytesIO(message),
                channels=1,
                endian="LITTLE",
                samplerate=SAMPLING_RATE,
                subtype="PCM_16",
                format="RAW",
            )
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)

            if sum(len(x) for x in audio_buffer) < proc.min_chunk * SAMPLING_RATE:
                # if less than min_chunk seconds are available
                audio_buffer.append(audio)

            if sum(len(x) for x in audio_buffer) >= proc.min_chunk * SAMPLING_RATE:
                # if min_chunk seconds are available
                audio_data = np.concatenate(audio_buffer)
                audio_buffer = []
                proc.online_asr_proc.insert_audio_chunk(audio_data)
                o = proc.online_asr_proc.process_iter()
                await proc.send_result(websocket, o, result_type="RESULT")

        else:
            # If the message is neither a string nor a binary data, it is an error
            error_message = proc.format_message('ERROR')
            await proc.ws_send(websocket, json.dumps(error_message))


# Start the WebSocket server
async def main():
    start_server = websockets.serve(asr_server, args.host, args.port)

    server = await start_server
    logger.info(f'Server started at ws://{args.host}:{args.port}')

    async with server:
        await server.wait_closed()

asyncio.run(main())
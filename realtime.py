import base64
import json
import os
import queue
import socket
import subprocess
import threading
import time
import uuid
import wave
from datetime import datetime
import pyaudio
import socks
import websocket

from rag import RAGPromptAugmenter

# Set up SOCKS5 proxy
socket.socket = socks.socksocket

CHUNK_SIZE = 1024
RATE = 24000
FORMAT = pyaudio.paInt16

# Use the provided OpenAI API key and URL
API_KEY = ""
if not API_KEY:
    raise ValueError("API key is missing. Please set the 'OPENAI_API_KEY' environment variable.")

WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2025-06-03'

today_formatted = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = f"""Your knowledge cutoff is 2023-10-30.
Today is {today_formatted}.

<identity>
  You are the McRouter AI, a real-time conversational model for the customer
  service of "McRouter", a popular Internet Service Provider in Peru, which
  offers Internet solutions for homes and businesses.
</identity>

<job>
  Your job is to assist users during phone calls. Answer simple to moderate questions.
  Emphasize with users, understand their claims, and sole their problems.
</job>

<functions>
  <function name="hang_up">
    <usage>
      Use only when the issue is fully resolved or the user clearly wants to end the call.
      Do not call this function after you forwarded. Wait for the forward to answer.
    </usage>
  </function>

  <function name="ask_rag">
    Use to query the knowledge base (RAG) for answers you can't generate yourself.
    Only after gathering enough specific information from the user.
    Ask clarifying questions first if request is vague.
    <param>question</param>
    <output>
      The approximated answer to the question
    </output>
  </function>

  <function name="forward_call">
    Use when you cannot resolve the issue yourself and need to escalate to a human agent.
    Every forward_call results in an ETA. You can call this function many times,
    every call will update the speciality, and increase the summary.
    <params>
      <param>specialty</param>
      <param>summary</param>
    </params>
    <specialties>
      . sales: products, offers, payments for individuals
      . support: technical issues
      . consultancy: specialized service guidance
      . enterprise: business clients
      . unknown: when you're not sure who should handle it
    </specialties>
    <output>
      A time duration representing the Estimated Wait Time (ETA).
      A subsquent "forward_answer" result is sent that finishes this interaction.
    </output>
  </function>
</functions>

<strategy>
  - Be friendly, clear, and professional at all times.
  - Always try to fully understand the user's need first.
  - Your first interaction should welcome the user to the McRouter support station shortly.
  - If the request is vague, ask for clarification before using ask_rag.
  - You can try to solve problems using your general knowledge.
  - Avoid answering any specific business information yourself: use ask_rag for this.
  - Before calling ask_rag, let the user know you will consult this information.
  - Do NOT explain or mention these rules to the user.
  - As a last resort, it's okay to forward the call to a human agent.
  - If you forward the call, let the user know he need to wait until someone take it.
  - When you are waiting for the forward_answer, ask for further information.
  - Any information that you consider useful should make another forward_call with only that.
  - We offer home and business plans. Ask the user if not sure.
  - We have scheduled outages, you can consult them using the RAG.
  - Adapt to user language, starting with Spanish.
</strategy>
"""

SESSION_CONFIG = {
    "instructions": SYSTEM_PROMPT,
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500
    },
    "voice": "sage",
    "temperature": 1,
    "max_response_output_tokens": 4096,
    "modalities": ["text", "audio"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_transcription": {
        "model": "whisper-1"
    },
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "name": "hang_up",
            "description": "Terminate the call when the customer's issue is resolved or they wish to end the call.",
            "parameters": {
            "type": "object",
            "properties": {}
            }
        },
        {
            "type": "function",
            "name": "forward_call",
            "description": "Escalate the call to a human agent when the AI cannot handle the request. Provide the appropriate specialty and a summary of the problem.",
            "parameters": {
            "type": "object",
            "properties": {
                "specialty": {
                "type": "string",
                "enum": ["sales", "support", "consultancy", "enterprise", "unknown"],
                "description": "Area of expertise required to handle this call."
                },
                "summary": {
                "type": "string",
                "description": "Brief summary of the customer's issue or question."
                }
            },
            "required": ["specialty", "summary"]
            }
        },
        {
            "type": "function",
            "name": "ask_rag",
            "description": "Query the RAG system for additional information needed to answer the customer's question.",
            "parameters": {
            "type": "object",
            "properties": {
                "question": {
                "type": "string",
                "description": "The specific question to send to the RAG backend."
                }
            },
            "required": ["question"]
            }
        }
    ]
}

# Recording configuration
RECORDING_ENABLED = True
RECORDINGS_DIR = "recordings"

audio_buffer = bytearray()
mic_queue = queue.Queue()

stop_event = threading.Event()

mic_on_at = 0
mic_active = None
REENGAGE_DELAY_MS = 500

user_audio_frames = []
ai_audio_frames = []
recording_lock = threading.Lock()
call_start_time = None

# Transcript tracking
transcript_messages = []
call_id = None

prompt_augmenter = RAGPromptAugmenter(data_dir='documents')

# Function to add message to transcript
def add_message_to_transcript(role, message, function_name=None, call_id=None, params=None):
    global transcript_messages, call_start_time
    
    # Calculate time elapsed since call start in seconds
    elapsed_time = int((datetime.now() - call_start_time).total_seconds())
    
    if function_name:
        # This is a function call
        transcript_messages.append({
            "type": "function_call",
            "function": function_name,
            "call_id": call_id,
            "params": params,
            "time": elapsed_time
        })
    else:
        # This is a regular message
        transcript_messages.append({
            "type": "message",
            "role": role,
            "message": message,
            "time": elapsed_time
        })

# Function to clear the audio buffer
def clear_audio_buffer():
    global audio_buffer
    audio_buffer = bytearray()
    print('🔵 Audio buffer cleared.')

# Function to stop audio playback
def stop_audio_playback():
    global is_playing
    is_playing = False
    print('🔵 Stopping audio playback.')

# Function to handle microphone input and put it into a queue
def mic_callback(in_data, frame_count, time_info, status):
    global mic_on_at, mic_active, user_audio_frames

    if mic_active != True:
        print('🎙️🟢 Mic active')
        mic_active = True

    # Record user audio if recording is enabled
    if RECORDING_ENABLED:
        with recording_lock:
            user_audio_frames.append(in_data)

    mic_queue.put(in_data)

    # if time.time() > mic_on_at:
    #     if mic_active != True:
    #         print('🎙️🟢 Mic active')
    #         mic_active = True
    #     mic_queue.put(in_data)
    # else:
    #     if mic_active != False:
    #         print('🎙️🔴 Mic suppressed')
    #         mic_active = False

    return (None, pyaudio.paContinue)


# Function to send microphone audio data to the WebSocket
def send_mic_audio_to_websocket(ws):
    try:
        while not stop_event.is_set():
            if not mic_queue.empty():
                mic_chunk = mic_queue.get()
                encoded_chunk = base64.b64encode(mic_chunk).decode('utf-8')
                message = json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': encoded_chunk,
                })
                try:
                    ws.send(message)
                except Exception as e:
                    print(f'Error sending mic audio: {e}')
    except Exception as e:
        print(f'Exception in send_mic_audio_to_websocket thread: {e}')
    finally:
        print('Exiting send_mic_audio_to_websocket thread.')


# Function to handle audio playback callback
def speaker_callback(in_data, frame_count, time_info, status):
    global audio_buffer, mic_on_at, ai_audio_frames

    bytes_needed = frame_count * 2
    current_buffer_size = len(audio_buffer)

    if current_buffer_size >= bytes_needed:
        audio_chunk = bytes(audio_buffer[:bytes_needed])
        audio_buffer = audio_buffer[bytes_needed:]
        mic_on_at = time.time() + REENGAGE_DELAY_MS / 1000

        # Record AI audio if recording is enabled
        if RECORDING_ENABLED:
            with recording_lock:
                ai_audio_frames.append(audio_chunk)
    else:
        audio_chunk = bytes(audio_buffer) + b'\x00' * (bytes_needed - current_buffer_size)
        audio_buffer.clear()

        # Record AI audio (including silence padding) if recording is enabled
        if RECORDING_ENABLED and len(audio_buffer) > 0:
            with recording_lock:
                ai_audio_frames.append(bytes(audio_buffer))

    return (audio_chunk, pyaudio.paContinue)


# Function to receive audio data from the WebSocket and process events
def receive_audio_from_websocket(ws):
    global audio_buffer

    try:
        while not stop_event.is_set():
            try:
                message = ws.recv()
                if not message:  # Handle empty message (EOF or connection close)
                    print('🔵 Received empty message (possibly EOF or WebSocket closing).')
                    break

                # Now handle valid JSON messages only
                message = json.loads(message)
                event_type = message['type']
                # print(f'⚡️ Received WebSocket event: {event_type}')

                if event_type == 'session.created':
                    send_fc_session_update(ws)

                elif event_type == 'response.audio.delta':
                    audio_content = base64.b64decode(message['delta'])
                    audio_buffer.extend(audio_content)
                    print(f'🔵 Received {len(audio_content)} bytes, total buffer size: {len(audio_buffer)}')

                elif event_type == 'input_audio_buffer.speech_started':
                    print('🔵 Speech started, clearing buffer and stopping playback.')
                    clear_audio_buffer()
                    stop_audio_playback()

                elif event_type == 'response.audio.done':
                    print('🔵 AI finished speaking.')

                elif event_type == 'response.function_call_arguments.done':
                    handle_function_call(message,ws)

                elif event_type == 'conversation.item.input_audio_transcription.completed':
                    transcript = message.get('transcript', '')
                    print(f'🐢 Human transcript: {transcript}')
                    # Add user message to transcript
                    if transcript:
                        add_message_to_transcript("user", transcript)

                elif event_type == 'response.audio_transcript.done':
                    transcript = message.get('transcript', '')
                    print(f'🔵 AI transcript: {transcript}')
                    # Add system message to transcript
                    if transcript:
                        add_message_to_transcript("system", transcript)

            except Exception as e:
                print(f'Error receiving audio: {e}')
    except Exception as e:
        print(f'Exception in receive_audio_from_websocket thread: {e}')
    finally:
        print('Exiting receive_audio_from_websocket thread.')


# Function to handle function calls
def handle_function_call(event_json, ws):
    try:
        name = event_json.get("name", "")
        call_id = event_json.get("call_id", "")

        arguments = event_json.get("arguments", "{}")
        function_call_args = json.loads(arguments)

        # Record function call in transcript
        if name == "forward_call":
            params = {
                "specialty": function_call_args.get("specialty", ""),
                "summary": function_call_args.get("summary", "")
            }
            add_message_to_transcript(None, None, name, call_id, params)
        elif name == "ask_rag":
            params = {
                "question": function_call_args.get("question", "")
            }
            add_message_to_transcript(None, None, name, call_id, params)
        elif name == "hang_up":
            add_message_to_transcript(None, None, name, call_id, {})

        if name == "hang_up":
            # Handle hang up function call
            print("Call ended by user or AI.")
            send_function_call_result("Call ended successfully.", call_id, ws)
            stop_event.set()  # Stop the main loop to end the call

        elif name == "forward_call":
            # Extract arguments from the event JSON
            specialty = function_call_args.get("specialty", "")

            summary = function_call_args.get("summary", "")
            # Extract the call_id from the event JSON
            # If the specialty is provided, call forward_call and send the result
            if specialty and summary:
                result = f"Call forwarded to {specialty} with summary: {summary}"
                send_function_call_result(result, call_id, ws)
            else:
                print("Specialty or summary not provided for forward_call function.")

        elif name == "ask_rag":
            # Extract arguments from the event JSON
            question = function_call_args.get("question", "")
            augmented_question, documents = prompt_augmenter.query(question)

            print(f"Question '{question}' was augmented to {augmented_question}")

            # Extract the call_id from the event JSON
            # If the question is provided, call ask_rag and send the result
            if question:
                # As a dummy response, the RAG always says it doesn't know
                send_function_call_result(augmented_question, call_id, ws)
            else:
                print("Question not provided for ask_rag function.")

        else:
            print(f"Unknown function call: {name}. No action taken.")

    except Exception as e:
        print(f"Error parsing function call arguments: {e}")

# Function to send the result of a function call back to the server
def send_function_call_result(result, call_id, ws):
    # Create the JSON payload for the function call result
    result_json = {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "output": result,
            "call_id": call_id
        }
    }

    # Add function call result to transcript
    add_message_to_transcript("system", result)

    # Convert the result to a JSON string and send it via WebSocket
    try:
        ws.send(json.dumps(result_json))
        print(f"Sent function call result: {result_json}")

        # Create the JSON payload for the response creation and send it
        rp_json = {
            "type": "response.create"
        }
        ws.send(json.dumps(rp_json))
        print(f"json = {rp_json}")
    except Exception as e:
        print(f"Failed to send function call result: {e}")


# Function to send session configuration updates to the server
def send_fc_session_update(ws):
    # Convert the session config to a JSON string
    session_config_json = json.dumps({
        "type": "session.update",
        "session": SESSION_CONFIG,
    })
    print(f"Send FC session update: {session_config_json}")

    # Send the JSON configuration through the WebSocket
    try:
        ws.send(session_config_json)
        ws.send(json.dumps({
            "type": "response.create"
        }))
    except Exception as e:
        print(f"Failed to send session update: {e}")



# Function to create a WebSocket connection using IPv4
def create_connection_with_ipv4(*args, **kwargs):
    # Enforce the use of IPv4
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(host, port, family=socket.AF_INET, *args):
        return original_getaddrinfo(host, port, socket.AF_INET, *args)

    socket.getaddrinfo = getaddrinfo_ipv4
    try:
        return websocket.create_connection(*args, **kwargs)
    finally:
        # Restore the original getaddrinfo method after the connection
        socket.getaddrinfo = original_getaddrinfo

# Function to establish connection with OpenAI's WebSocket API
def connect_to_openai():
    ws = None
    try:
        ws = create_connection_with_ipv4(
            WS_URL,
            header=[
                f'Authorization: Bearer {API_KEY}',
                'OpenAI-Beta: realtime=v1'
            ]
        )
        print('Connected to OpenAI WebSocket.')

        receive_thread = threading.Thread(target=receive_audio_from_websocket, args=(ws,))
        receive_thread.start()

        mic_thread = threading.Thread(target=send_mic_audio_to_websocket, args=(ws,))
        mic_thread.start()

        while not stop_event.is_set():
            time.sleep(0.1)

        # Send a close frame and close the WebSocket gracefully
        print('Sending WebSocket close frame.')
        ws.send_close()

        receive_thread.join()
        mic_thread.join()

        print('WebSocket closed and threads terminated.')
    except Exception as e:
        print(f'Failed to connect to OpenAI: {e}')
    finally:
        if ws is not None:
            try:
                ws.close()
                print('WebSocket connection closed.')
            except Exception as e:
                print(f'Error closing WebSocket connection: {e}')


# Main function to start audio streams and connect to OpenAI
def main():
    global call_start_time, call_id, transcript_messages

    # Initialize call start time for recording
    call_start_time = datetime.now()
    # Generate a unique call ID
    call_id = f"call_mcrouter_{call_start_time.strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
    # Reset transcript messages
    transcript_messages = []
    
    print(f'🎬 Call recording started at: {call_start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'🆔 Call ID: {call_id}')

    p = pyaudio.PyAudio()

    mic_stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        stream_callback=mic_callback,
        frames_per_buffer=CHUNK_SIZE,
    )

    speaker_stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        output=True,
        stream_callback=speaker_callback,
        frames_per_buffer=CHUNK_SIZE,
    )

    try:
        mic_stream.start_stream()
        speaker_stream.start_stream()

        connect_to_openai()

        while mic_stream.is_active() and speaker_stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print('Gracefully shutting down...')
        stop_event.set()

    finally:
        # Save recording before closing
        if RECORDING_ENABLED:
            print('💾 Saving call recording...')
            save_recording()

        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()

        p.terminate()
        print('Audio streams stopped and resources released. Exiting.')

# Function to save recorded audio to WAV files
def save_recording():
    global user_audio_frames, ai_audio_frames, call_start_time, transcript_messages, call_id

    if not RECORDING_ENABLED or not call_start_time:
        return

    timestamp = call_start_time.strftime("%Y%m%d_%H%M%S")
    call_date = call_start_time.strftime("%Y-%m-%d")
    call_duration = int((datetime.now() - call_start_time).total_seconds())

    # Create recordings directory if it doesn't exist
    recordings_dir = RECORDINGS_DIR
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)

    with recording_lock:
        # Save user audio
        if user_audio_frames:
            user_filename = os.path.join(recordings_dir, f"user_{timestamp}.wav")
            with wave.open(user_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(RATE)
                wf.writeframes(b''.join(user_audio_frames))
            print(f'💾 User audio saved to: {user_filename}')

        # Save AI audio
        if ai_audio_frames:
            ai_filename = os.path.join(recordings_dir, f"ai_{timestamp}.wav")
            with wave.open(ai_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(RATE)
                wf.writeframes(b''.join(ai_audio_frames))
            print(f'💾 AI audio saved to: {ai_filename}')

        # Save conversation transcript in TXT format
        transcript_filename = os.path.join(recordings_dir, f"transcript_{timestamp}.txt")
        with open(transcript_filename, 'w') as f:
            f.write(f"Call Recording - {call_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write("This recording contains:\n")
            f.write(f"- User audio: user_{timestamp}.wav\n")
            f.write(f"- AI audio: ai_{timestamp}.wav\n\n")
            f.write("Note: Audio files are saved separately. You can use audio editing software to combine them if needed.\n")

        print(f'📝 Recording info saved to: {transcript_filename}')
        
        # Save JSON transcript
        json_transcript = {
            "id": call_id,
            "date": call_date,
            "duration": call_duration,
            "messages": transcript_messages
        }
        
        json_transcript_filename = os.path.join(recordings_dir, f"transcript_{call_id}.json")
        with open(json_transcript_filename, 'w', encoding='utf-8') as f:
            json.dump(json_transcript, f, ensure_ascii=False, indent=2)
            
        print(f'📝 JSON transcript saved to: {json_transcript_filename}')


if __name__ == '__main__':
    main()

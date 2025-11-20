# web_server.py
from flask import Flask, Response, redirect, url_for
import time
import os
import signal
import shared_state  # Import our shared state module

app = Flask(__name__)
PORT = 2131

def generate_frames():
    """Generator function to stream pre-encoded JPEG frames."""
    while True:
        with shared_state.frame_lock:
            if shared_state.jpeg_frame is None:
                # If the first frame isn't ready, wait
                time.sleep(0.1)
                continue
            frame_to_send = shared_state.jpeg_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
        
        time.sleep(0.03) # Prevent hogging CPU

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """HTML page with controls."""
    with shared_state.recording_lock:
        rec_status = shared_state.is_recording
    
    record_button_text = "Stop Recording" if rec_status else "Start Recording"

    return f"""
    <html>
    <head><title>Fall Detection Stream</title></head>
    <body>
        <h1>Fall Detection Stream</h1>
        <p>Live feed from Raspberry Pi (Port {PORT})</p>
        <img src="/video_feed" width="640" height="480">
        <hr>
        <form action="/toggle_record" method="post" style="display:inline-block;">
            <input type="submit" value="{record_button_text}" style="padding: 10px; font-size: 16px;">
        </form>
        <form action="/shutdown" method="post" style="display:inline-block; margin-left: 20px;">
            <input type="submit" value="Exit Program" style="padding: 10px; font-size: 16px; background-color: #ffcccc;">
        </form>
    </body>
    </html>
    """

@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    """Flips the is_recording boolean. The main loop handles the rest."""
    with shared_state.recording_lock:
        shared_state.is_recording = not shared_state.is_recording
        if shared_state.is_recording:
            print("Web request: START recording")
        else:
            print("Web request: STOP recording")
    
    return redirect(url_for('index'))

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Stops the detection loop and then kills the script."""
    print("Shutdown request received via web...")
    shared_state.running = False # Signal detection thread to stop
    time.sleep(1.0) # Give the thread time to clean up
    os.kill(os.getpid(), signal.SIGINT) # Send Ctrl+C to self
    return "Server is shutting down..."

def start_server():
    """Starts the Flask web server."""
    print(f"Starting web server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
from flask import Flask, Response, redirect, url_for, render_template, jsonify
import time
import os
import signal
import shared_state

app = Flask(__name__)
PORT = 2131

prev_time = None

prev_frame_count = 0

def generate_frames():
    """Generator function to stream pre-encoded JPEG frames."""
    global prev_time
    while True:
        with shared_state.frame_lock:
            if prev_time == None:
                prev_time = time.time()
            d_time = (time.time() - prev_time) * 1000
            if shared_state.frame_count[0] - prev_frame_count == 0 or d_time < 6.6  :
                continue
            prev_time = time.time()
            frame_to_send = shared_state.jpeg_frame
        # print(shared_state.frame_count[0])
            
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
    

    return render_template('index.html', port=PORT)

@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    """Flips the is_recording boolean. The main loop handles the rest."""
    with shared_state.recording_lock:
        shared_state.is_recording = not shared_state.is_recording
        if shared_state.is_recording:
            print("Web request: START recording")
            return jsonify({"is_recording": True})
        else:
            print("Web request: STOP recording")
            return jsonify({"is_recording": False})
    
    

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
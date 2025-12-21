import requests
import shared_state
import time

def send_msg():
    try:
        response = requests.get("https://n8n.siddarthnaik.online/webhook/247a06cb-70a5-4091-bfdb-e2c5cccfd281", timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"Alert request failed: {e}")

def alert():
    while shared_state.running:
        if shared_state.fall_detected and (not shared_state.message_sent):
            send_msg()
            shared_state.message_sent = True
        time.sleep(0.1)  

if __name__ == "__main__":
    send_msg()
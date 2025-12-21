import requests
import shared_state
import time

def send_msg():
    response = requests.get("https://n8n.siddarthnaik.online/webhook/247a06cb-70a5-4091-bfdb-e2c5cccfd281")

def alert():
    while shared_state.running:
        print("in msg")
        time.sleep(1)
        print(shared_state.fall_detected, shared_state.message_sent)
        if shared_state.fall_detected and (not shared_state.message_sent):
            print("again in")
            send_msg()
            shared_state.message_sent = True

if __name__ == "__main__":
    send_msg()
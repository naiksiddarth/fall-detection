import requests
def send_msg():
    response = requests.get("https://n8n.siddarthnaik.online/webhook/247a06cb-70a5-4091-bfdb-e2c5cccfd281")

if __name__ == "__main__":
    send_msg()
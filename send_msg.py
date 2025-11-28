import requests
def send_msg():
    response = requests.get("https://n8n.siddarthnaik.online/webhook/e4f307cc-ce59-4172-a80d-7e7fbaf4b2e6")


if __name__ == "__main__":
    send_msg()
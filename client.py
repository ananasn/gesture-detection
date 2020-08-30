import time
import requests
import random

while True:
    payload = {
        'entity': random.choice(('voice', 'gesture')),
        'label': random.choice(('nya', 'desu', 'kawaii neko'))
    }
    r = requests.get('http://127.0.0.1:8080/detect', params=payload)
    time.sleep(1)

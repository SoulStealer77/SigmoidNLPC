import datetime
import requests
import json
from pprint import pprint

api_key = "288af201969c4233798017e17354af37"
lat = "48.208176"
lon = "16.373819"
url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (lat, lon, api_key)

response = requests.get(url)
info = response.json()
#data = info['temp']


pprint(info)

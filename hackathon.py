try:
    import requests
except ImportError:
    import os
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'ensurepip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
    import requests
import json

from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5',
'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': '8730b816-0710-4ed2-958b-514e1befe79e',
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  #print(data)

  coins = data['data']

  for x in coins :
    print (x['symbol'], x['quote']['USD']['price'])

except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)
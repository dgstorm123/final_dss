import requests
from bs4 import BeautifulSoup
import pandas as pd



# copy url 
url = 'http://......'


# lấy nội dung html 
req = requests.get(url)
content = BeautifulSoup(req.content,'html.parser')
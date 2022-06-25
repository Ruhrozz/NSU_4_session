import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


url = 'https://yarcheplus.ru/catalog/limonady-gazirovannye-napitki-161'
headers = {'User-Agent': 'Mozilla/5.0 (Linux; arm_64; Android 10; HRY-LX1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.141 YaBrowser/22.3.4.86.00 SA/3 Mobile Safari/537.36'}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

items = soup.find_all('a', class_="g2mGXj5-x")
items = [item.text.replace(" ", ' ') for item in items]
prices = soup.find_all('div', class_="b34tOzx2Q")
prices = [price.text.replace(" ", ' ') for price in prices]

res = np.asarray([items, prices])
df = pd.DataFrame(res.T, columns=["Product", "Price"])
df.to_csv("parse.csv")





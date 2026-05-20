import pandas as pd
import requests

url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage#Combinations_of_matches_in_the_round_of_32"

# 1. Define a User-Agent to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 2. Get the page content with the headers
response = requests.get(url, headers=headers)

# 3. Pass the HTML text to read_html
tables = pd.read_html(response.text)

df = tables[0]

df.drop(['Unnamed: 13'], inplace=True, axis=1)
df.to_csv('third_place_combinations.csv', index=False)
import requests
import re
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Function to clean text
def preprocess(text):
    token = word_tokenize(text)
    token = [text.lower() for text in token]
    token = [text for text in token if text.isalpha()]
    token = [text for text in token if not text in stopwords.words("english")]
    lemma = WordNetLemmatizer()
    token = [lemma.lemmatize(text) for text in token]
    return token

# Pass the home page to the BeautifulSoup() API
page = requests.get("https://dragonball.fandom.com/wiki/Dragon_Ball_Z")
soup = BeautifulSoup(page.content, "html.parser")

# Get all the links (which are stored inside an ordered list)
links = soup.find("div", "mw-parser-output").find_all("ol")[1]
str_link = str(links)
pat = re.compile('href="/wiki\\S*')
links = pat.findall(str_link)
links = [re.sub('href="',"",link) for link in links]
links = [re.sub('"',"",link) for link in links]
len(links)

# Get all paragraph from each link
story = list()

for link in links:
    page = requests.get("https://dragonball.fandom.com" + link)
    soup = BeautifulSoup(page.content, "html.parser")
    text = [tag.text for tag in soup.find("div", class_="mw-parser-output").find_all("p")]
    text = " ".join(text)
    story.append(text)

# Clean Text
text_clean = [preprocess(text) for text in story]
ta = [" ".join(text) for text in text_clean]

# Save Data
clean_text = pd.DataFrame({"text" : ta})
clean_text.to_csv("data_input/dbz.csv", index=False)
# Yamareco Flower Finder
#ヤマレコの山行記録から写真を取得し、OpenAI CLIPを用いて花写真を抽出します。
import os
import csv
from PIL import Image
import torch
import clip
import time
import requests
from bs4 import BeautifulSoup
import io
import re

response = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0"
    }
)

html = response.text
urls = re.findall(
     r'https://yamareco\.info/modules/yamareco/upimg/"ここにURL/写真のURL"/[A-Za-z0-9]+\.JPG',
    html
)
urls = list(set(urls))

print("候補URL数 =", len(urls))
for u in urls[:20]:
    print(u)


# ===== 設定 =====               
#OUTPUT_CSVの場所はpythonプログラムの入っているフォルダに出力されます
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(TARGET_DIR, "flower_photos_yamareco.csv")

# ===== CLIP読み込み =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

labels = [
    "a photo containing flowers",
    "a photo without flowers",
    "a snowy mountain landscape",
    "a mountain landscape without flowers",
    "a person",
    "a food"
]

text = clip.tokenize(labels).to(device)
results = []

start_time = time.time()
image_count = 0

# ===== ヤマレコ画像探索 =====
for src in urls:

    image_count += 1
    try:
        response = requests.get(src, timeout=10)
        image = preprocess(
            Image.open(
                io.BytesIO(response.content)
            ).convert("RGB")
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(image, text)
            probs = logits.softmax(
                dim=-1
            ).cpu().numpy()[0]
        flower_score = probs[0]

        if flower_score > 0.8:
            results.append([
                src,
                round(float(flower_score), 4)
            ])
            print(
                "FLOWER",
                round(float(flower_score), 3),
                src
            )

    except Exception as e:
        print("ERROR:", src, e)

# ===== CSV出力 =====
with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "record_id",
        "image_url",
        "flower_score"
    ])
    writer.writerows(results)

print(f"{len(results)} files exported.")
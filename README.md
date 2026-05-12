# ASL Recognition System

## Project Overview

This project focuses on American Sign Language (ASL) alphabet recognition using deep learning and transfer learning techniques. The system classifies static hand gesture images into ASL letters and special symbols, and supports real-time gesture recognition using a webcam.

Multiple CNN architectures were implemented, trained, and compared to evaluate their performance in terms of accuracy, robustness, and generalization. The project also integrates model explainability (Grad-CAM), Arabic translation via NLLB, Arabic Sign Language output, and a graphical user interface (GUI) to demonstrate results interactively.

---

## Project Goals

- Classify ASL alphabet hand gestures from images and live video
- Compare multiple CNN architectures using transfer learning
- Improve robustness to lighting, background, and pose variations
- Implement real-time gesture recognition using a webcam
- Provide visual explanations of model predictions using Grad-CAM
- Translate recognized English words to Arabic using NLLB
- Display Arabic Sign Language representation of translated words
- Build a simple and user-friendly GUI for interaction

---

## Dataset

- **Name:** ASL Alphabet Dataset
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- **Description:**
  - RGB images representing American Sign Language letters (A–Z) and special classes (space, delete, nothing)
  - Large number of samples per class
  - Variations in hand orientation, scale, and background

---

## Data Preprocessing & Augmentation

To enhance model generalization and reduce overfitting, the following techniques were applied:

- Image resizing to a fixed input size
- Pixel normalization
- Random rotation
- Horizontal flipping
- Random zoom and shift
- Brightness and contrast adjustments

These techniques help the models handle real-world variations in lighting and hand positioning.

---

## Implemented Models

The following deep learning models were implemented and evaluated:

| Model | Accuracy |
|-------|----------|
| InceptionV3 | 97% |
| ResNet50 | 96% |
| EfficientNetB0 | 98.70% |

All models were trained on the same dataset split and evaluated using identical metrics to ensure fair comparison.

---

## Results & Discussion

- **InceptionV3** achieved the best overall performance, with the highest accuracy, precision, and recall, and the cleanest confusion matrix with minimal misclassifications.
- **ResNet50** showed strong and stable performance, with very good generalization across all ASL classes.
- **EfficientNetB0** achieved competitive results while maintaining high efficiency, making it suitable for lightweight or resource-constrained deployments.

Overall, the results demonstrate the effectiveness of transfer learning and data augmentation for ASL gesture recognition. Based on quantitative metrics and confusion matrix analysis, InceptionV3 was selected as the final deployment model.

### ResNet50
![ResNet](https://github.com/user-attachments/assets/7d349bda-73ca-4e19-acbe-59e5fc8bdcfc)

### InceptionV3
![Inception](https://github.com/user-attachments/assets/4bac77c2-efd3-4e52-a8af-43d7c269bf69)

### EfficientNetB0
![EfficientNet](https://github.com/user-attachments/assets/360e0486-d212-46b6-8512-78833aa4cdb3)

### Models Comparison
![Comparison 1](https://github.com/user-attachments/assets/c9106e4b-0d2e-4c35-9fb8-2a8ca51384ee)

![Comparison 2](https://github.com/user-attachments/assets/eef983b4-150e-4e69-bef2-0334254ae8be)

---

## Real-Time Gesture Recognition

A real-time gesture recognition system was implemented using a webcam:

- Captures live video frames
- Detects hand region using MediaPipe
- Applies preprocessing in real time
- Performs model inference on each frame
- Displays predicted class and confidence on screen
- Hold-to-add logic: hold a sign for 2 seconds to add the letter to the sentence
- Progress bar overlay shows hold progress

Notebook: `web-cam.ipynb`
Standalone script: `camera.py`

---

## Model Explainability

To improve transparency and trust in the system, Grad-CAM was applied to visualize the regions of the image that contribute most to the model's predictions. These heatmaps confirm that the models focus primarily on the hand and finger regions, which aligns with human intuition.

---

## Arabic Translation & Arabic Sign Language Output

Recognized ASL words are translated to Arabic using the **NLLB-200** multilingual translation model (`facebook/nllb-200-distilled-600M`).

The translation API is hosted on a **Kaggle notebook** (GPU) and exposed via **ngrok**, allowing the local Streamlit app to call it over HTTP. The ngrok tunnel creates a public HTTPS URL that proxies requests to the FastAPI server running inside the Kaggle session.

After translation, the Arabic word is broken down letter by letter and displayed with the corresponding **Arabic Sign Language** image for each letter, shown right-to-left.

> **Note:** The ngrok URL changes every Kaggle session. Update `NGROK_URL` in `app.py` each time you restart the notebook.

---

## GUI

A Streamlit-based GUI was developed with three main features:

1. **Camera mode** — launches `camera.py` in a separate process, captures ASL letters via webcam, saves the sentence, then translates and displays Arabic sign images
2. **Multi-image upload** — upload multiple ASL letter images in order, get the predicted word, translate to Arabic, display Arabic signs
3. **Single image + Grad-CAM** — upload one ASL image, see top 3 predictions and a Grad-CAM heatmap

---

## Project Structure

```
project/
├── app.py                  ← Streamlit GUI
├── camera.py               ← Standalone webcam ASL detection
├── class_names.json        ← ASL class labels
├── sentence.txt            ← Written by camera.py, read by app.py
├── models/
│   ├── inception.h5        ← InceptionV3 trained model
│   ├── resnet.h5           ← ResNet50 trained model
│   ├── efficientnet.pb     ← EfficientNet exported model
│   └── variables/          ← EfficientNet variables folder
└── data/
    └── arabic/             ← Arabic sign language images (ا.jpg, ب.jpeg ...)
```

---

## Requirements

### Streamlit environment (runs `app.py`)
```
streamlit
tensorflow
opencv-python
pillow
numpy
requests
```

### Camera environment — `mp` conda env (runs `camera.py`)
```
tensorflow
mediapipe
opencv-python
numpy
```

### Kaggle notebook (runs NLLB translation API)
```
transformers
fastapi
uvicorn
pyngrok
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Setup Instructions

### 1. Configure paths in `app.py`

Edit the paths block at the top of `app.py`:

```python
CLASS_NAMES_PATH  = "class_names.json"
INCEPTION_PATH    = "models/inception.h5"
RESNET_PATH       = "models/resnet.h5"
EFFICIENTNET_PB   = "models/efficientnet.pb"
ARABIC_SIGNS_DIR  = "data/arabic"
NGROK_URL         = "https://your-url.ngrok-free.app"   # update every Kaggle session
API_KEY           = "your-secret-key"
MP_PYTHON         = r"C:\path\to\envs\mp\python.exe"
```

### 2. Configure paths in `camera.py`

```python
MODEL_PATH     = "best_inception_model.h5"
CAMERA_INDEX   = 0      # change if not using default webcam
HOLD_TIME      = 2.0    # seconds to hold a sign before letter is added
CONF_THRESHOLD = 0.7    # minimum confidence to accept a prediction
```

### 3. Start the NLLB translation API on Kaggle

Run this in a Kaggle notebook (GPU recommended):

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI, Request, HTTPException
import uvicorn, threading, time, socket
from pyngrok import ngrok, conf

API_KEY     = "your-secret-key"    # must match app.py
NGROK_TOKEN = "your-ngrok-token"   # from dashboard.ngrok.com

model_name = "facebook/nllb-200-distilled-600M"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

def translate(text, target_lang="arb_Arab"):
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=512
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

app = FastAPI()

@app.post("/translate")
async def trans(req: Request):
    if req.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    data = await req.json()
    return {"response": translate(data.get("english", ""))}

def free_port():
    s = socket.socket(); s.bind(('', 0))
    port = s.getsockname()[1]; s.close(); return port

port = free_port()
conf.get_default().auth_token = NGROK_TOKEN
public_url = ngrok.connect(port).public_url
print(public_url)   # <-- copy this into NGROK_URL in app.py

def run(): uvicorn.run(app, host="0.0.0.0", port=port)
threading.Thread(target=run, daemon=True).start()
time.sleep(1)
```

Copy the printed URL into `NGROK_URL` in `app.py`. **This URL changes every Kaggle session.**

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## How to Use

### Camera mode
1. Click **📷 Open Camera** in the sidebar
2. Show ASL letters to the webcam — hold each sign for 2 seconds to add the letter
3. Press **Q** to quit — sentence is saved to `sentence.txt`
4. Back in the browser, click **🌐 Translate to Arabic**
5. Arabic translation and Arabic sign images appear below

### Multi-image upload
1. Under **📤 Upload Multiple ASL Images**, upload images in letter order
2. Each image is classified — letter and confidence shown under thumbnail
3. Click **🌐 Translate uploaded word**

### Single image + Grad-CAM
1. Under **🖼️ Single Image**, upload one ASL hand image
2. Top 3 predictions shown with confidence percentages
3. Grad-CAM heatmap highlights what the model focused on

---

## Camera Controls

| Key | Action |
|-----|--------|
| Q | Quit and save sentence |
| C | Clear sentence |
| B | Backspace (delete last letter) |

---

## Arabic Sign Images

Name each file after its Arabic letter: `ا.jpg`, `ب.jpeg`, `ت.jpg`, etc. Mixed `.jpg` and `.jpeg` extensions are supported.

Letters automatically normalized before lookup:
- `أ إ آ` → `ا`
- `ى` → `ي`
- `ة` → `ه`

---

## Notes

- The ngrok URL expires when the Kaggle session ends — update `NGROK_URL` in `app.py` each time
- `camera.py` must run in the `mp` conda environment where mediapipe is installed
- The Streamlit app launches `camera.py` using the hardcoded `MP_PYTHON` path — update this to match your system
- NLLB language code for Modern Standard Arabic: `arb_Arab` — use `arz_Arab` for Egyptian Arabic

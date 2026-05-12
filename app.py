# ============================================================
# PATHS — edit these to match your setup
# ============================================================
CLASS_NAMES_PATH  = "class_names.json"
INCEPTION_PATH    = "models/inception.h5"
RESNET_PATH       = "models/resnet.h5"
EFFICIENTNET_PB   = "models/efficientnet.pb"
ARABIC_SIGNS_DIR  = "data/arabic"
NGROK_URL         = "https://26ee-136-107-38-156.ngrok-free.app"
API_KEY           = "secret123"
MP_PYTHON         = r"C:\Users\omarg\anaconda3\envs\mp\python.exe"
# ============================================================

import streamlit as st
import numpy as np
import cv2
import json
import shutil
import subprocess
import os
import re
import unicodedata
import requests
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# =====================================================================
# PAGE SETUP
# =====================================================================
st.set_page_config(page_title="ASL Recognition", layout="centered")
st.title("🤟 ASL Alphabet Recognition")

# =====================================================================
# CLASS NAMES
# =====================================================================
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# =====================================================================
# MODEL LOADER
# =====================================================================
class _SavedModelPredictWrapper:
    def __init__(self, loaded, fn, input_dtype, input_key):
        self._loaded, self._fn = loaded, fn
        self.input_dtype, self.input_key = input_dtype, input_key

    def predict(self, x):
        x_tf = tf.convert_to_tensor(x)
        if hasattr(self._fn, "structured_input_signature"):
            _, kwargs = self._fn.structured_input_signature
            outputs = self._fn(**{next(iter(kwargs)): x_tf}) if kwargs else self._fn(x_tf)
        else:
            outputs = self._fn(x_tf)
        if isinstance(outputs, dict):            outputs = next(iter(outputs.values()))
        elif isinstance(outputs, (list, tuple)): outputs = outputs[0]
        return outputs.numpy()


def _try_load_efficientnet_pb():
    export_dir  = Path(EFFICIENTNET_PB).parent
    pb_src      = Path(EFFICIENTNET_PB)
    variables   = export_dir / "variables"
    if not (pb_src.exists() and variables.is_dir()):
        return None
    pb_expected = export_dir / "saved_model.pb"
    if not pb_expected.exists():
        shutil.copyfile(pb_src, pb_expected)
    try:
        loaded = tf.saved_model.load(str(export_dir))
        fn = loaded.signatures.get("serving_default") or getattr(loaded, "__call__", None)
        if fn is None:
            return None
        input_dtype = input_key = None
        if hasattr(fn, "structured_input_signature"):
            _, kwargs = fn.structured_input_signature
            if kwargs:
                input_key   = next(iter(kwargs))
                input_dtype = getattr(kwargs[input_key], "dtype", None)
        return _SavedModelPredictWrapper(loaded, fn, input_dtype, input_key)
    except Exception as e:
        st.warning(f"EfficientNet load failed: {e}")
        return None


@st.cache_resource
def load_selected_model(name):
    if name == "InceptionV3": return load_model(INCEPTION_PATH)
    if name == "ResNet50":    return load_model(RESNET_PATH)
    m = _try_load_efficientnet_pb()
    if m is None:
        st.error("EfficientNet not found. Check EFFICIENTNET_PB path.")
        st.stop()
    return m


# =====================================================================
# GRAD-CAM
# =====================================================================
def _iter_layers(layer):
    yield layer
    if isinstance(layer, tf.keras.Model):
        for sub in layer.layers:
            yield from _iter_layers(sub)

def _find_layer(model, name):
    for l in _iter_layers(model):
        if getattr(l, "name", None) == name:
            return l
    return None

def _last_conv(model):
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D,
                  tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv2DTranspose)
    for l in reversed(list(_iter_layers(model))):
        if isinstance(l, conv_types):
            return l
    return None

def make_gradcam_heatmap(img_array, model, conv_layer_name, pred_index=None):
    conv_layer = _find_layer(model, conv_layer_name) or _last_conv(model)
    if conv_layer is None:
        raise ValueError("No Conv2D layer found for Grad-CAM.")
    grad_model = tf.keras.models.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if isinstance(preds, (list, tuple)): preds = preds[0]
        preds = tf.convert_to_tensor(preds)
        if preds.shape.rank == 1: preds = preds[tf.newaxis, :]
        tape.watch(conv_out)
        idx      = tf.cast(pred_index if pred_index is not None else tf.argmax(preds[0]), tf.int32)
        class_ch = preds[:, idx]
    grads = tape.gradient(class_ch, conv_out)
    if grads is None:
        raise ValueError("Grad-CAM gradients are None.")
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    d       = tf.reduce_max(heatmap)
    heatmap = tf.where(d > 0, heatmap / d, tf.zeros_like(heatmap))
    return heatmap.numpy()

def overlay_gradcam(img_rgb, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0])))
    blended = cv2.addWeighted(
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1 - alpha,
        cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


LAST_CONV_MAP = {
    "InceptionV3":  "mixed10",
    "ResNet50":     "conv5_block3_out",
    "EfficientNet": "top_conv",
}

# =====================================================================
# ARABIC SIGN DISPLAY HELPERS
# =====================================================================
def find_letter_image(letter):
    for ext in ("jpg", "jpeg", "png"):
        p = Path(ARABIC_SIGNS_DIR) / f"{letter}.{ext}"
        if p.exists():
            return str(p)
    return None

def display_arabic_signs(arabic_text):
    """Show each Arabic letter as a sign image, right-to-left."""
    cleaned = "".join(c for c in arabic_text if not unicodedata.combining(c))
    cleaned = (cleaned.replace("أ", "ا").replace("إ", "ا")
                      .replace("آ", "ا").replace("ى", "ي")
                      .replace("ة", "ه"))
    letters = [c for c in cleaned if re.match(r"[\u0621-\u064A]", c)]
    letters_rtl = list(reversed(letters))

    if not letters_rtl:
        st.info("No Arabic letters to display.")
        return

    cols = st.columns(len(letters_rtl))
    for col, letter in zip(cols, letters_rtl):
        with col:
            img_path = find_letter_image(letter)
            if img_path:
                st.image(img_path, use_column_width=True)
            else:
                st.markdown("❌")
            st.markdown(
                f"<div style='text-align:center;font-size:24px;'>{letter}</div>",
                unsafe_allow_html=True
            )

def translate_to_arabic(text):
    """Call ngrok NLLB endpoint. Returns (arabic_text, error_message)."""
    try:
        r = requests.post(
            f"{NGROK_URL}/translate",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"english": text},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("response", ""), None
        return None, f"API error {r.status_code}: {r.text}"
    except Exception as e:
        return None, f"Request failed: {e}"

# =====================================================================
# MODEL SELECTION (must run before any section that uses `model`)
# =====================================================================
model_choice = st.selectbox("Model", ["InceptionV3", "ResNet50", "EfficientNet"])
model        = load_selected_model(model_choice)

# =====================================================================
# SIDEBAR — camera launcher
# =====================================================================
with st.sidebar:
    if st.button("📷 Open Camera"):
        subprocess.Popen(
            [MP_PYTHON, "camera.py"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        st.success("Camera launched")

# =====================================================================
# SECTION 1 — Camera → Translate
# =====================================================================
st.markdown("---")
st.subheader("🔤 Camera → Translate")

sentence_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentence.txt")

if os.path.exists(sentence_path):
    with open(sentence_path, "r", encoding="utf-8") as f:
        captured = f.read().strip()

    if captured:
        st.write(f"**Captured (English):** `{captured}`")

        if st.button("🌐 Translate to Arabic", key="translate_cam"):
            arabic, err = translate_to_arabic(captured)
            if err:
                st.error(err)
            else:
                st.success(f"**Arabic:** {arabic}")
                st.subheader("🖐️ Arabic Sign Language")
                display_arabic_signs(arabic)
    else:
        st.info("sentence.txt is empty — capture something with the camera first.")
else:
    st.info("No sentence yet. Open camera, sign letters, press Q when done.")

# =====================================================================
# SECTION 2 — Upload multiple ASL images → Word → Arabic
# =====================================================================
st.markdown("---")
st.subheader("📤 Upload Multiple ASL Images → Word → Arabic")

asl_files = st.file_uploader(
    "Upload ASL letter images in order (left to right)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="multi_asl"
)

if asl_files:
    st.write(f"**{len(asl_files)} images uploaded** — predicting each letter...")

    thumb_cols = st.columns(len(asl_files))
    predicted_letters = []

    for col, file in zip(thumb_cols, asl_files):
        img = Image.open(file).convert("RGB")
        img_arr = np.array(img.resize((224, 224)))
        img_in = np.expand_dims(img_arr.astype("float32") / 255.0, axis=0)

        raw = model.predict(img_in)
        if isinstance(raw, (list, tuple)): raw = raw[0]
        preds = np.asarray(raw).squeeze()
        idx = int(np.argmax(preds))
        letter = class_names[idx]
        conf = float(preds[idx])

        if letter.lower() == "space":
            predicted_letters.append(" ")
        elif letter.lower() in ("del", "delete", "nothing"):
            pass
        else:
            predicted_letters.append(letter)

        with col:
            st.image(img, use_column_width=True)
            st.markdown(
                f"<div style='text-align:center;font-size:20px;'><b>{letter}</b><br>"
                f"<span style='font-size:12px;color:gray;'>{conf*100:.0f}%</span></div>",
                unsafe_allow_html=True
            )

    english_word = "".join(predicted_letters)
    st.write(f"**Predicted word (English):** `{english_word}`")

    if english_word.strip() and st.button("🌐 Translate uploaded word", key="translate_upload"):
        arabic, err = translate_to_arabic(english_word.lower())
        if err:
            st.error(err)
        else:
            st.success(f"**Arabic:** {arabic}")
            st.subheader("🖐️ Arabic Sign Language")
            display_arabic_signs(arabic)

# =====================================================================
# SECTION 3 — Single image upload + Grad-CAM
# =====================================================================
st.markdown("---")
st.subheader("🖼️ Single Image — Top 3 Predictions + Grad-CAM")

uploaded = st.file_uploader("Upload an ASL hand image", type=["jpg", "jpeg", "png"], key="single")

if uploaded:
    image   = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_arr = np.array(image.resize((224, 224)))
    img_in  = np.expand_dims(img_arr.astype("float32") / 255.0, axis=0)

    raw = model.predict(img_in)
    if isinstance(raw, (list, tuple)): raw = raw[0]
    preds = np.asarray(raw).squeeze()

    if preds.ndim != 1 or len(preds) != len(class_names):
        st.error(f"Model output {preds.shape} doesn't match class_names ({len(class_names)} classes).")
        st.stop()

    top3 = [int(i) for i in np.argsort(preds)[-3:][::-1]]

    st.subheader("Top 3 Predictions")
    for rank, idx in enumerate(top3, 1):
        st.write(f"{rank}. **{class_names[idx]}** — {preds[idx]*100:.2f}%")

    st.subheader("Grad-CAM")
    try:
        heatmap   = make_gradcam_heatmap(img_in, model, LAST_CONV_MAP[model_choice], top3[0])
        cam_image = overlay_gradcam(img_arr, heatmap)
        st.image(cam_image, caption="Grad-CAM", use_column_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
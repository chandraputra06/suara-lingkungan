
import json, numpy as np, streamlit as st
import librosa, librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(page_title="Klasifikasi Suara Lingkungan", page_icon="🔊", layout="wide")

@st.cache_resource
def load_all():
    with open("deploy/config.json") as f:
        cfg = json.load(f)
    bundle = {"cfg": cfg}
    if cfg["best_model"] == "transfer":
        import tensorflow_hub as hub
        bundle["yamnet"] = hub.load("https://tfhub.dev/google/yamnet/1")
        bundle["head"] = tf.keras.models.load_model("deploy/model_transfer.keras")
    else:
        bundle["cnn"] = tf.keras.models.load_model("deploy/model_cnn.keras")
    return bundle

B = load_all(); cfg = B["cfg"]; labels = cfg["labels"]

st.title("🔊 Klasifikasi Suara Lingkungan (ESC-50)")
st.caption(f"Model aktif: {cfg['best_model'].upper()} | "
           f"Akurasi CNN {cfg['acc_cnn']}% · Transfer {cfg['acc_transfer']}%")

def load_audio(file):
    y, _ = librosa.load(file, sr=cfg["sr"], duration=cfg["duration"])
    target = cfg["sr"] * cfg["duration"]
    y = np.pad(y, (0, target - len(y))) if len(y) < target else y[:target]
    return y

def predict(y):
    if cfg["best_model"] == "transfer":
        _, emb, _ = B["yamnet"](y)
        vec = tf.reduce_mean(emb, axis=0).numpy()[None, :]
        return B["head"].predict(vec, verbose=0)[0]
    mel = librosa.feature.melspectrogram(y=y, sr=cfg["sr"], n_fft=cfg["n_fft"],
                                         hop_length=cfg["hop"], n_mels=cfg["n_mels"])
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - cfg["mean"]) / (cfg["std"] + 1e-6)
    return B["cnn"].predict(lm[None, ..., None], verbose=0)[0]

up = st.file_uploader("Unggah file audio (.wav / .ogg / .mp3)", type=["wav", "ogg", "mp3"])
if up:
    st.audio(up)
    y = load_audio(up)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(np.linspace(0, cfg["duration"], len(y)), y, color="steelblue", lw=0.5)
        ax.set_title("Waveform"); ax.set_xlabel("Waktu (s)")
        st.pyplot(fig)
    with c2:
        mel = librosa.feature.melspectrogram(y=y, sr=cfg["sr"], n_fft=cfg["n_fft"],
                                             hop_length=cfg["hop"], n_mels=cfg["n_mels"])
        lm = librosa.power_to_db(mel, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        img = librosa.display.specshow(lm, sr=cfg["sr"], hop_length=cfg["hop"],
                                       x_axis="time", y_axis="mel", ax=ax2)
        ax2.set_title("Log-Mel Spectrogram"); fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)

    probs = predict(y)
    top = probs.argsort()[-5:][::-1]
    st.subheader(f"Prediksi: **{labels[top[0]]}**  ({probs[top[0]]*100:.1f}%)")
    st.bar_chart({labels[i]: float(probs[i]) for i in top})
else:
    st.info("Silakan unggah file audio untuk memulai klasifikasi.")

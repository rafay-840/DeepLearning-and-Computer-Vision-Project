"""
Student Engagement Recognition — Live Webcam Demo

Streams webcam video via streamlit-webrtc, buffers a rolling window of 8
frames, and runs a trained CNN+LSTM model to predict engagement in real
time. Supports two modes:

  - Binary:   engaged / disengaged   (ResNet-18+LSTM, 2-class head)
  - 3-class:  engaged / bored / other (ResNet-18+LSTM, 3-class head)

All inference runs locally in the browser session's backend process; no
video is stored or transmitted beyond the active session.

Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from collections import deque
import threading

# ─────────────────────────────────────────────────────────
# Model definition (must match training exactly)
# ─────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEQUENCE_LENGTH = 8
FRAME_SIZE = (112, 112)

# Mode configuration: each mode has its own class names, default checkpoint
# filename, and display colors. Adding a new mode later just means adding
# an entry here -- no other code needs to change.
MODE_CONFIG = {
    "Binary (engaged / disengaged)": {
        "num_classes": 2,
        "class_names": ["disengaged", "engaged"],
        "default_checkpoint": "resnet18_lstm_best.pth",
        "colors": {
            "disengaged": (0, 0, 220),   # red (BGR)
            "engaged": (0, 200, 0),      # green
        },
    },
    "3-class (engaged / bored / other)": {
        "num_classes": 3,
        "class_names": ["engaged", "bored", "other"],
        "default_checkpoint": "resnet18_lstm_3class_best.pth",
        "colors": {
            "engaged": (0, 200, 0),      # green
            "bored": (0, 165, 255),      # orange
            "other": (180, 0, 180),      # purple
        },
    },
}


class CNNLSTMClassifier(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=2, lstm_hidden_size=256,
                 lstm_num_layers=1, dropout=0.5, pretrained=False):
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            cnn = models.resnet18(weights=weights)
            self.cnn_backbone = nn.Sequential(*list(cnn.children())[:-2])
            self.cnn_pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone for demo: {backbone}")

        self.feature_dim = feature_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=lstm_hidden_size,
                             num_layers=lstm_num_layers, batch_first=True,
                             dropout=dropout if lstm_num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_hidden_size, num_classes))

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn_backbone(x)
        features = self.cnn_pool(features)
        features = features.view(batch_size, seq_len, self.feature_dim)
        lstm_out, (h_n, c_n) = self.lstm(features)
        final_hidden = h_n[-1]
        logits = self.classifier(final_hidden)
        return logits


@st.cache_resource
def load_model(checkpoint_path, num_classes):
    """
    Loads the trained model once per (checkpoint_path, num_classes) combination,
    cached across reruns. Including num_classes in the cache key is essential:
    without it, switching modes would silently reuse a model with the WRONG
    output head size, either crashing on load_state_dict or -- worse -- loading
    successfully but producing meaningless predictions if shapes happened to
    partially match.
    """
    model = CNNLSTMClassifier(backbone="resnet18", num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def preprocess_frame(frame_bgr):
    """OpenCV BGR frame -> normalized tensor matching training preprocessing."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, FRAME_SIZE)
    tensor = T.ToTensor()(frame_resized)
    tensor = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)
    return tensor


class EngagementVideoProcessor(VideoProcessorBase):
    """
    Maintains a rolling buffer of the last SEQUENCE_LENGTH frames and runs
    the CNN+LSTM model whenever the buffer is full, updating a shared
    prediction state that the Streamlit UI reads for display.

    class_names and colors are injected externally (set right after
    instantiation) so the SAME processor class works for both binary and
    3-class modes without needing two near-duplicate classes.
    """
    def __init__(self):
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.lock = threading.Lock()
        self.latest_prediction = "Buffering..."
        self.latest_confidence = 0.0
        self.model = None          # set externally after instantiation
        self.class_names = None    # set externally after instantiation
        self.colors = None         # set externally after instantiation

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        tensor = preprocess_frame(img_bgr)
        with self.lock:
            self.frame_buffer.append(tensor)

            ready = (len(self.frame_buffer) == SEQUENCE_LENGTH
                     and self.model is not None
                     and self.class_names is not None)

            if ready:
                sequence = torch.stack(list(self.frame_buffer)).unsqueeze(0)  # (1, 8, 3, 112, 112)
                with torch.no_grad():
                    logits = self.model(sequence)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = logits.argmax(dim=1).item()
                    confidence = probs[0, pred_class].item()

                self.latest_prediction = self.class_names[pred_class]
                self.latest_confidence = confidence

        # Overlay the current prediction on the displayed frame
        display_frame = img_bgr.copy()
        if self.latest_confidence > 0:
            label = f"{self.latest_prediction} ({self.latest_confidence:.0%})"
        else:
            label = self.latest_prediction

        color = (200, 200, 200)  # default gray while buffering
        if self.colors is not None and self.latest_prediction in self.colors:
            color = self.colors[self.latest_prediction]

        cv2.rectangle(display_frame, (10, 10), (360, 55), (30, 30, 30), -1)
        cv2.putText(display_frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


# ─────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Engagement Recognition", layout="centered")

st.title("Student Engagement Recognition")
st.caption("Computer Vision & Deep Learning — University of Verona, A.Y. 2025\u201326")

# ── Mode selection ──
selected_mode = st.sidebar.radio(
    "Classification mode",
    options=list(MODE_CONFIG.keys()),
    help="Binary uses our best-performing model (test macro-F1 0.595). "
         "3-class is an exploratory extension (test macro-F1 0.324, near chance "
         "level for minority classes) -- see Section 4.8 of the project report."
)

mode_cfg = MODE_CONFIG[selected_mode]
num_classes = mode_cfg["num_classes"]
class_names = mode_cfg["class_names"]
colors = mode_cfg["colors"]

st.markdown(
    f"This demo runs a trained **ResNet-18 + LSTM** model live on your webcam feed, "
    f"in **{selected_mode}** mode. Frames are buffered into 8-frame sequences and "
    f"classified as **{' / '.join(class_names)}**. All processing happens locally "
    f"in this session's backend process — no video is stored, saved to disk, or "
    f"transmitted anywhere beyond this live session."
)

if num_classes == 3:
    st.info(
        "**Note on this mode:** the 3-class model performs well on 'engaged' but "
        "struggles on 'bored' and especially 'other' (test recall as low as 4.1% "
        "for 'other'), due to severe class imbalance in the training data. This is "
        "discussed transparently in Section 4.8 of the project report.",
        icon="ℹ️"
    )

# ── Checkpoint path, defaulting per-mode ──
# IMPORTANT: st.text_input does NOT automatically update its displayed value
# when `value=` changes on a later script rerun if the widget already has a
# value in session_state. We force a reset whenever the MODE changes by
# keying the widget's session_state entry to the mode itself, so switching
# modes always shows the correct default path rather than a stale one.
checkpoint_state_key = f"checkpoint_path_{selected_mode}"
if checkpoint_state_key not in st.session_state:
    st.session_state[checkpoint_state_key] = mode_cfg["default_checkpoint"]

CHECKPOINT_PATH = st.sidebar.text_input(
    "Model checkpoint path",
    key=checkpoint_state_key,
    help=f"Path to the trained checkpoint for {selected_mode} mode (.pth file)"
)

# Extra safety: warn explicitly if the filename doesn't look like it matches
# the selected mode, BEFORE attempting to load (the load itself will also
# catch a genuine mismatch via the RuntimeError handler below, but this
# gives an earlier, clearer hint of what likely went wrong).
expected_filename = mode_cfg["default_checkpoint"]
if CHECKPOINT_PATH != expected_filename and "3class" in CHECKPOINT_PATH.lower() and num_classes == 2:
    st.sidebar.warning(
        f"This checkpoint path looks like a 3-class checkpoint, but Binary mode "
        f"is selected. Expected something like '{expected_filename}'.",
        icon="⚠️"
    )
elif CHECKPOINT_PATH != expected_filename and "3class" not in CHECKPOINT_PATH.lower() and num_classes == 3:
    st.sidebar.warning(
        f"This checkpoint path looks like a binary checkpoint, but 3-class mode "
        f"is selected. Expected something like '{expected_filename}'.",
        icon="⚠️"
    )

try:
    model = load_model(CHECKPOINT_PATH, num_classes)
    st.sidebar.success(f"Model loaded successfully ({num_classes}-class head)")
except FileNotFoundError:
    st.sidebar.error(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    st.stop()
except RuntimeError as e:
    # Most likely cause: checkpoint's output layer shape doesn't match
    # num_classes (e.g. pointed a 2-class checkpoint at 3-class mode, or
    # vice versa -- this is exactly the mismatch the sidebar warning above
    # tries to catch earlier, but this is the hard backstop)
    st.sidebar.error(
        f"**Checkpoint / mode mismatch.** The selected mode expects a "
        f"{num_classes}-class output head, but the checkpoint at "
        f"'{CHECKPOINT_PATH}' has a different number of classes. "
        f"\n\nFix: either switch the mode to match this checkpoint, or change "
        f"the checkpoint path to '{mode_cfg['default_checkpoint']}' for "
        f"{selected_mode} mode."
        f"\n\nTechnical details: {e}"
    )
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Note:** Predictions update once every 8 frames as the rolling buffer fills. "
    "There may be a brief delay before the first prediction appears."
)
st.sidebar.markdown(
    "**Switching modes:** changing the mode above will reload the model "
    "with the matching checkpoint and output head automatically."
)

# Use a key that includes the mode so switching modes creates a fresh
# webrtc component instance rather than reusing a stale video processor
# still configured for the previous mode's class count.
webrtc_ctx = webrtc_streamer(
    key=f"engagement-demo-{selected_mode}",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EngagementVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Attach the loaded model and mode-specific config to the processor instance
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.model = model
    webrtc_ctx.video_processor.class_names = class_names
    webrtc_ctx.video_processor.colors = colors

st.markdown("---")
st.markdown(
    "**Privacy note:** This is an academic prototype built on the DAiSEE dataset for a "
    "university course project. No frames are saved, logged, or transmitted to any server. "
    "A production deployment would require explicit informed consent and GDPR-compliant "
    "data handling, as discussed in the accompanying project report."
)
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
    def __init__(self, num_classes: int = 2, lstm_hidden: int = 256,
                 lstm_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        cnn              = models.resnet18(weights=None)
        self.backbone    = nn.Sequential(*list(cnn.children())[:-2])
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 512
        self.lstm        = nn.LSTM(
            input_size=self.feature_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = x.shape
        feat = self.pool(self.backbone(x.view(B * S, C, H, W)))
        feat = feat.view(B, S, self.feature_dim)
        _, (h_n, _) = self.lstm(feat)
        return self.classifier(h_n[-1])

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
        img  = frame.to_ndarray(format="bgr24")
        disp = img.copy()
 
        with self._lock:
            self._n += 1
            roi, bbox = img, None
 
            # Enhancement 2: face crop
            if self.use_face_roi:
                cropped, bbox = self._cropper.crop(img)
                if cropped is not None:
                    roi = cropped
 
            # Enhancement 8: frame stride
            if self._n % max(1, self.frame_stride) == 0:
                self._buf.append(preprocess(roi))
 
            # Inference
            ready = (
                len(self._buf) == SEQUENCE_LENGTH
                and self.model is not None
                and self.class_names is not None
            )
            if ready:
                # Enhancement 9: move to model's device
                device = next(self.model.parameters()).device
                seq = torch.stack(list(self._buf)).unsqueeze(0).to(device)
 
                with torch.no_grad():
                    logits = self.model(seq)
                    probs  = torch.softmax(logits, dim=1)
                    idx    = logits.argmax(1).item()
                    raw_conf = probs[0, idx].item()
 
                # Enhancement 7: confidence threshold → 'uncertain'
                raw_label = (
                    self.class_names[idx]
                    if raw_conf >= self.conf_threshold
                    else "uncertain"
                )
 
                # Enhancement 3: temporal smoothing
                label, conf = self._smoother.update(
                    raw_label, raw_conf, self.smoothing_window
                )
 
                self.latest_label = label
                self.latest_conf  = conf
 
                # Enhancement 4: log for analytics
                self.logger.log(label, conf)
 
                # Enhancement 6: disengagement timer
                if label == "disengaged":
                    if self._diseng_t is None:
                        self._diseng_t = time.time()
                    self.disengaged_secs = time.time() - self._diseng_t
                else:
                    self._diseng_t       = None
                    self.disengaged_secs = 0.0
 
        self._draw_overlay(disp, bbox)
        return av.VideoFrame.from_ndarray(disp, format="bgr24")

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
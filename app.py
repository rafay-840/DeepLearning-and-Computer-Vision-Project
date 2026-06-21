"""
Student Engagement Recognition — Live Webcam Demo

Streams webcam video via streamlit-webrtc, buffers a rolling window of 8
frames, and runs the trained ResNet-18+LSTM model to predict engagement
(engaged / disengaged) in real time. All inference runs locally in the
browser session's backend process; no video is stored or transmitted
beyond the active session.

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
CLASS_NAMES = ["disengaged", "engaged"]


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
def load_model(checkpoint_path):
    """Loads the trained model once per session, cached across reruns."""
    model = CNNLSTMClassifier(backbone="resnet18", num_classes=2)
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
    """
    def __init__(self):
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.lock = threading.Lock()
        self.latest_prediction = "Buffering..."
        self.latest_confidence = 0.0
        self.model = None  # set externally after instantiation

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")

        tensor = preprocess_frame(img_bgr)
        with self.lock:
            self.frame_buffer.append(tensor)

            if len(self.frame_buffer) == SEQUENCE_LENGTH and self.model is not None:
                sequence = torch.stack(list(self.frame_buffer)).unsqueeze(0)  # (1, 8, 3, 112, 112)
                with torch.no_grad():
                    logits = self.model(sequence)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = logits.argmax(dim=1).item()
                    confidence = probs[0, pred_class].item()

                self.latest_prediction = CLASS_NAMES[pred_class]
                self.latest_confidence = confidence

        # Overlay the current prediction on the displayed frame
        display_frame = img_bgr.copy()
        label = f"{self.latest_prediction} ({self.latest_confidence:.0%})" if self.latest_confidence > 0 else self.latest_prediction
        color = (0, 200, 0) if self.latest_prediction == "engaged" else (0, 0, 220)
        cv2.rectangle(display_frame, (10, 10), (340, 55), (30, 30, 30), -1)
        cv2.putText(display_frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


# ─────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Engagement Recognition", layout="centered")

st.title("Student Engagement Recognition")
st.caption("Computer Vision & Deep Learning — University of Verona, A.Y. 2025\u201326")

st.markdown(
    "This demo runs the trained **ResNet-18 + LSTM** model live on your webcam feed. "
    "Frames are buffered into 8-frame sequences and classified as **engaged** or **disengaged**. "
    "All processing happens locally in this session's backend process — no video is stored, "
    "saved to disk, or transmitted anywhere beyond this live session."
)

CHECKPOINT_PATH = st.sidebar.text_input(
    "Model checkpoint path",
    value="resnet18_lstm_best.pth",
    help="Path to the trained ResNet-18+LSTM checkpoint (.pth file)"
)

try:
    model = load_model(CHECKPOINT_PATH)
    st.sidebar.success("Model loaded successfully")
except FileNotFoundError:
    st.sidebar.error(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Note:** Predictions update once every 8 frames as the rolling buffer fills. "
    "There may be a brief delay before the first prediction appears."
)

webrtc_ctx = webrtc_streamer(
    key="engagement-demo",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EngagementVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Attach the loaded model to the processor instance once the stream starts
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.model = model

st.markdown("---")
st.markdown(
    "**Privacy note:** This is an academic prototype built on the DAiSEE dataset for a "
    "university course project. No frames are saved, logged, or transmitted to any server. "
    "A production deployment would require explicit informed consent and GDPR-compliant "
    "data handling, as discussed in the accompanying project report."
)

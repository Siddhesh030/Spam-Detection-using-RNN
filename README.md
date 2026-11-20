# 📩 Spam Detection Using Recurrent Neural Network (RNN)

<p align="center">
  <a href="https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/">
    <img src="https://img.shields.io/badge/Streamlit%20Cloud-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
  <a href="https://github.com/Siddhesh030/Spam-Detection-using-RNN">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras" alt="Keras 3">
  <img src="https://img.shields.io/badge/Engine-PyTorch%20Backend-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch Backend">
</p>

An intelligent Natural Language Processing (NLP) and Deep Learning web application designed to classify text messages as **Spam** or **Ham (Legitimate)** using a trained Recurrent Neural Network with an LSTM (Long Short-Term Memory) layer.

Built on modern **Keras 3 multi-backend architecture** and powered by **Streamlit**, featuring interactive theme customization, token breakdown with Out-Of-Vocabulary (OOV) analysis, prediction confidence scoring, and an active audit trail.

---

## 🌐 Live Application

### 👉 [Launch Spam Detection System](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)

Experiment with real-time text classification directly in your browser without local setup.

---

## 📌 Table of Contents

- [Project Highlights](#-project-highlights)
- [System Architecture](#-system-architecture)
- [How the System Works](#-how-the-system-works)
- [Neural Network Specifications](#-neural-network-specifications)
- [UI & Interactive Features](#-ui--interactive-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Streamlit Cloud Deployment](#-streamlit-cloud-deployment)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## ✨ Project Highlights

* **🧠 Deep Sequential Learning**: Leverages an LSTM recurrent neural network trained on sequential text data to capture semantic and contextual message patterns.
* **⚡ Keras 3 Multi-Backend Engine**: Utilizes Keras 3 with the PyTorch CPU backend, ensuring high-speed inference and eliminating heavy TensorFlow memory overhead for cloud deployments.
* **🔍 Token & Vocabulary Inspector**: Breaks down input messages into individual tokens, highlighting mapped vocabulary indices and identifying Out-of-Vocabulary (OOV) words in real time.
* **🎨 5 Dynamic Themes**: Built-in multi-color theme switcher supporting *Cyber Dark*, *Emerald Matrix*, *Sunset Neon*, *Royal Indigo*, and *Slate Monolith*.
* **📊 Visual Confidence Gauges**: Interactive Plotly gauges showing spam vs. ham probabilities with customizable decision thresholds.
* **📜 Real-Time Audit Trail**: Keeps an in-memory session log of analyzed messages with timestamps, labels, and latencies.

---

## ⚙️ How the System Works

```text
  User Text Input
        │
        ▼
  Text Preprocessing & Cleaning
        │
        ▼
  Tokenization (Vocabulary: 8,143 words)
        │
        ▼
  Sequence Padding (Length: 100, Post-padding)
        │
        ▼
  LSTM Neural Network Inference (Keras 3)
        │
        ▼
  Sigmoid Output Probability (0.0 to 1.0)
        │
        ▼
  Threshold Decision Logic (Default: 0.50)
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
[ SPAM ]                     [ HAM (Not Spam) ]
(Alert Box & Probability)    (Verification & Metrics)
```

1. **Input**: User submits an SMS or text message.
2. **Tokenization**: Converted into integer sequence using `tokenizer.pkl` based on an 8,143-word vocabulary.
3. **Padding**: Padded to a uniform sequence length of 100 integers.
4. **LSTM Layer**: Evaluates the sequential dependencies and recurrent representations.
5. **Output**: Dense sigmoid layer outputs a spam probability score compared against the decision threshold.

---

## 🧠 Neural Network Specifications

The underlying deep learning model (`spam_rnn.keras`) is configured as follows:

| Layer | Type | Specifications | Details |
|---|---|---|---|
| **Layer 1** | `InputLayer` | `(None, 100)` | Accepts padded integer sequences |
| **Layer 2** | `Embedding` | `input_dim=8143, output_dim=64` | Learns 64-dimensional dense word representations (`mask_zero=True`) |
| **Layer 3** | `LSTM` | `units=64, activation='tanh'` | Recurrent layer capturing sequential message context |
| **Layer 4** | `Dropout` | `rate=0.3` | Regularization to prevent overfitting |
| **Layer 5** | `Dense` | `units=32, activation='relu'` | Fully connected feature extraction layer |
| **Layer 6** | `Dropout` | `rate=0.3` | Additional regularizer |
| **Layer 7** | `Dense` | `units=1, activation='sigmoid'` | Final classification probability output |

* **Optimizer**: Adam (`learning_rate=0.00025`)
* **Loss Function**: Binary Crossentropy
* **Evaluation Metrics**: Accuracy, Precision, Recall

---

## 🎨 UI & Interactive Features

* **Theme Customization**: Easily switch color themes via the sidebar without reloading the app.
* **Confidence Adjuster**: Customize the classification decision threshold (from 0.1 to 0.9) to test precision vs. recall trade-offs.
* **Sample Presets**: One-click preset messages (bank scams, lottery alerts, authentic business notices, personal texts) for quick demonstration.
* **Performance Benchmark**: Real-time display of model inference latency in milliseconds.

---

## 🛠️ Technology Stack

* **Language**: Python 3.11
* **Deep Learning**: [Keras 3](https://keras.io/) (with PyTorch CPU inference engine)
* **Web Framework**: [Streamlit](https://streamlit.io/)
* **Data & Analytics**: NumPy, Pandas, Scikit-learn
* **Visualization**: Plotly Express & Plotly Graph Objects
* **Deployment**: Streamlit Community Cloud

---

## 📁 Project Structure

```text
Spam-Detection-using-RNN/
├── .python-version        # Pins Python 3.11 for Streamlit Cloud
├── app.py                 # Streamlit web application & inference logic
├── label_encoder.pkl      # Serialized label encoder ('ham', 'spam')
├── model_config.pkl       # Hyperparameters (max_length, vocab_size, threshold)
├── requirements.txt       # Production dependencies
├── spam_rnn.keras         # Trained Keras 3 LSTM model artifact
├── tokenizer.pkl          # Trained text tokenizer and vocabulary index
└── README.md              # Project documentation
```

---

## 💻 Local Installation & Setup

Follow these steps to run the application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Siddhesh030/Spam-Detection-using-RNN.git
cd Spam-Detection-using-RNN
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

The application will start locally at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment

This repository is optimized for **Streamlit Community Cloud**:
* `.python-version` pins **Python 3.11** for stable wheel support.
* Uses lightweight CPU-optimized dependencies to prevent container memory exhaustion.
* Pure Keras 3 multi-backend engine eliminates circular imports.

If re-deploying or restarting your own instance:
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select repository `Siddhesh030/Spam-Detection-using-RNN` and branch `main`.
3. Set the entrypoint to `app.py`.
4. Click **Deploy**.

---

## 👨‍💻 Author

**Siddhesh**
* **GitHub**: [@Siddhesh030](https://github.com/Siddhesh030)
* **Repository**: [Spam-Detection-using-RNN](https://github.com/Siddhesh030/Spam-Detection-using-RNN)
* **Live App**: [Streamlit Deployment](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)

---

## ⭐ Support

If you find this project helpful for learning Natural Language Processing and Deep Learning with Keras 3 and Streamlit, please give the repository a star ⭐!

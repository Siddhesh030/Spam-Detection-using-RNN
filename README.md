# 📩 Spam Detection Using Recurrent Neural Network (RNN)

An intelligent Natural Language Processing (NLP) and Deep Learning project designed to classify text messages as **Spam** or **Not Spam (Ham)** using a Recurrent Neural Network (RNN). The project demonstrates how deep learning techniques can be applied to text classification and integrated into an interactive web application using Streamlit.

🌐 **Live Demo:** [Try the Spam Detection App](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)

📂 **GitHub Repository:** [Spam Detection using RNN](https://github.com/Siddhesh030/Spam-Detection-using-RNN)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [How the System Works](#-how-the-system-works)
- [Deep Learning Approach](#-deep-learning-approach)
- [Technology Stack](#-technology-stack)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Installation and Setup](#-installation-and-setup)
- [How to Use the Application](#-how-to-use-the-application)
- [Model Evaluation](#-model-evaluation)
- [Applications](#-applications)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Live Deployment](#-live-deployment)
- [Author](#-author)

---

## 🚀 Project Overview

Spam messages are a common problem in digital communication. They often contain unwanted advertisements, fraudulent offers, misleading information, suspicious links, or potentially harmful content.

The **Spam Detection using RNN** project explores how deep learning and Natural Language Processing can help automatically distinguish spam messages from legitimate messages.

The system processes text input and uses a trained Recurrent Neural Network to generate a classification. The model can learn sequential patterns in text, making RNNs a relevant deep learning approach for natural language classification tasks.

The project combines a deep learning model with an interactive Streamlit interface, allowing users to experiment with message classification through a web browser.

### 🎯 Main Goal

To develop an end-to-end text classification application that demonstrates the practical use of deep learning for spam detection, from text preprocessing and model training to prediction and web deployment.

---

## ❓ Problem Statement

The increasing volume of unwanted messages makes manual identification inefficient. Traditional filtering methods based only on predefined keywords may struggle when spam messages use different wording or sentence structures.

A machine learning or deep learning system can learn patterns from labeled examples and use those patterns to classify previously unseen messages.

This project addresses the problem by applying a Recurrent Neural Network to the task of spam message classification.

The system aims to:

- Accept text messages as input.
- Prepare text for processing by a deep learning model.
- Identify patterns associated with spam and legitimate messages.
- Predict the appropriate message category.
- Present the prediction through an accessible web interface.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. **Text Classification:** Build a system for distinguishing spam messages from legitimate messages.
2. **Natural Language Processing:** Apply appropriate text preprocessing and numerical representation techniques.
3. **Deep Learning:** Explore the application of Recurrent Neural Networks to sequential text data.
4. **Model Integration:** Connect the trained model to an application that accepts user-provided text.
5. **Interactive Interface:** Provide an easy-to-use interface through Streamlit.
6. **Practical Deployment:** Make the application accessible online.
7. **Hands-on Learning:** Demonstrate the complete workflow of a deep learning project, from data preparation to deployment.

---

## ✨ Key Features

### 📩 1. Spam Message Classification

The system is designed to classify a text message into one of two categories:

- **Spam:** A message identified by the model as spam.
- **Ham (Not Spam):** A message identified by the model as legitimate.

### 🧠 2. Deep Learning-Based Prediction

The project uses a Recurrent Neural Network approach to learn patterns from sequential text representations.

Unlike a simple collection of manually specified keyword rules, a trained neural network can learn relationships represented in its training examples.

### 📝 3. Natural Language Processing

Text classification generally requires transforming raw messages into a numerical representation suitable for a neural network.

Depending on the implementation, this process can include text cleaning, tokenization, vocabulary mapping, and sequence padding.

### ⚡ 4. Interactive Predictions

Users can enter a message into the deployed application and obtain a classification without running the model manually from a terminal.

### 🌐 5. Streamlit Web Application

The project provides a browser-based interface, making the underlying deep learning functionality accessible to users without requiring them to interact directly with Python code.

### 🔗 6. Online Accessibility

The deployed application can be accessed through the following link:

**[Open the Live Spam Detection Application](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)**

---

## ⚙️ How the System Works

The application follows a text classification pipeline.

### Step 1: Input Message

The user enters a text message into the application's input field.

Example:

`Congratulations! You have won a free prize. Click here to claim your reward.`

### Step 2: Text Preprocessing

The input is prepared for the model using the preprocessing procedure associated with the training pipeline.

Depending on the implementation, preprocessing may include:

- Converting text to a consistent case.
- Removing or normalizing selected characters.
- Tokenizing the message into words or tokens.
- Converting tokens into numerical identifiers.
- Padding or truncating sequences to the expected input length.

**Important:** The inference-time preprocessing must match the preprocessing used when training the model.

### Step 3: Numerical Representation

Neural networks operate on numerical data rather than raw text.

A typical RNN-based text classifier uses a vocabulary and token indices to represent a message as a sequence of numbers.

If an embedding layer is used, these token indices are converted into dense vector representations that the network can learn from.

### Step 4: Recurrent Neural Network Processing

The sequence is passed to the trained RNN model.

A recurrent network processes sequential information while maintaining a hidden state that can represent information from earlier elements in the sequence.

This makes recurrent architectures useful for learning patterns in ordered text data.

### Step 5: Classification

The model produces an output used to determine the predicted message category.

For a binary classifier, the output may be a probability or score that is converted into a Spam or Ham prediction using the model's configured decision rule.

### Step 6: Display the Result

The Streamlit application presents the resulting classification to the user.

The exact output format depends on the implementation of the deployed application.

---

## 🧠 Deep Learning Approach

### What Is a Recurrent Neural Network?

A Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data.

In text classification, the order of words can provide useful information. Recurrent networks process a sequence step by step and maintain a hidden state that carries information across the sequence.

For example, the phrases below have different meanings:

- "You have won a prize."
- "You have not won a prize."

A text classifier must learn patterns from the available training data to distinguish between different contexts.

### Typical RNN Text Classification Architecture

A conventional RNN-based text classification pipeline may contain the following components:

| Component | Purpose |
|---|---|
| Text preprocessing | Standardizes the input message. |
| Tokenizer | Maps words or tokens to numerical identifiers. |
| Sequence padding | Makes input sequences compatible with the model's expected dimensions. |
| Embedding layer | Learns dense representations of tokens, if used. |
| Recurrent layer | Processes sequential information across the message. |
| Output layer | Produces the classification score or probability. |
| Decision threshold | Converts the model output into a class label. |

The table describes a typical RNN text classification architecture. The exact layers, activation functions, vocabulary size, sequence length, and threshold should be documented from the actual model implementation.

### Why Use an RNN for Spam Detection?

- **Sequential Processing:** RNNs process text as an ordered sequence.
- **Contextual Patterns:** Their hidden states can represent information from preceding tokens.
- **Trainable Representations:** Neural network parameters are learned from training data.
- **Flexible Classification:** The same general approach can be adapted to different text classification problems.

The performance of an RNN depends on the training data, preprocessing pipeline, architecture, hyperparameters, and evaluation methodology.

---

## 🛠️ Technology Stack

The project is centered on Python, deep learning, text processing, and web application development.

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Recurrent Neural Network (RNN) | Sequential text classification |
| Natural Language Processing (NLP) | Preparing and representing text |
| Streamlit | Interactive web application |
| Deep Learning Framework | Model implementation and inference |
| Tokenization and Sequence Processing | Converting text into model inputs |
| Git and GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Hosting the live application, if deployed through this service |

The precise libraries and framework versions should match the project's `requirements.txt` and source code.

---

## 🔄 Project Workflow

The overall workflow can be represented as follows:

```text
             ┌──────────────────────┐
             │   Text Dataset       │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Data Preparation     │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Text Preprocessing   │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Tokenization and     │
             │ Sequence Preparation │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ RNN Model Training   │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Model Evaluation     │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Save Trained Model   │
             │ and Preprocessor     │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Streamlit Interface  │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ User Enters Message  │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Generate Prediction  │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Display Spam / Ham   │
             └──────────────────────┘
```

This diagram illustrates a conventional end-to-end RNN workflow. The exact training and deployment steps should be adjusted to reflect the files and processing sequence in the repository.

---

## 📊 Dataset

The dataset is an important part of developing a supervised spam classification model.

A typical spam detection dataset contains text messages and their corresponding labels.

| Field | Description |
|---|---|
| Message | The text to be classified |
| Label | The category assigned to the message |
| Spam | Indicates an unwanted or spam message |
| Ham | Indicates a legitimate message |

### Dataset Preparation

A typical preparation process includes:

1. Loading the dataset.
2. Inspecting the available columns and labels.
3. Handling missing or invalid records.
4. Converting class labels into the format required by the model.
5. Splitting the data into training and testing subsets.
6. Preparing text sequences using the training-time vocabulary and preprocessing rules.

### Training and Testing

The training set is used to learn model parameters. The testing set is used to assess performance on examples not used for training.

To obtain a meaningful evaluation, preprocessing and vocabulary construction should avoid information leakage from the test set.

**Dataset details to add:** Include the actual dataset name, source, number of messages, class distribution, and train/test split after verifying the project's data files or notebook.

---

## 💻 Installation and Setup

Follow these instructions to run the project locally.

### Prerequisites

Make sure the following are installed:

- Python
- Git
- pip
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Siddhesh030/Spam-Detection-using-RNN.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Spam-Detection-using-RNN
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

If the repository contains a `requirements.txt` file, install the dependencies with:

```bash
pip install -r requirements.txt
```

If the project uses a particular deep learning framework, ensure its installed version is compatible with the model files and Python environment.

### Step 6: Launch the Application

If the Streamlit entry-point file is named `app.py`, run:

```bash
streamlit run app.py
```

If your repository uses another filename, replace `app.py` with the correct entry-point filename.

Streamlit will provide a local address that you can open in your browser.

---

## 🖥️ How to Use the Application

You can try the deployed version without installing Python or downloading the source code.

### Step 1: Open the Live Application

Visit:

**[Spam Detection using RNN — Live Demo](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)**

### Step 2: Enter a Message

Type or paste the message you want to analyze into the application's text input.

Example spam message:

```text
Congratulations! You have won a free gift.
Click the link to claim your reward now!
```

Example legitimate message:

```text
Hi, I will meet you at the office tomorrow morning.
```

These are illustrative test inputs, not verified outputs from the deployed model.

### Step 3: Run the Prediction

Use the interface as designed to submit or analyze the message.

### Step 4: Review the Classification

Check the result displayed by the application to see whether the model classifies the message as Spam or Ham.

Remember that a prediction is a model-generated estimate and may not always be correct.

---

## 📈 Model Evaluation

Evaluating a spam detection model requires more than measuring overall accuracy.

A useful evaluation should consider the consequences of both types of classification error.

### Recommended Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | The proportion of all predictions that are correct. |
| Precision | The proportion of messages predicted as spam that are actually spam. |
| Recall | The proportion of actual spam messages correctly identified. |
| F1-score | The harmonic mean of precision and recall. |
| Confusion Matrix | A breakdown of correct and incorrect predictions for each class. |

### Why These Metrics Matter

**Precision:** Helps assess how often messages flagged as spam are genuinely spam.

**Recall:** Helps assess how much of the actual spam content the model identifies.

**F1-score:** Provides a combined measure of precision and recall.

**Confusion Matrix:** Helps reveal false positives, where legitimate messages are flagged as spam, and false negatives, where spam messages are classified as legitimate.

For a practical spam detection system, both types of errors matter. High accuracy alone does not guarantee that the model is suitable for real-world use.

**Performance results:** Add the measured test accuracy, precision, recall, and F1-score from your own model evaluation. Avoid reporting unverified performance numbers.

---

## 🌍 Applications

The techniques demonstrated in this project can be relevant to several areas.

- **SMS Filtering:** Identifying unwanted text messages.
- **Email Classification:** Categorizing incoming emails into spam and legitimate messages.
- **Messaging Platforms:** Exploring automated message moderation and filtering.
- **Fraud Awareness:** Supporting the identification of suspicious promotional messages.
- **NLP Learning:** Demonstrating how sequential text data can be used in a deep learning model.
- **Deep Learning Education:** Providing a practical example of integrating model inference into a web application.

These are potential applications of the underlying approach, not claims that the current application implements all of these capabilities.

---

## ⚠️ Limitations

Like other supervised text classification systems, an RNN-based spam detector has limitations.

### 1. Dependence on Training Data

The model's ability to classify messages depends on the quality, quantity, and diversity of the training dataset.

### 2. Unseen Message Patterns

New spam techniques, unusual wording, and adversarial messages may not be classified correctly.

### 3. Language Limitations

Performance in languages other than those represented in the training data may be limited.

### 4. Contextual Ambiguity

Short messages, sarcasm, ambiguous wording, and missing context can make classification difficult.

### 5. False Predictions

A legitimate message may be flagged as spam, or a spam message may be classified as legitimate.

### 6. Model Confidence

If the application displays a probability or confidence score, it should not automatically be interpreted as a guarantee of correctness.

---

## 🔮 Future Enhancements

Potential improvements for future versions include:

- **LSTM-Based Classification:** Compare the RNN model with a Long Short-Term Memory network.
- **GRU-Based Classification:** Evaluate a Gated Recurrent Unit architecture.
- **Model Comparison:** Compare recurrent models with conventional machine learning classifiers and transformer-based approaches.
- **Improved Preprocessing:** Experiment with normalization, tokenization, and vocabulary strategies.
- **Hyperparameter Tuning:** Evaluate different sequence lengths, embedding dimensions, hidden units, and learning rates.
- **Detailed Evaluation:** Add confusion matrices, precision, recall, and F1-score visualizations.
- **Confidence Visualization:** Present prediction scores when supported by the model.
- **Bulk Message Analysis:** Explore classifying multiple messages from a file.
- **Improved Interface:** Add clearer prediction summaries and example messages.
- **Multilingual Support:** Explore datasets and preprocessing methods for additional languages.
- **Continuous Evaluation:** Test the model against newer and more diverse message samples.

These enhancements can help improve the model's usability, evaluation, and adaptability.

---

## 🚀 Live Deployment

The application is available online through the following link:

### 🌐 Try the Live Application

**[Launch Spam Detection using RNN](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)**

Users can access the deployed interface to experiment with text classification without setting up the project locally.

The application is intended for demonstration and educational purposes. Its predictions should not be treated as definitive proof that a message is safe or malicious.

---

## 👨‍💻 Author

**Siddhesh**

GitHub: [@Siddhesh030](https://github.com/Siddhesh030)

Project Repository: [Spam-Detection-using-RNN](https://github.com/Siddhesh030/Spam-Detection-using-RNN)

Live Application: [Spam Detection using RNN](https://spam-detection-using-rnn-bns7iih96fmtirgrnz4zh6.streamlit.app/)

---

## ⭐ Support

If you find this project useful for learning about deep learning, Natural Language Processing, or text classification, consider giving the repository a star ⭐.

Feedback and suggestions for improvements are welcome.

---

**Disclaimer:** This project is intended for educational and demonstration purposes. Spam predictions can be incorrect, so users should exercise caution when interacting with suspicious messages or links.

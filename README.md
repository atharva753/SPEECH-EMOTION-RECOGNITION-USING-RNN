# SPEECH-EMOTION-RECOGNITION-USING-RNN
Sentiment analysis and emotion recognition are important tasks in natural language processing. Universal speech representations are a type of pre-trained deep learning model that can be used for these tasks. The system is evaluated on a held-out test set of speech recordings.

## Dataset Description

The project uses the RAVDESS Emotional Speech Audio dataset (Ryerson Audio-Visual Database of Emotional Speech and Song). It contains 2,880 speech clips recorded by 24 actors (12 male, 12 female) expressing eight emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised. Each clip is a WAV file sampled at 48 kHz, lasting about 3–4 seconds. Kaggle Link - https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## Stage 1: Preprocessing and Feature Extraction

• Objective: Convert raw audio into feature arrays suitable for deep learning.
• Methodology:
– Load each .wav file and resample to 16 kHz.
– Extract 40 MFCC coefficients per frame (hop length 512), plus zero-crossing rate, chroma, and spectral contrast.
– Standardize and pad/truncate to a fixed length of 143 frames.
– Save the resulting feature tensor X.npy (shape 2880×143×40) and labels.csv mapping filenames to emotion labels.

## Stage 2: Data Preparation and Label Encoding

• Objective: Prepare numerical datasets for model training.
• Methodology:
– Read X.npy and labels.csv.
– One-hot encode the eight emotion labels into y.npy (shape 2880×8).
– Split into training and testing sets (80/20).
– Save train/test splits if desired; confirm shapes: X\_train (2304×143×40), y\_train (2304×8), X\_test (576×143×40), y\_test (576×8).

## Stage 3: Baseline CRNN Model

• Objective: Establish a baseline Speech Emotion Recognition (SER) accuracy.
• Methodology:
– Reshape features for a TimeDistributed Conv2D + Bidirectional LSTM architecture.
– Model layers: Input → TimeDistributed(Conv2D→BatchNorm→MaxPool→Dropout) ×2 → TimeDistributed(Flatten) → Bi-LSTM(128)→Dropout×2 → Dense(8, softmax).
– Compile with Adam, categorical crossentropy, and track accuracy.
– Train 50 epochs with EarlyStopping (patience 10) and ReduceLROnPlateau.
– Evaluate on test set: achieved ≈71.7% accuracy, indicating overfitting and room for improvement.
– Save baseline\_model.h5.

## Stage 4: Hyperparameter Tuning

• Objective: Tune convolutional filters, dropout rates, LSTM units, dense units, learning rate, and L1/L2 regularization.
• Methodology:
– Implement a HyperModel class for KerasTuner’s RandomSearch.
– Define search space: filters1 (32–64), filters2 (64–128), lstm\_units (128–256), dense\_units (64–192), dropout (0.2–0.6), reg\_factor (1e-5–1e-3), learning\_rate (1e-4–1e-2), noise\_std (0–0.1).
– Run max\_trials=15, epochs=30 with validation split 0.2 and EarlyStopping (patience 5).
– Observe best validation accuracy plateauing around 30–40%, revealing limitations of from-scratch CRNN on MFCC features alone.
– Save notebook4\_best\_model.h5 for comparison.

## Stage 5: Advanced Optimization with Pre-trained Embeddings

• Objective: Leverage self-supervised pre-trained speech models and stronger regularization/augmentation to break the 70% barrier.
• Methodology:
– Use Facebook’s Wav2Vec 2.0 to extract 768-dim embeddings for each audio clip (mean-pooled over time).
– Save embeddings as X\_wav2vec.npy (shape 2880×768).
– Build a lightweight MLP classifier: Dense(512→ReLU→BatchNorm→Dropout) → Dense(256→ReLU→Dropout) → Dense(128→ReLU→Dropout) → Dense(8→softmax) with L1/L2 regularization.
– Train 40 epochs with a validation split of 0.2, EarlyStopping, and learning rate reduction.
– Evaluate on test set; expected improvement, often achieving 85–90%+ accuracy with this approach.
– Save optimized\_model.h5.

## Comparison to Previous Methodology

• Baseline (Notebook 3) MFCC → CRNN: 71.7% test accuracy.
• Hyperparameter Tuning (Notebook 4) yielded no major gains (max \~40% val\_accuracy).
• Advanced (Notebook 5) using Wav2Vec 2.0 embeddings and regularized MLP significantly improves generalization, targeting >90% accuracy.

## Streamlit App: Interactive Demo

• Place `app.py`, `optimized_model.h5`, and `label_mapping.npy` in your working directory.
• Ensure dependencies are installed:
`pip install streamlit tensorflow librosa soundfile numpy`
• Run with:
`streamlit run app.py`

1. Upload a RAVDESS .wav file.
2. App displays audio player, MFCC plot, and emotion prediction with confidence.
3. Probability bar chart for all eight emotions.

## Future Scope

• Data: Incorporate larger or multi-language emotional speech corpora (CREMA-D, EmoDB, SAVEE).
• Models: Explore transformer architectures (Audio Spectrogram Transformer, HuBERT fine-tuning), end-to-end 1D CNN on raw waveforms.
• Augmentation: Add SpecAugment, time-stretch, pitch-shift, mixup, noise injection.
• Cross-Validation & Ensembling: k-fold training and ensemble voting or averaging for robust predictions.
• Multimodal Fusion: Combine audio with facial expression analysis for video-based emotion recognition.
• Deployment: Optimize and deploy as a web or mobile app with real-time inference and low-latency streaming.


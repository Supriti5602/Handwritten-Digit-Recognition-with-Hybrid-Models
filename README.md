# Handwritten-Digit-Recognition-with-Hybrid-Models

## Overview

This project compares **traditional machine learning, deep learning, and hybrid models** for recognizing handwritten digits from **0 to 9**. Six models are evaluated on MNIST, followed by robustness testing, error analysis, prediction on a photographed digit, and deployment of the CNN-SVM model in an interactive **Pygame drawing application**.

The best reported model, **CNN-SVM**, achieved **99.28% test accuracy**, correctly classifying 9,928 of 10,000 MNIST test images.

## Features

- Comparison of KNN, SVM, MLP, CNN, KNN-SVM, and CNN-SVM.
- Accuracy, precision, recall, F1-score, and confusion-matrix evaluation.
- Robustness testing with Gaussian noise, rotation, and pixel shifts.
- Analysis of commonly confused digits and misclassified images.
- Preprocessing of photographed handwriting into MNIST-style input.
- Saved CNN and SVM components for reuse without retraining.
- Interactive drawing interface with automatic prediction and a processed-image preview.

## Technologies

Python, TensorFlow / Keras , scikit-learn, NumPy and pandas, Matplotlib and Seaborn, OpenCV and Pillow, Joblib, Pygame, Jupyter Notebook 

## Dataset and Preprocessing

The project uses **MNIST**, containing 70,000 grayscale digit images of size **28 × 28 pixels**.

| Split | Images |
| --- | --- |
| Training | 48,000 |
| Validation | 12,000 |
| Testing | 10,000 |

The original 60,000-image training set is divided using a stratified 80/20 split with `random_state=42`. The original test set is retained for evaluation.

Preprocessing includes:

1. Convert image values to `float32` and divide by 255 to obtain values in `[0, 1]`.
2. Flatten images into **784-dimensional vectors** for KNN, SVM, KNN-SVM, and MLP.
3. Reshape images to **28 × 28 × 1** for CNN and CNN-SVM.
4. One-hot encode labels for neural-network training.

MNIST is loaded using `tensorflow.keras.datasets.mnist.load_data()`.

## Models

| Model | Configuration |
| --- | --- |
| KNN | 3 nearest neighbours using flattened pixels |
| SVM | RBF kernel, `C=2.0`, `gamma="scale"` |
| MLP | Dense layers with 256 and 128 ReLU units, dropout of 0.3 after each hidden layer, and 10-class softmax output |
| CNN | Two 3×3 convolutional layers with 32 and 64 filters, max pooling, a 128-unit dense layer, dropout of 0.5, and softmax output |
| KNN-SVM | KNN shortlists three digit classes; pairwise SVMs vote on the final class |
| CNN-SVM | CNN extracts 128-dimensional features; an RBF SVM performs classification |

### Neural-Network Training

The MLP, CNN, and CNN-SVM base network use:

- **Optimizer:** Adam, learning rate `0.001`.
- **Loss:** Categorical cross-entropy.
- **Maximum epochs:** 20.
- **Batch size:** 128.

### KNN-SVM Hybrid

The implementation trains **45 binary SVMs**, one for each pair of digit classes. At prediction time, KNN selects three candidate classes, and only the **three pairwise SVMs** corresponding to those candidates are evaluated. Their predictions are combined through majority voting.

### CNN-SVM Hybrid

1. Train a CNN with a named 128-unit dense layer, `feature_layer`.
2. Create a feature extractor whose output is this intermediate layer.
3. Extract 128-dimensional vectors from the training images.
4. Train an SVM with an RBF kernel, `C=2.0`, and `gamma="scale"` on these vectors.
5. For new images, apply the same preprocessing, extract CNN features, and predict with the SVM.

The saved Keras file contains the full CNN. The application recreates the feature extractor from `feature_layer` rather than using the CNN's softmax output for its final prediction.

## Setup

Install the project dependencies in a Python environment:

```bash
python -m pip install tensorflow scikit-learn numpy pandas matplotlib seaborn opencv-python pillow joblib pygame notebook
```

For architecture diagrams generated with Keras `plot_model`, also install `pydot` and the Graphviz system application:

```bash
python -m pip install pydot
```

## Usage

### Train and Evaluate

1. Start Jupyter from the project directory:

   ```bash
   jupyter notebook
   ```

2. Open the project's training notebook and run the cells in order.
3. The notebook loads MNIST, prepares the splits, trains the models, and generates evaluation plots and tables.
4. Run the model-saving cells to create `cnn_svm_feature_extractor.keras` and `cnn_svm_classifier.pkl`.
5. For the photographed-digit demonstration, place `Four.jpg` in the working directory or update `image_path` in the notebook.

### Run the Drawing Application

Place both saved model files beside `app.py`, then run from that directory:

```bash
python app.py
```
Draw a digit and the application displays the predicted digit at the top left and a magnified preview of the processed **28 × 28** input at the top right. Draw one digit at a time and clear the canvas before starting a new digit.

## Test Results

All six models were evaluated on the same **10,000-image MNIST test set**.

| Model | Test Accuracy | Misclassified Images |
| --- | --- | --- |
| CNN-SVM | **99.28%** | **72** |
| CNN | 99.19% | 81 |
| SVM | 98.13% | 187 |
| MLP | 98.07% | 193 |
| KNN-SVM | 98.04% | 196 |
| KNN | 96.85% | 315 |


## Robustness Analysis

The test images were modified under three conditions:

- **Gaussian noise:** Standard deviation `0.15`, with values clipped to `[0, 1]`.
- **Rotation:** 10 degrees.
- **Translation:** A two-pixel shift in both horizontal and vertical directions.

| Model | Original | Noisy | Rotated | Shifted |
| --- | --- | --- | --- | --- |
| KNN | 96.85% | 96.94% | 95.39% | 42.88% |
| SVM | 98.13% | 97.33% | 96.69% | 52.96% |
| MLP | 98.07% | 93.25% | 96.59% | 54.99% |
| CNN | 99.19% | 98.97% | 98.53% | 80.84% |
| KNN-SVM | 98.04% | 97.39% | 96.57% | 51.02% |
| CNN-SVM | **99.28%** | **99.14%** | **98.74%** | **87.86%** |

CNN-SVM performed best under all tested conditions.

## Error Analysis

The notebook uses confusion matrices, per-class error rates, and visual inspection of incorrect predictions.

For CNN-SVM:

- **72 images** were misclassified, giving an overall error rate of **0.72%**.
- The most frequent directional confusions were **2 → 7** and **7 → 2**, with five cases each.
- Digit **9** had the highest per-class error rate, approximately **1.29%**.
- All **1,135** test examples of digit **1** were classified correctly in the reported run.

## Photographed-Digit Preprocessing

The report demonstrates prediction on a photographed handwritten **4**. Direct resizing produced a prediction of **0**; after preprocessing, the model predicted **4**.

The photograph pipeline performs:

1. Grayscale conversion.
2. Gaussian blur with a 5×5 kernel.
3. Binary inverse thresholding at 150.
4. Bounding-box cropping around nonzero pixels.
5. Black padding of 20 pixels on each side.
6. Resizing to 28 × 28 and normalization to `[0, 1]`.
7. CNN feature extraction and SVM prediction.

The drawing application uses a related pipeline adapted to its canvas: inversion, thresholding at 50, cropping, padding, resizing, and normalization. Empty drawings are skipped.

The photographed example illustrates the effect of preprocessing; it is not a separate quantitative benchmark of real-world accuracy.



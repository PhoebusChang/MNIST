# MNIST
Basic neural network that takes numerical input to generate images, and image input to generate numbers.

---

## Bidirectional Neural Network Playground

This project demonstrates two powerful neural network models for the MNIST dataset:
- **Image-to-Label:** Classifies handwritten digit images (0–9).
- **Label-to-Image:** Generates realistic digit images from numerical labels.

Built with [PyTorch Lightning](https://www.pytorchlightning.ai/) for clean, modular training.

---

## Features

- **Digit Classifier:** Accurately predicts the digit in any MNIST-style image.
- **Digit Generator:** Creates new, realistic digit images from any label (0–9).
- **Easy Training & Inference:** Simple scripts for number prediction and image generation.

---

## Getting Started

### 1. Install Dependencies

```sh
pip install torch torchvision pytorch-lightning matplotlib pillow
```

### 2. Train the Models

#### Train the Classifier

```sh
python main.py
```

#### Train the Generator

```sh
python reverse.py
```

---

## Usage

### Predict a Digit from an Image

1. Place your image as `test.png` (28x28 grayscale, normalized like MNIST).
2. Run:

```sh
python main.py
```

3. When prompted, press Enter to predict the digit.

### Generate a Digit Image from a Label

1. Run:

```sh
python reverse.py
```

2. Enter a digit (0–9) when prompted.  
   The generated image will be saved as `generated.png`.

---

## File Structure

- `main.py`: Image-to-label classifier (predicts digits from images).
- `reverse.py`: Label-to-image generator (creates images from digits).
- `generated.png`, `test.png`: Example output and input images.
- `README.md`: Project documentation.

---

## Model Details

### Classifier (`SimpleNN` in main.py)

- 3-layer fully connected neural network.
- Trained on MNIST for digit classification.

### Generator (`LabelToImageGenerator` in reverse.py)

- Conditional generator using label embeddings and latent vectors.
- Produces 28x28 grayscale digit images.

---

## Example

**Generate a "8":**
```sh
python reverse.py
# Enter digit 8
```
Result (`generated.png`):

![Generated digit 8](generated.png)

**Classify an Image:**
```sh
python main.py
# Press Enter to classify 'test.png'
```
Input (`test.png`):

![Test image](test.png)

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
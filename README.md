# DFD-using-PSD

A beginner-friendly project that explores how frequency-domain analysis (using Power Spectrum Density, PSD) can help detect face forgeries (deepfakes) with a MobileNetV2 classifier.  
We use the DFDC dataset, extract frames, identify and crop faces, analyze their 2D PSD, and observe interesting patterns that help spot GAN-based deepfakes.

---

## Introduction

With the rise of deepfake technology, it’s becoming easier to generate realistic but fake videos and images of people. Detecting these forgeries is a challenge, especially as the technology improves.

In this project, we use a unique approach:  
- **Convert face images into the frequency domain** using Power Spectrum Density (PSD).
- **Observe the PSD patterns** for both real and GAN-generated (forged) faces.
- **Classify** the images as "deepfake" or "real" using a MobileNetV2 neural network.

---

## What is Power Spectrum Density (PSD) and Frequency Domain?

Most images are viewed and processed in the "spatial domain" (pixels, colors, etc.). But images can also be represented in the "frequency domain", which shows how quickly pixel values change across the image.

- **PSD (Power Spectrum Density)** is a way to measure the strength of different frequency components (patterns) in an image, separately along the x and y axes.
- **Frequency domain analysis** can reveal subtle artifacts introduced by image generation methods like GANs, which are often hard to see in the original image.

**Example scenario:**  
Imagine looking at a face image — you see eyes, nose, mouth (spatial domain). But if you transform it into the frequency domain using PSD, you might notice repeating patterns or unusual spikes that don’t usually occur in real faces. These can be clues that the face is a deepfake.

|Types | image | PSD |
|-|-|-|
|real image| ![realface1](https://github.com/user-attachments/assets/08f4e6a9-1498-4d49-a424-521cab253948) | ![realrfft1](https://github.com/user-attachments/assets/891e3544-ed21-4261-b00a-1f7ef6c53b4e) |
|fake image| ![fakeface1](https://github.com/user-attachments/assets/2bbfd582-a896-498f-a395-a2dbe7ff75da) | ![fakerfft1](https://github.com/user-attachments/assets/ac48e075-d68d-44a0-a7c4-7c6513c07e61) |



---

## Project Workflow

1. **Extract frames from DFDC dataset videos.**
2. **Detect faces in each frame.**
3. **Crop the detected faces.**
4. **Compute and plot the 2D PSD for each cropped face.**
5. **Observe and analyze PSD patterns:**
   - For real faces, the PSD is typically smooth.
   - For GAN-generated deepfakes (especially frontal faces), there are significant variations in the PSD along both x and y directions.
6. **Feed the PSD images into MobileNetV2 for classification:**
   - The model learns to distinguish between real and forged faces based on their PSD patterns.

---

## Example Scenario

- You have a video from the DFDC dataset.
- Frame 100 is extracted, and a face is detected and cropped.
- You calculate the 2D PSD of the cropped face.
- In the PSD plot, you notice strong spikes or patterns not seen in genuine faces—this is a possible sign of GAN-based forgery.
- MobileNetV2 uses these PSD images to learn what "real" and "fake" look like in the frequency domain and classifies new faces accordingly.

---

## Why Use PSD for Deepfake Detection?

Deepfake (GAN) methods often introduce repetitive patterns or artifacts in the frequency domain, especially for frontal faces. These are hard for humans to spot in the original image but become clear when visualized with PSD.

By analyzing these patterns, we can give our classifier (MobileNetV2) a better chance of spotting fakes—even the sophisticated ones!

---

## How to Run This Project

### Prerequisites

- Python 3.x
- Jupyter Notebook
- numpy, matplotlib, opencv-python, tensorflow/keras
- DFDC dataset

### Steps

1. **Clone this repository:**
   ```sh
   git clone https://github.com/parth721/DFD-using-PSD.git
   cd DFD-using-PSD
   ```


2. **Prepare your data:**
   - Download/extract the DFDC dataset.
   - Organize video files or image frames as needed.

3. **Open the Jupyter Notebook and follow along:**
   ```sh
   jupyter notebook
   ```
   - Use the provided notebooks to extract frames, crop faces, compute PSD, and train/test the MobileNetV2 model.

---

## Key Files and Folders

- `notebooks/` – Jupyter notebooks for each step (frame extraction, face cropping, PSD calculation, training, etc.) (Private)
- `evaluationodPSDmethod` – Helper functions for PSD calculation. (public)
- `models/` – Scripts or checkpoints for MobileNetV2 training and evaluation. (public) 

---

## Results & Observations

- **Frontal deepfake faces** often show strong, unusual variations in their PSD along both axes—a telltale sign of GAN-generated forgeries.
- MobileNetV2 achieves good accuracy by learning from these PSD patterns, even when the original images look realistic to humans.

---

## Further Reading

- [What is a Power Spectrum? (Practical tutorial)](https://www.l3harrisgeospatial.com/docs/powerspectrum.html)
- [DFDC dataset info](https://ai.facebook.com/datasets/dfdc/)
- [MobileNetV2 Overview](https://arxiv.org/abs/1801.04381)

---

## License

MIT License

---

## Contact

Maintainer: [@parth721](https://github.com/parth721)  
If you have questions or want to contribute, feel free to reach out or submit a pull request!

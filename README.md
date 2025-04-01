# Ultrasound-Based Implant Detection and Classification (SNUH)

This repository contains code and documentation for an AI model developed at **Seoul National University Hospital (SNUH)** to detect and classify **types of breast implants** from ultrasound images. This project was conducted as part of a clinical AI research initiative and applied in real-world medical scenarios.

---

## Project Overview

- **Purpose**: Automatically detect and classify the type of implant used in post-operative breast ultrasound images.
- **Significance**: Different implant types are used depending on patient condition and purpose. Accurate identification of the implant type is critical for diagnosis, monitoring, and further treatment.

---

## Methodology

This is a **2-stage pipeline**:

1. **Detection (Stage 1)**  
   - A **YOLOv5 model** is used to detect the region of interest (implant location) from the raw ultrasound image.
   - Detected regions are cropped and passed to the classification stage.

2. **Classification (Stage 2)**  
   - Cropped images are classified into implant types using deep learning-based models:
     - **ResNet**
     - **VGGNet**
     - **EfficientNet**
     - **Vision Transformer (ViT)**
     - **Xception**
   - Background information is removed during this stage to reduce noise and improve performance.

---

## Dataset

- **Type**: Ultrasound DICOM files
- **Preprocessing**:
  - DICOM parsing and grayscale normalization
  - Object detection → ROI cropping → classification-ready input generation
- **Grad-CAM** is used for model interpretability to visualize learned regions

---

## Key Outcomes

- Achieved **79.1% classification accuracy** by combining detection and classification stages.
- Enhanced explainability and trustworthiness in AI-based ultrasound analysis.
- Demonstrated potential applicability to **real-time object detection and interpretation in aviation and other fields**.

---

## Directory Structure (Example)




---

## Research Experience Summary

- Built a robust two-stage medical imaging pipeline combining detection and classification
- Solved real-world challenges such as background noise using preprocessing and region cropping
- Applied visualization (Grad-CAM) to interpret model decisions and increase clinical trust
- Developed transferable AI technologies for use in other real-time detection environments (e.g., aviation)

---

## Related Tools

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Keras Applications](https://keras.io/api/applications/)
- [timm - PyTorch image models](https://github.com/huggingface/pytorch-image-models)

---

## License

This project is intended for research and educational use. Please contact the project maintainer for details regarding data access or clinical use.

# Face Mask Detection Inference with ONNX and Rust

This is a very simple project to use Rust for inference ONNX models.
This repository has two parts. 
- The first part contains model implementation by PyTorch with the [Kaggle face mask detection dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) and export model to ONNX. 
- The second part is the ONNX model inference with Rust using the Tract-ONNX library.
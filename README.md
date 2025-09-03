# Neural Network Framework from Scratch

This project implements a foundational deep learning framework in Python, covering essential concepts from data handling and array manipulation to advanced topics like regularization and recurrent neural networks. The entire framework is built from scratch **without the use of high-level machine learning libraries**, emphasizing a deep understanding of the underlying algorithms.

---

## 🚀 Key Features

### 🔹 Foundational Components
- Core building blocks such as a flexible **base layer** and various **optimizers**.

### 🔹 Vectorized Operations
- All implementations, from basic array patterns to a robust **image data generator**, rely on **NumPy** for efficient, vectorized operations without the use of loops.

### 🔹 Data Handling & Augmentation
- **ImageGenerator** capable of:
  - Dataset loading  
  - Batching  
  - On-the-fly data augmentation with rigid transformations like **random mirroring** and **rotation**  

### 🔹 Regularization Techniques
- Includes strategies to combat overfitting:
  - **L1/L2 Regularizers**  
  - **Dropout**  
  - **Batch Normalization**

### 🔹 Advanced Architectures
- **Convolutional Layers** → Core building blocks for CNNs  
- **Recurrent Layers** → Implements **Elman RNNs** and **LSTMs** for sequential data  
- **LeNet** → Build a variant of the classic **LeNet architecture**  

### 🔹 Serialization
- Models can be **saved and loaded** using Python’s `pickle` module.  

---

## 📌 Summary
This framework offers a bottom-up understanding of how neural networks work, from mathematical foundations to training advanced architectures, making it both a learning tool and a flexible experimental framework.  

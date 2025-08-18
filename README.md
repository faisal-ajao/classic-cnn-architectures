# Classic CNN Architectures in Keras 📚🤖

This repository contains implementations of several **classic convolutional neural network (CNN) architectures** in Keras and Python.  
It serves as a **portfolio project** for demonstrating understanding of deep learning, CNN design, and modern computer vision workflows.

Each architecture includes both **Python scripts (.py)** and **Jupyter notebooks (.ipynb)** for easy experimentation and learning.

---

## 📂 Repository Structure

- **AlexNet** 🟢  
  `alexnet_implementation_in_keras.ipynb`  
  `alexnet_implementation_in_keras.py`  
  *Description:* One of the pioneering deep CNNs for large-scale image classification (ImageNet). Includes convolutional, max-pooling, and fully connected layers with ReLU activations and Dropout.

- **Inception** 🌐  
  `inception_implementation_in_keras.ipynb`  
  `inception_implementation_in_keras.py`  
  *Description:* Implements GoogLeNet/Inception modules with multi-scale convolutions and global average pooling. Designed for deeper networks with reduced computational cost.

- **LeNet-5** 🎯  
  `lenet5_mnist.ipynb`  
  `lenet5_mnist.py`  
  *Description:* Classic CNN for handwritten digit recognition (MNIST dataset). Lightweight architecture with convolutional, pooling, and fully connected layers. Ideal for learning CNN basics.

- **ResNet-50** 🚀  
  `resnet50_implementation_in_keras.ipynb`  
  `resnet50_implementation_in_keras.py`  
  *Description:* Residual network with 50 layers. Introduces skip connections to mitigate vanishing gradients and enable training of very deep networks.

- **ResNet (CIFAR 10)** 🔹  
  `resnet_cifar_10.ipynb`  
  `resnet_cifar_10.py`  
  *Description:* Residual network adapted for CIFAR-10 dataset. Demonstrates practical training on small-scale image datasets.

- **VGGNet** 🏗️  
  `vggnet_implementation_in_keras.ipynb`  
  `vggnet_implementation_in_keras.py`  
  *Description:* Deep CNN architectures (VGG-16 and VGG-19). Focuses on uniform 3x3 convolution layers and fully connected layers for image classification (ImageNet).

---

## ⚡ Key Features

- Clean and modular **Keras implementations** of classic CNNs.  
- Includes both notebooks for exploration and scripts for deployment.  
- Readable and well-commented code demonstrating CNN architecture and deep learning concepts.
- Compatible with **TensorFlow 2.x / Keras API**.  
- Ready for **ImageNet, MNIST, CIFAR** dataset experiments.

---

## 🚀 Usage

1. Clone the repository:

```bash
git clone https://github.com/AjaoFaisal/classic-cnn-architectures.git
cd classic-cnn-architectures
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook of your choice (`.ipynb`) or run the corresponding script (`.py`) to train or test models.

---

## 📖 References

- **AlexNet:** Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*  
- **Inception/GoogLeNet:** Szegedy, C. et al. (2015). *Going Deeper with Convolutions*  
- **LeNet-5:** LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition*  
- **ResNet:** He, K. et al. (2016). *Deep Residual Learning for Image Recognition*  
- **VGGNet:** Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*

---

## 📌 Notes

- Designed for learning, experimentation, and portfolio presentation.  
- All models implemented from scratch without pre-trained weights unless specified.  
- Performance may vary depending on dataset, hardware, and training configuration.

---

## ⭐ Contribution

Contributions are welcome! Suggest improvements, add new architectures, or optimize existing implementations.

---

## 🔗 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

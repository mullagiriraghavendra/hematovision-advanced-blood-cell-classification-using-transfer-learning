# HematoVision-Advanced-Blood-Cell-Classification-Using-Transfer-Learning
## Fast & Efficient AI-Powered Medical Image Analysis

A lightweight Flask web application that uses **MobileNetV2 transfer learning** to classify blood cells into 4 types: Eosinophil, Lymphocyte, Monocyte, and Neutrophil.

## 🚀 **Why MobileNetV2?**
- ⚡ **Lightning Fast**: Trains in ~5 minutes (vs hours with VGG16)
- 🏃 **Lightweight**: Perfect for deployment and real-time inference
- 🎯 **Accurate**: Excellent results with efficient architecture
- 💡 **Smart**: Optimized for mobile and web applications

## 📁 **Project Structure**
```
blood-cell-classification/
├── model_training.ipynb         # MobileNetV2 training notebook
├── app.py                       # Flask web application
├── Blood Cell.h5               # Trained MobileNetV2 model
├── requirements.txt            # Optimized dependencies
├── static/
│   └── uploads/               # Uploaded images storage
└── templates/
    ├── home.html             # Upload interface
    └── result.html           # Results display
```

## 🔄 **Project Flow (Following Structure)**

### 1️⃣ **Data Collection And Preparation**
- Download "Blood Cell Images.zip" from Kaggle
- Auto-extract and organize dataset2-master
- Data visualization and augmentation

### 2️⃣ **Split Data And Model Building**  
- **Model**: MobileNetV2 transfer learning
- **Input Size**: 150x150 (optimized for speed)
- **Architecture**: MobileNetV2 + GlobalAveragePooling + Dense layers

### 3️⃣ **Testing Model & Data Prediction**
- **Fast Training**: 5 epochs, ~5 minutes total
- **Evaluation**: Test accuracy and confusion matrix
- **Model Saving**: Export as 'Blood Cell.h5'

### 4️⃣ **Application Building**
- **Flask Backend**: Optimized for 150x150 images
- **HTML Templates**: Modern, responsive UI
- **Real-time Prediction**: Upload → Analyze → Results

## 🛠️ **Quick Setup**

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Get Dataset**
1. Download: https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data
2. Extract "Blood Cell Images.zip" to project folder
3. Should create `dataset2-master/` and `dataset-master/` folders

### **Step 3: Train Model (Fast!)**
```bash
# Open and run the training notebook
jupyter notebook model_training.ipynb
# Training completes in ~5 minutes!
```

### **Step 4: Run Web Application**
```bash
python app.py
```

### **Step 5: Test the System**
- Open: `http://localhost:5000`
- Upload a blood cell image
- Get instant AI classification!

## 🔬 **Technical Specifications**

| Component | Specification |
|-----------|---------------|
| **Base Model** | MobileNetV2 (ImageNet weights) |
| **Input Size** | 150×150×3 |
| **Architecture** | Transfer Learning + Custom Head |
| **Training Time** | ~5 minutes |
| **Classes** | 4 (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL) |
| **Framework** | TensorFlow/Keras + Flask |
| **Deployment** | Web-based, real-time inference |

## 🎯 **Model Architecture**
```
MobileNetV2 (frozen)
    ↓
GlobalAveragePooling2D
    ↓  
Dense(64, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(4, activation='softmax')
```

## 📊 **Performance Optimizations**
- **Image Size**: 150×150 (vs 224×224) = ~70% speed boost
- **Batch Size**: 64 (vs 32) = 50% fewer training steps  
- **Model**: MobileNetV2 (vs VGG16) = ~80% faster training
- **Epochs**: 5 (vs 20) = Quick convergence

## 🔬 **Blood Cell Types Detected**

| Cell Type | Description | Function |
|-----------|-------------|----------|
| **EOSINOPHIL** | Fight parasites & allergic reactions | 1-4% of WBCs |
| **LYMPHOCYTE** | T cells, B cells, NK cells | 20-40% of WBCs |
| **MONOCYTE** | Become macrophages & dendritic cells | 2-8% of WBCs |
| **NEUTROPHIL** | First responders to infections | 50-70% of WBCs |

## 🌐 **Web Interface Features**
- ✅ **Drag & Drop Upload**: Easy image selection
- ✅ **Real-time Processing**: Results in seconds  
- ✅ **Confidence Scores**: Shows prediction certainty
- ✅ **Responsive Design**: Works on mobile & desktop
- ✅ **Visual Results**: Image + classification display

## 🚀 **Why This Approach Works**
1. **Fast Development**: MobileNetV2 trains quickly
2. **Production Ready**: Lightweight for deployment
3. **High Accuracy**: Transfer learning leverages ImageNet knowledge
4. **User Friendly**: Simple web interface
5. **Scalable**: Easy to add more cell types

## 🔧 **Troubleshooting**

**If training is slow:**
- Reduce batch size to 32
- Use smaller image size (128×128)
- Check if GPU is available

**If accuracy is low:**
- Increase epochs to 10
- Add more data augmentation
- Try unfreezing some MobileNetV2 layers

**If Flask app errors:**
- Verify 'Blood Cell.h5' exists
- Check image preprocessing matches training (150×150)
- Ensure all dependencies installed

## 📈 **Future Enhancements**
- Add more blood cell types
- Implement confidence thresholding
- Add batch processing capability
- Mobile app development
- API endpoints for integration

---

**🎯 Fast, Accurate, and Production-Ready Blood Cell Classification!**
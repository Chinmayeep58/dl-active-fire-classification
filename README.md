# 🛰️ Satellite-Based Active Fire Classification

A Streamlit-based application for **active fire source classification using satellite imagery**. The project combines **ESRI RGB imagery** and **Sentinel-2 multispectral data** with deep learning models to classify active fire locations into different source categories.

Project demo link: (https://huggingface.co/spaces/chinmayeep2385/fire_classification_3)

## 📌 Project Overview

The application provides an end-to-end workflow for:

1. Extracting satellite imagery for active fire locations.
2. Preparing and labeling the extracted datasets.
3. Loading trained deep learning models.
4. Predicting the source/class of active fires.
5. Visualizing predictions on an interactive map.
6. Using manually verified labels for feedback-based model training.

The application is built using **Streamlit** and is organized into multiple pages for each stage of the workflow.

---

## 🗂️ Project Structure

```text
project/
│
├── streamlit_app.py
├── global_styles.py
│
├── 0_Home.py
├── 1_ESRI_Extraction.py
├── 1_Sent_Extraction.py
├── 2_Labeling.py
├── 3_Model_Prediction.py
├── 4_Feedback_Training.py
│
├── requirements.txt
└── README.md
```

---

## 🔗 Models

The trained models and datasets used in this project are available at the following links.

### 🤖 Trained Models

| Model | Description | Binary Model | Multiclass Model |
|---|---|---|---|
| ESRI Model | ResNet-based model trained on ESRI RGB imagery | [Model Link](https://www.kaggle.com/models/chinmayeep2385/esri-binary-class) | [Model Link](https://www.kaggle.com/models/chinmayeep2385/esri-multi-class) |
| Sentinel-2 Model | ResNet-based model using Sentinel-2 multispectral imagery | [Model Link](https://www.kaggle.com/models/chinmayeep2385/sent2-comb-binary) | [Model Link](https://www.kaggle.com/models/chinmayeep2385/sent2-comb-multiclass) |
| Combined Model | Multimodal model combining ESRI and Sentinel-2 features | [Model Link](https://www.kaggle.com/models/chinmayeep2385/comb-binary) | [Model Link](https://www.kaggle.com/models/chinmayeep2385/comb-multiclass) |

---

## 📄 Main Files

### `streamlit_app.py`

The main Streamlit application file.

It:

* Configures the Streamlit page.
* Loads the global CSS styling.
* Initializes the application theme.
* Defines the different application pages.
* Controls navigation between the different stages of the workflow.

The application pages are:

```text
Home
   ↓
ESRI Extraction
   ↓
Sentinel-2 Extraction
   ↓
Model/Dataset Loading and Labeling
   ↓
Model Prediction and Map Plotting
   ↓
Feedback Training
```

---

### `global_styles.py`

Contains the global CSS styling used throughout the Streamlit application.

It is responsible for maintaining a consistent visual appearance across all pages.

---

### `0_Home.py`

The **Home** page of the application.

It provides an introduction to the project and gives users an overview of the active fire classification workflow.

---

### `1_ESRI_Extraction.py`

Responsible for **ESRI satellite imagery extraction**.

It extracts/crops ESRI imagery corresponding to the selected active fire locations and prepares the resulting imagery for further processing.

---

### `1_Sent_Extraction.py`

Responsible for **Sentinel-2 imagery extraction**.

It retrieves the required Sentinel-2 imagery and spectral bands for the selected fire locations and prepares the data for model usage.

The project uses Sentinel-2 observations from different temporal periods to capture additional spectral and temporal information.

---

### `2_Labeling.py`

Handles **dataset and model loading and manual labeling**.

This page allows the user to:

* Load datasets.
* Load trained models.
* View satellite images.
* Generate model predictions.
* Manually verify or correct predictions.
* Store verified labels for further model improvement.

---

### `3_Model_Prediction.py`

Responsible for **model inference and geographical visualization**.

It:

* Loads the trained models.
* Generates predictions for active fire locations.
* Displays predicted classes and confidence scores.
* Plots predictions on an interactive map.
* Allows users to inspect individual fire locations and their associated information.

---

### `4_Feedback_Training.py`

Handles **feedback-based model training**.

Verified labels obtained from the labeling stage can be used to improve the existing models.

Instead of training the model completely from scratch, the feedback data can be used for **fine-tuning** the trained model.

---

## 🔄 Overall Workflow

```text
Active Fire Locations
        │
        ▼
ESRI Extraction ──────► ESRI RGB Images
        │
        ▼
Sentinel-2 Extraction ► Sentinel-2 Images
        │
        ▼
Dataset & Model Loading
        │
        ▼
Manual Labeling / Verification
        │
        ▼
Model Prediction
        │
        ▼
Map Visualization
        │
        ▼
Feedback Training
        │
        ▼
Improved Model
```

---

## 🧠 Deep Learning

The project uses **ResNet-based deep learning models** for satellite image classification.

The models utilize:

* ESRI RGB imagery
* Sentinel-2 multispectral imagery
* Multi-temporal Sentinel-2 observations
* Multimodal feature representations

The final multimodal approach combines information from different satellite sources to improve active fire source classification.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* PyTorch
* Torchvision
* ResNet-18
* Rasterio
* GeoPandas
* NumPy
* Pandas
* Folium
* Streamlit-Folium
* Sentinel-2
* ESRI World Imagery

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd <project-directory>
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run streamlit_app.py
```

The application will open in the browser and provide navigation between the different stages of the satellite data processing and active fire classification workflow.

---

## 🎯 Project Goal

The primary goal of this project is to develop an **interactive multimodal satellite-based system for active fire source classification**, while also providing a feedback mechanism through which manually verified predictions can be used to further improve the trained models.



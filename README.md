# Jonathan-Smith-HCC-AI
Deep Learning ITAI 2376
# Deep Learning Portfolio - ITAI 2376

**Author:** Jonathan Smith  
**Institution:** Houston City College (HCC)  
**Course:** ITAI 2376 – Deep Learning

## 📂 Repository Overview

This repository contains coursework, labs, and the final capstone project for the Deep Learning curriculum. The work progresses from foundational neural network concepts to advanced architectures including Convolutional Neural Networks (CNNs), Generative Diffusion Models, and Autonomous AI Agents.

## 🚀 Featured Projects

### 1. Intelligent Process Automation Agent (IPAA) (Capstone)
**Collaborators:** Jonathan Smith, Miguel Mora  
An autonomous AI agent designed to solve "Contextual Action" problems in administrative automation. Unlike standard chatbots, this agent uses the **ReAct (Reasoning + Acting)** paradigm to perceive user intent, plan execution steps, and perform real-world actions.

* **Key Features:**
    * **Autonomous Planning:** Decomposes high-level requests (e.g., "Schedule a meeting") into atomic executable steps.
    * **RAG Memory:** Utilizes **FAISS** (Vector Database) to retrieve context from past interactions and documents.
    * **Tool Integration:** Connects to the **Google Calendar API** to authenticate and book real events via an OOB (Out-of-Band) flow.
    * **Safety Mechanisms:** Implements "Human-in-the-Loop" validation before executing external API calls.

### 2. Generative Diffusion Model
**Technology:** PyTorch, U-Net, Einops  
A deep learning model implemented from scratch to generate realistic handwritten digits (MNIST) by reversing a noise-addition process.

* **Highlights:**
    * Built a U-Net architecture to predict noise residuals.
    * Implemented linear noise schedulers for forward and reverse diffusion processes.
    * Visualized the denoising process, transforming pure static into recognizable digits over 100+ timesteps.

### 3. High-Performance CNN Classifier
**Technology:** TensorFlow, Keras  
A specialized Convolutional Neural Network (CNN) designed for computer vision tasks.

* **Performance:** Achieved **99.15% accuracy** on the MNIST test set.
* **Architecture:** Utilized `Conv2D` layers for feature extraction (edges, curves) and `MaxPooling2D` for spatial downsampling, demonstrating the advantage over dense networks for image data.

## 📝 Theoretical Analysis & Research

* **NLP & "Arrival":** A comparative analysis of the film *Arrival* and modern Natural Language Processing, exploring linguistic relativity, lexical ambiguity, and non-linear cognition in AI communication.
* **The Neural Network Zoo:** An exploration of CNN architectures through the metaphor of a cheetah ("Fast and focused"), contrasting them with RNNs and Transformers.
* **Backpropagation Explained:** A breakdown of how neural networks learn from mistakes, explained via the "Whisper Team" analogy.

## 🛠️ Technologies & Tools

* **Languages:** Python
* **Frameworks:** PyTorch, TensorFlow, Keras
* **Libraries:** FAISS (Vector Search), Einops, PyMuPDF (Document Processing), Matplotlib, Pandas, NumPy
* **APIs:** Google Calendar API, Google Colab (GPU Acceleration)

## 📦 Setup & Installation

To run the notebooks locally, ensure you have Python 3.10+ installed.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/smithj334/Jonathan-Smith-HCC-AI.git](https://github.com/smithj334/Jonathan-Smith-HCC-AI.git)
    cd Jonathan-Smith-HCC-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch tensorflow keras faiss-cpu pymupdf google-api-python-client google-auth-httplib2 google-auth-oauthlib
    ```

3.  **Run the Agent:**
    Open `FN_Notebook_Mora_Miguel_Group3_ITAI2376ipynb.ipynb` in Jupyter or Google Colab to initialize the IPAA agent.

---
*This portfolio represents a journey from foundational deep learning concepts to the engineering of autonomous intelligent systems.*

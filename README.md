# 🛡️ TruthVerify: SaaS-Grade Fake News Detection

![TruthVerify Banner](static/images/saas_hero.png)

TruthVerify is a premium, AI-powered web application designed to combat misinformation in the digital age. Leveraging advanced Natural Language Processing (NLP) and Machine Learning, TruthVerify provides real-time authenticity scoring for news articles and headlines.

## ✨ Key Features

- **🎯 High-Precision Analysis**: Powered by a Decision Tree classifier trained on extensive news datasets.
- **🌓 Dual-Theme Support**: SaaS-level Light and Dark mode toggle with persistent user preferences.
- **🎨 Modern SaaS UI**: A clean, high-end interface featuring glassmorphism, responsive grids, and premium 3D illustrations.
- **⚡ Real-time Verification**: Instant analysis with latency benchmarks under 300ms.
- **📱 Fully Responsive**: Seamless experience across mobile, tablet, and desktop devices.

## 🚀 Technical Stack

- **Backend**: Python 3.12+, Flask
- **Machine Learning**: Scikit-learn (Decision Tree Classifier), TF-IDF Vectorization
- **Data Handling**: Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), Inter Typeface

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- `pip` (Python package manager)

### Local Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sriram-93/Fakenews.git
   cd Fakenews
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python3 app.py
   ```
   The application will be available at `http://127.0.0.1:5000`.

## 🧠 Model Training
The core of TruthVerify is a Decision Tree model trained on labeled news datasets. Text content is vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)** to extract linguistic patterns unique to misinformation.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
Built with ❤️ for a more informed digital world.
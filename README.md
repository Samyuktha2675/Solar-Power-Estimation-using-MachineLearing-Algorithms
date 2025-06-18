#**🌞 Solar Power Estimation Using Machine Learning**

A comprehensive project that explores the estimation of solar power using advanced hybrid machine learning models. This project combines Random Forest (RF), XGBoost, and Long Short-Term Memory (LSTM) networks, along with advanced optimization strategies like Bayesian Optimization and Particle Swarm Optimization, to effectively model both spatial and temporal dependencies in solar power data.

**📌Table of Contents**

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

#**RESULT**
![download (1)](https://github.com/user-attachments/assets/45f95b09-eb80-41ff-9edb-ca978bde1c58)
![download (2)](https://github.com/user-attachments/assets/bcca03be-6a46-4f77-9324-e385ef181966)



#**📖Project Overview**

This project aims to build a hybrid machine learning model to predict solar power generation more accurately. It focuses on leveraging the strengths of ensemble learning (RF and XGBoost) and deep learning (LSTM) to capture complex patterns in environmental and historical solar data. The methodology includes detailed data preprocessing, feature engineering, and hyperparameter tuning through optimization techniques.

---

#**🚀Features**

- 📊 Real-time solar power prediction using historical and weather data.
- 🤖 Hybrid modeling: Combines Random Forest, XGBoost, and LSTM.
- 🧠 Captures both spatial and temporal patterns.
- 🔍 Performance evaluation with metrics like RMSE, MAE, R².
- 🔧 Optimization via Bayesian Optimization & Particle Swarm Optimization.
- 📈 Visual analytics for data and predictions.

---

#**🧱Architecture**

```text
                +--------------------+
                |  Input Features    |
                | (Weather, Solar,   |
                |  Time, etc.)       |
                +--------+-----------+
                         |
                         v
       +-----------------+-------------------+
       | Ensemble Models (RF & XGBoost)      |
       +-----------------+-------------------+
                         |
                         v
              +----------+----------+
              | Deep Learning (LSTM)|
              +----------+----------+
                         |
                         v
              +----------+----------+
              |   Final Prediction   |
              +---------------------+
````

---

##**📂Dataset**

* **Source**: Public solar energy and meteorological datasets (e.g., NREL, Kaggle, local station data).
* **Attributes**:

  * Solar irradiance
  * Temperature
  * Humidity
  * Wind speed
  * Date & Time

> *Note: Dataset preprocessing scripts are included in the `data/` and `scripts/` folders.*

---

## ⚙️ Installation

1. Clone the repo:

```bash
git clone https://github.com/your-username/solar-power-estimation-ml.git
cd solar-power-estimation-ml
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Usage

Run the model training and evaluation:

```bash
python train_model.py
```

To predict using the trained model:

```bash
python predict.py --input your_input_file.csv
```

---

## 📊 Results

#**MAE**: *6.180190366017883*
#**MSE**: *82.76455966624914*
#**RMSE**: *9.0975029357648*
#**R^2 Score**: *0.9993947806323973*
> The hybrid model significantly outperformed individual models, especially in scenarios with high temporal variability.

---

## 🛠 Technologies Used

* Python 3.x
* Scikit-learn
* XGBoost
* TensorFlow / Keras (for LSTM)
* Optuna / Hyperopt (for optimization)
* Pandas, NumPy, Matplotlib, Seaborn

---

## 👨‍💻 Supervisor

* Supervised by \[Keerthana,Assistant Professor Dept of AI&DS], \[VCET]

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).


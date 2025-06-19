
# 🚗 Car Price Prediction using PyTorch

This project builds a machine learning model using **PyTorch** to predict the selling price of cars based on various features like year, present price, fuel type, and more. It covers the full pipeline from data preprocessing to model training and prediction.

---

## 📌 Overview

- **Framework**: PyTorch
- **Model Type**: Linear Regression
- **Dataset Size**: 258 rows × 9 columns
- **Target Variable**: `Selling_Price`

---

## 📊 Dataset Features

| Column Name     | Description                              |
|-----------------|------------------------------------------|
| Car_Name        | Name of the car (dropped later)          |
| Year            | Year of manufacture                      |
| Present_Price   | Ex-showroom price of the car             |
| Kms_Driven      | Total kilometers driven                  |
| Fuel_Type       | Type of fuel used (Petrol/Diesel/CNG)    |
| Seller_Type     | Dealer or Individual                     |
| Transmission    | Manual or Automatic                      |
| Owner           | Number of previous owners                |
| Selling_Price   | Price at which the car is being sold     |

---

## 🧹 Data Preprocessing

- Dropped non-informative column: `Car_Name`
- Converted categorical columns to numeric using label encoding
- Scaled `Year` and `Selling_Price` using a custom name-based random function
- Converted all data into PyTorch tensors
- Split data into training and validation sets
- Loaded with `TensorDataset` and `DataLoader` for batching

---

## 🧠 Model Architecture

A simple Linear Regression model implemented using PyTorch:

```python
class CarsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self, xb):
        return self.linear(xb)
```

- Loss Function: `L1Loss` (Mean Absolute Error)
- Optimizer: Stochastic Gradient Descent (SGD)

---

## 🔁 Training Strategy

- Used custom training and evaluation loops
- Validation loss printed every 20 epochs
- Trained in two phases:
  - Phase 1: `lr=1e-8`, 90 epochs
  - Phase 2: `lr=1e-9`, 20 epochs

---

## 📈 Evaluation

Sample Validation Loss Trend:

```
Epoch [20], val_loss: 1692.0131
Epoch [40], val_loss: 1119.7253
Epoch [60], val_loss: 638.9708
Epoch [80], val_loss: 357.3529
Epoch [90], val_loss: 317.1693
Epoch [110], val_loss: 7.9774
```

---

## 🔮 Predictions

Example output:
```python
Input: tensor([1955.5, 8.4000, 12000, 0])
Target: tensor([6.9760])
Prediction: tensor([-0.4069])
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/udita2404/CAR_PRICES_PREDICTION.git
cd CAR_PRICES_PREDICTION
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `CAR_PRICES_PREDICTION.ipynb` in Jupyter Notebook or Google Colab.

---

## 📚 Key Learnings

- Hands-on implementation of PyTorch for tabular data
- Deep understanding of custom training loops and gradient updates
- Preprocessing of mixed-type features (categorical + numeric)
- Regression performance analysis with L1 loss

---

## 📄 License


## 👤 Author

Project by [udita2404](https://github.com/udita2404)

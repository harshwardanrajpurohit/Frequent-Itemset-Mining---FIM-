# 🚀 Frequent Itemset Mining (FIM)

### 📊 Market Basket Analysis using Apriori & FP-Growth

---

## 📌 Overview

This project focuses on **Frequent Itemset Mining (FIM)** — a core concept in data mining used to discover relationships between items in large transaction datasets.

It is widely used in:

* 🛒 E-commerce (Amazon recommendations)
* 🏬 Retail analytics
* 🎯 Targeted marketing

👉 Example:
*"Customers who bought a Laptop also bought a Mouse"*

---

## 🎯 Objective

To analyze transaction data and identify:

* Frequently bought item combinations
* Strong association rules
* Efficient algorithm performance

---

## 🧠 Key Concepts

### 🔹 Support (Popularity)

Measures how frequently an itemset appears.

[
Support(A,B) = \frac{Transactions\ containing\ A\ &\ B}{Total\ Transactions}
]

---

### 🔹 Confidence (Reliability)

Probability that a customer buys B if they buy A.

[
Confidence(A \rightarrow B) = \frac{Transactions(A & B)}{Transactions(A)}
]

---

### 🔹 Lift (True Relationship)

Indicates whether items are actually related.

* Lift > 1 → Strong association
* Lift = 1 → Independent

---

## ⚙️ Algorithms Implemented

### 🧪 1. Brute Force

* Generates all possible combinations
* Time Complexity: ❌ Exponential
* Not scalable

---

### ⚡ 2. Apriori Algorithm

* Uses **Anti-Monotone Property**
* Eliminates infrequent items early
* Requires multiple database scans

✔ Better than brute force
❌ Still slow for large datasets

---

### 🚀 3. FP-Growth Algorithm

* Uses **FP-Tree (Compressed Structure)**
* No candidate generation
* Only **2 database scans**

✔ Fast
✔ Memory efficient
✔ Industry standard

---

## 📊 Algorithm Comparison

| Feature      | Brute Force | Apriori     | FP-Growth   |
| ------------ | ----------- | ----------- | ----------- |
| Speed        | ❌ Very Slow | ⚠️ Moderate | 🚀 Fast     |
| Memory Usage | ❌ Very High | ⚠️ High     | ✅ Low       |
| Scalability  | ❌ No        | ⚠️ Limited  | ✅ Excellent |
| Accuracy     | ✅ 100%      | ✅ 100%      | ✅ 100%      |

---

## 🗂️ Project Structure

```
DBPROJECT/
│── app.py
│── run.ipynb
│── transactions_final.csv
│── requirements.txt
│── .gitignore
│── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/harshwardanrajpurohit/Frequent-Itemset-Mining---FIM-.git
cd Frequent-Itemset-Mining---FIM-
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Project

```bash
python app.py
```

---

## 📁 Dataset

* Online Retail Dataset
* Preprocessed into transaction format

---

## 🧪 Features Implemented

* ✅ Data Cleaning
* ✅ Transaction Processing
* ✅ Frequent Itemset Generation
* ✅ Dataset Optimization (1000 rows version)

---

## 📌 Key Findings

* All algorithms produce **same results**
* FP-Growth is significantly **faster and scalable**
* Brute Force is impractical for real-world data

---

## 🎤 Viva / Interview Questions

**Q: Why not use Brute Force?**
👉 Exponential complexity → crashes on large datasets

**Q: Why is FP-Growth better?**
👉 No candidate generation + only 2 scans

**Q: What is Minimum Support?**
👉 Threshold to filter rare itemsets

---

## 🚀 Future Improvements

* 📊 Add data visualizations (graphs & charts)
* 🌐 Build interactive dashboard
* 🤖 Integrate ML-based recommendation system

---

## 👨‍💻 Author

**Harshwardhan Rajpurohit**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

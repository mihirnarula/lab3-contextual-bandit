# Lab 3: Contextual Bandit-Based News Recommendation System

**Name:** Mihir Narula  
**Roll Number:** U20230029  

---

## 📌 Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** system for personalized news recommendation.

- **User categories (User1, User2, User3)** are treated as contexts.
- **News categories (Entertainment, Education, Tech, Crime)** are treated as bandit arms.
- Rewards are generated using the provided `rlcmab-sampler` package.

The system integrates:
- Supervised Learning (User Classification)
- Reinforcement Learning (Contextual Bandits)

---

## 🧹 Data Preprocessing

- Loaded:
  - `news_articles.csv`
  - `train_users.csv`
  - `test_users.csv`
- Removed rows with missing critical fields (headline/category).
- Standardized news categories to:
  - Entertainment
  - Education
  - Tech
  - Crime
- Filled missing age values using median imputation.
- Encoded user labels for classification.

---

## 🧠 User Classification (Context Detection)

A Logistic Regression classifier was trained using:

- age  
- income  
- clicks  
- purchase_amount  

The model was:
- Trained on `train_users.csv`
- Evaluated on `test_users.csv`
- Test accuracy is printed in the notebook output.

This classifier determines the user context before applying the bandit policy.

---

## 🎰 Contextual Bandit Algorithms Implemented

Three contextual bandit strategies were implemented separately for each user context:

### 1️⃣ Epsilon-Greedy
Tested with:
- ε = 0.01
- ε = 0.1
- ε = 0.2

### 2️⃣ Upper Confidence Bound (UCB)
Tested with:
- C = 0.5
- C = 1
- C = 2

### 3️⃣ Softmax
- Temperature parameter τ = 1

---

## 🎯 Arm Mapping

The 12-arm structure follows the lab specification:

- Arms 0–3 → User1  
- Arms 4–7 → User2  
- Arms 8–11 → User3  

Implemented using:

```python
base = context_id * 4
reward_sampler.sample(base + arm)
```

All rewards are obtained strictly using:

```python
reward_sampler.sample(j)
```

No synthetic or hard-coded rewards are used.

---

## 📊 Reinforcement Learning Simulation

- Simulation length: **T = 10,000**
- Generated plots:
  - Average Reward vs Time (per context)
  - Hyperparameter comparison plots
- All plots include:
  - Title
  - X-axis label
  - Y-axis label
  - Legend

Expected reward distributions (Q-values) are printed for each context.

---

## 🤖 Recommendation Engine

The system performs:

1. Classify user → determine context  
2. Select optimal news category using trained bandit policy  
3. Randomly sample an article from that category  
4. Return article headline and category  

---

## ⚙️ How to Run

1. Install Python 3.12  
2. Install required package:
   ```
   pip install rlcmab-sampler==1.0.1
   ```
3. Ensure the `data/` folder is present in the repository root  
4. Open:
   ```
   lab3_results_U20230029.ipynb
   ```
5. Select Python 3.12 kernel  
6. Run all cells (Restart & Run All)

---

## 📈 Key Observations

- Context-aware bandits improve personalization.
- UCB provides stable convergence.
- ε = 0.1 offers a good exploration–exploitation balance.
- Hyperparameter tuning significantly affects performance.

---

## ✅ Checklist Compliance

- Correct branch used  
- Single notebook submission  
- Correct notebook naming  
- Sampler initialized with correct roll number (29)  
- Rewards obtained only via sampler  
- All three algorithms implemented  
- T = 10,000 simulation completed  
- Required plots included and labeled  
- No synthetic reward generation  

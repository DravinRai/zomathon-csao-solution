# Zomathon 2026: Cart Super Add-On (CSAO) Recommendation Engine

**End-to-End ML Solution | Problem Statement 2 (Track 2)**

This repository contains the code and methodology for our production-ready Cart Super Add-On (CSAO) Rail Recommendation System, built for Zomathon 2026. 

Our solution introduces the **Dynamic Meal Arc Engine**, which models the user's cart as a live sequence. Instead of static recommendations, the system fully re-ranks the add-on rail in real-time (< 30ms latency) every time an item is added or removed, ensuring hyper-relevant, context-aware suggestions.

## 🚀 Key Business Impact & Metrics

* **Model Performance:** AUC of **0.9452** (vs. 0.8887 baseline) and NDCG@5 of **0.9244** (+9.2% lift).
* **Projected Revenue:** +9.1% Average Order Value (AOV) lift at a conservative 20% acceptance rate, translating to an estimated **₹3,805 Crore** annual incremental uplift at Zomato's scale.
* **Production Speed:** End-to-end inference pipeline runs in **~20ms** (well within the strict 200ms SLA).

## 🧠 3-Layer Hybrid Architecture

To balance deep personalization, cold-start constraints, and ultra-low latency, the engine uses a 3-layer architecture:

1.  **Candidate Retrieval (Redis):** O(1) lookup of co-purchase history fetching top candidates instantly.
2.  **Contextual Ranker (GBM):** A 200-tree Gradient Boosting Machine (served via ONNX Runtime) that evaluates real-time cart state, temporal signals (hour, weekend), and user budget segments.
3.  **The "AI Edge" Semantic Layer:** Handles day-1 cold-start items and new restaurants. Uses `sentence-transformers` and a custom Food Domain Ontology (mapping items to meal stages: Starter -> Main -> Side -> Dessert -> Drink) to deduce meal completeness without any historical co-purchase data.

*A Post-Processing MMR (Maximal Marginal Relevance) Diversity Filter (λ=0.3) is applied to penalize embedding-space redundancy and ensure varied recommendations.*

## 📊 Datasets

To ensure real-world validity (capturing true sparsity, discount behaviors, and city-tier trends), the model was trained on 4 real-world Kaggle datasets rather than purely synthetic data:
* [Zomato Order History](https://www.kaggle.com/datasets/order-history-zomato)
* [Enhanced Zomato Menus](https://www.kaggle.com/datasets/enhanced-zomato-menus)
* [Cleaned Zomato Menus](https://www.kaggle.com/datasets/cleaned-zomato-menus)
* [Zomato Delivery Dataset](https://www.kaggle.com/datasets/zomato-delivery-dataset)

## 🛠️ Repository Contents

* `CSAO_Recommendation_System.ipynb`: The primary Google Colab notebook containing the complete data preparation, feature engineering, model training (with zero-leakage GroupShuffleSplit), evaluation, and MMR routing logic. 

## ⚙️ How to Run

1. Clone the repository.
2. Open `CSAO_Recommendation_System.ipynb` in Google Colab or your local Jupyter environment.
3. Download the required Kaggle datasets (links above) and update the path configurations in the first cell.
4. Run all cells to execute the end-to-end pipeline (Data Prep -> Feature Engineering -> Training -> Evaluation).

---
*Note: This solution was developed independently for Zomathon 2026. All architecture decisions, model choices, and business projections reflect original analysis.*

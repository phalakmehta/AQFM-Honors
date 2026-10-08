# Air Quality Forecasting: A Simple Guide to Our Testing and Metrics

Air pollution in Delhi is a massive challenge. When we build AI models to predict pollution (like PM2.5 and PM10), we need to test them rigorously before trusting their predictions for real-world decisions, like closing schools or halting construction.

This document explains **how we test our models** and **what the metrics mean** in simple, everyday English.

---

## 1. How We Test the Models

Imagine trying to predict tomorrow's weather. If you train a model on data from 2010 to 2020, you would test it on 2021 to see if it actually works. We do exactly this.

### The Train/Test Split
* **Training Data (The Past):** We feed the model 7 years of historical air quality data (2018 to 2024). The model looks at this data to learn the patterns (e.g., "pollution goes up in winter", or "when the wind is calm, PM2.5 spikes").
* **Testing Data (The Future):** We hide the year 2025 from the model during training. After the model is trained, we ask it to predict the air quality for every single day in 2025. 

### The Forecast Horizons
We care about different timeframes for different types of real-world decisions:
1. **1-hour ahead:** Useful for real-time health alerts (e.g., "Don't go running right now").
2. **6-hours ahead:** Useful for same-day planning.
3. **24-hours ahead:** Crucial for next-day decisions, like closing schools.
4. **72-hours ahead:** Crucial for big government interventions (like GRAP), which need days to implement (e.g., stopping trucks from entering the city).

We test the model on all four of these horizons to see where it excels and where it starts to struggle.

---

## 2. Understanding the "Baselines"

Before we deploy a complex Deep Learning Foundation Model (AQFM), we need to know if it's actually better than simple, existing methods. These simple methods are called **"Baselines."** We must beat them to prove our new model is worth using.

* **Persistence:** The simplest guess. It assumes tomorrow will be exactly the same as today. If it's polluted today, it will be polluted tomorrow. *(This is the absolute minimum standard. If our AI cannot beat Persistence, it is completely broken).*
* **Seasonal Naive:** Predicts that today at 2 PM will be exactly like yesterday at 2 PM. This is good because it captures the daily cycle (traffic hours, temperature changes).
* **Climatology:** Uses the historical average. If we want to predict January 15th at 6 PM, it just takes the average of all past January 15ths at 6 PM. It is a very safe guess for long-term forecasts (like 72 hours).
* **XGBoost:** A very powerful traditional machine learning model. It looks at recent pollution trends and weather to make a guess. It is the "gold standard" of old-school AI that we are trying to beat with our new Foundation Model.
* **IBM TTM (Zero-Shot):** A pre-trained "Foundation Model" for time-series forecasting. We test it "out of the box" without specifically training it on our Delhi dataset. This shows us how well a general-purpose AI performs compared to our upcoming specialized model.

---

## 3. How We Score the Models (The Metrics)

If a student takes an exam, they get a score out of 100. When an AI makes a prediction, we score it based on how far off its guess was from reality. 

### MAE (Mean Absolute Error)
* **What it means:** On average, how many units (µg/m³) was the model's prediction wrong by?
* **Example:** If the true PM2.5 was 100, and the model guessed 120, the error is 20. If it guessed 80, the error is also 20. MAE averages all these errors together.
* **How to interpret:** **Lower is better**. An MAE of 15 is excellent. An MAE of 100 means the model is wildly guessing.

### RMSE (Root Mean Squared Error)
* **What it means:** Similar to MAE, but it heavily penalizes massive mistakes. 
* **Example:** If the model is off by 10 units on five different days, the MAE is 10. But if it is perfectly correct for four days, and off by 50 units on the fifth day, the MAE is still 10, but the **RMSE will be much higher**. 
* **How to interpret:** We want RMSE to be low. If RMSE is much higher than MAE, it means the model occasionally makes catastrophic, huge errors (which is bad for public trust).

### Stratification by AQI (Air Quality Index) Band
* **The Problem:** Delhi is relatively clean in summer (PM2.5 around 40) but extremely polluted in winter (PM2.5 around 400). If a model only predicts "average" numbers (like 100), its overall MAE might look okay because it does well in the summer. But it would completely fail to warn us about the deadly winter smog!
* **The Solution:** We break the results down by "AQI Bands" (Good, Satisfactory, Moderate, Poor, Very Poor, Severe). We check the MAE specifically for the "Severe" days to make sure the model is actually predicting the dangerous episodes, not just coasting on easy summer days.

### Episode Detection (Future Metric)
* **What it means:** Did the model successfully sound the alarm when pollution crossed the dangerous 250 µg/m³ threshold? 
* **How to interpret:** This is a Yes/No accuracy score for predicting extreme events.

### Calibration and Confidence (Future Metric)
* **What it means:** Our advanced model doesn't just give a single number; it gives a range (e.g., "PM2.5 will be between 200 and 300"). Calibration checks if the model's confidence is actually justified. If it says it is 90% sure, is it actually right 9 times out of 10?

---

## Summary
To sum it up: We train models on the past and test them on an unseen future year (2025). We compare them against simple, common-sense guessing strategies (Baselines). We judge them primarily by how few units they are off by (MAE/RMSE), and we pay special attention to whether they can accurately predict the deadliest, most polluted days (AQI Stratification).

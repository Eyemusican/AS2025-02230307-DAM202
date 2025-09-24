# Simple RNN Weather Prediction Exercises

This project contains 5 exercises that teach you how Simple RNN (Recurrent Neural Networks) work for weather prediction. Each exercise focuses on a different aspect of RNN training and evaluation.

## Overview

The exercises use synthetic weather data to predict temperature and other weather variables. You'll learn about sequence length, model complexity, feature selection, different prediction targets, and error analysis.

## Exercise 1: Sequence Length Experiment

**What it does:** Tests how many days of past weather data the model should look at to make predictions.

**Steps:**
- Generate 500 days of synthetic weather data
- Test sequence lengths: 3, 5, 7, and 10 days
- Train separate models for each sequence length
- Compare performance using MAE (Mean Absolute Error) and RMSE
- Create visualizations showing which sequence length works best

**Key Findings:**
- Different sequence lengths give different prediction accuracy
- Too short (3 days) may miss important patterns
- Too long (10 days) can make training harder
- Usually 7 days works well for weather prediction

## Exercise 2: Hidden Units Experiment

**What it does:** Tests how complex the neural network should be by changing the number of hidden units.

**Steps:**
- Use the best sequence length from Exercise 1 (usually 7 days)
- Test different hidden unit sizes: 16, 32, 64, and 128
- Train models and measure both performance and overfitting
- Compare training time and model complexity
- Identify the best balance between performance and simplicity

**Key Findings:**
- More hidden units = more complex model
- Too few units (16) may underfit (too simple)
- Too many units (128) may overfit (memorizes training data)
- Usually 32-64 hidden units work well

## Exercise 3: Feature Selection Experiment

**What it does:** Tests which weather measurements are most important for making predictions.

**Steps:**
- Test different combinations of weather features:
  - Single features (only temperature, only humidity, etc.)
  - Pairs of features (temperature + humidity, etc.)
  - All features together
- Train models using optimal settings from Exercises 1 & 2
- Rank feature combinations by prediction accuracy
- Identify which weather variables are most useful

**Key Findings:**
- Not all weather measurements are equally useful
- Temperature history is usually the most important feature
- Adding more features doesn't always improve predictions
- Some feature combinations work better together

## Exercise 4: Predict Different Variables

**What it does:** Instead of just predicting temperature, try predicting humidity, pressure, or wind speed.

**Steps:**
- Use the same model architecture from previous exercises
- Train separate models to predict:
  - Temperature (baseline)
  - Humidity
  - Air pressure
  - Wind speed
- Compare how well the model predicts each weather variable
- Analyze which variables are easier or harder to predict

**Key Findings:**
- Some weather variables are easier to predict than others
- Variables with strong correlations are usually easier to predict
- The same model architecture can predict different things
- Wind speed is often the hardest to predict (it's more random)

## Exercise 5: Error Analysis

**What it does:** Analyzes when and why the model makes mistakes to understand its limitations.

**Steps:**
- Train a model using best settings from previous exercises
- Analyze prediction errors by:
  - Season (spring, summer, fall, winter)
  - Weather conditions (hot, cold, windy, etc.)
  - Time patterns (consecutive error days)
- Identify the worst prediction cases
- Find systematic problems with the model

**Key Findings:**
- Models struggle more in certain seasons
- Extreme weather conditions are harder to predict
- Large errors often happen during weather transitions
- Simple RNNs have limitations that advanced models can address

## How to Run the Code

### Requirements
- Python 3.7+
- Libraries: numpy, pandas, matplotlib, scikit-learn, tensorflow

### Installation
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Code Structure

Each exercise follows this pattern:
1. **Setup**: Import libraries and set random seeds
2. **Data Generation**: Create synthetic weather data
3. **Data Preparation**: Format data for neural network training
4. **Model Creation**: Build Simple RNN architecture
5. **Training**: Train models with different parameters
6. **Evaluation**: Calculate performance metrics
7. **Visualization**: Create plots showing results
8. **Analysis**: Explain findings and insights

## Key Concepts Learned

- **Sequence Length**: How much historical data to use
- **Model Complexity**: Balancing performance vs overfitting
- **Feature Engineering**: Choosing the right input variables
- **Problem Variation**: Adapting models to different prediction tasks
- **Error Analysis**: Understanding model limitations and failures

## Files Overview

- `weather_data_generation()`: Creates realistic synthetic weather data
- `prepare_data_for_simple_rnn()`: Formats data for RNN training
- `create_simple_rnn_model()`: Builds the neural network architecture
- `train_and_evaluate_model()`: Trains models and calculates performance
- Various analysis and visualization functions for each exercise



From this exercise, I learned:

- How RNNs can be used for time series prediction

- How to adjust hyperparameters in a step-by-step way

- How to check and compare model performance

- When Simple RNNs are useful and when they are not

## Summary

These five exercises gave me a full introduction to Simple RNNs for weather prediction. I learned how to build models step by step, tune hyperparameters, choose features, and analyze errors. Using synthetic weather data made the results easy to repeat and understand.
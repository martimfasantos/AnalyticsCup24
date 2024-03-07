# Analytics Cup 23/24

This repository contains work developed for the Analytics Cup 23/24 as part of the Business Analytics and Machine Learning master's course @ TUM. 
Within this year's Analytics Cup challenge, the focus is on LLMeals, a startup dedicated to suggesting personalized recipes to their customers. This repository contains the predictive model developed for LLMeals. The model aims to predict whether a customer will like or dislike a suggested recipe based on their preferences and dietary requirements.

## Team *tugas* (1/4 members)
| Authors        | GitHub                                      |
|----------------|---------------------------------------------|
| Martim Santos  | https://github.com/martimfasantos    |

**Ranking:** 38 / 160
**Project Grade:** 1.3 (German grading scale)

---

## Project Overview

LLMeals aims to address the quick turnover rate of customers observed after the test period. Despite the high-quality suggestions provided by their language models, customers often found the suggested recipes did not align with their specific preferences and dietary needs. To tackle this issue, the team decided to develop a predictive model to classify recipes based on users' potential preferences.

### The Challenge
The goal is to predict whether a customer likes or dislikes a recipe based on various factors including cooking time, macronutrients, and dietary preferences. The provided datasets include:

- **recipes.csv**: Contains details about 75,604 recipes including cooking time, macronutrient content, and other nutritional information.
- **reviews.csv**: Provides feedback from users including ratings and whether they liked the recipe.
- **diet.csv**: User profiles containing dietary preferences.
- **requests.csv**: User search behavior and preferences.


## Data Description

You can see the full data description [here](https://github.com/martimfasantos/AnalyticsCup24/blob/main/data_description.pdf).


## Evaluation
Predictions will be evaluated based on the balanced accuracy metric, which is the arithmetic mean of Sensitivity and Specificity achieved on the private test set.

- **Sensitivity (True Positive Rate)**: TP / (TP + FN)
- **Specificity (True Negative Rate)**: TN / (FP + TN)
- **Balanced Accuracy**: (Sensitivity + Specificity) / 2

**Final Balanced Accuracy:** 84.2 / 100

---

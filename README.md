## Project Title

**Predicting Student Placement: CGPA and Resume Score**

## Overview

This project aims to create a machine learning model that can predict whether a student will be placed or not based on their CGPA and resume score.

## Table of Contents

- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
## Project Description

Predicting student placement is a crucial step for universities and recruiters in understanding and enhancing the employability of students. This machine learning model uses CGPA and resume scores as input features to predict the likelihood of a student being placed.

### Features

- **Input Features**: CGPA, Resume Score
- **Output**: Placement status (Placed/Not Placed)

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

## Dataset

The dataset contains information on students' CGPA and resume scores along with their placement status. It is sourced from university placement data and preprocessed for the model's use.

### Dataset Structure

- **CGPA**: Cumulative Grade Point Average of the student
- **Resume Score**: A score calculated based on resume attributes such as skills, projects, etc.
- **Placement Status**: Binary target variable (Placed/Not Placed)

These models are trained to predict the placement status based on CGPA and resume score.

## Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/StudentPlacementPrediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd StudentPlacementPrediction
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data with CGPA and resume score.
2. Load the trained model:
   ```python
   import pickle
    import numpy as np
    pipe = pickle.load(open('pipe.pkl','rb'))
   ```
3. Use the model to predict placement:
   ```python
   input_data = [[cgpa, resume_score]]  # Replace with your data
   prediction = model.predict(input_data)
   print(prediction)
   ```


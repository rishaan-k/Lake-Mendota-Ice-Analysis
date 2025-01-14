# HW5: Frozen Days Analysis and Linear Regression

This repository contains a Python-based solution for analyzing the relationship between the number of frozen days in a year and time, using linear regression. The project explores multiple aspects of linear regression, including normal equation and gradient descent, as well as visualizing the results.

## Features
- **Data Analysis**: Read a CSV file containing yearly data of frozen days and analyze it using linear regression.
- **Visualization**: Generate plots to visualize the number of frozen days over the years and loss during gradient descent.
- **Linear Regression**: Implement linear regression with both normal equation and gradient descent.
- **Prediction**: Predict the number of frozen days for a specific year (2023) and solve for when the frozen days could be zero.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/frozen-days-analysis.git
    cd frozen-days-analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

   The required libraries include:
   - `numpy`
   - `pandas`
   - `matplotlib`

3. Make sure you have a CSV file containing the following columns:
   - `year`: Year of observation.
   - `days`: Number of frozen days in that year.

   Example CSV format:
   ```csv
   year,days
   1990,45
   1991,48
   ...
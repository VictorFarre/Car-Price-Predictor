import pandas as pd
import matplotlib as plt
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import requests
from flask import Flask, Response, request


cars = pd.read_excel("/Users/victorfarre/Documents/Seat_Code_1.xlsx", sheet_name= "Cars")


cars = cars.drop(columns=['last_price', 'CYLINDER_VOL', 'SEATS', '_source/provider_name', 'TITLE' ])
cars = cars.dropna(subset=['YEAR', 'MILEAGE'])
cars = cars.drop(cars['last_financed_price'].idxmax(), inplace=False)
cars['YEAR'] = pd.to_numeric(cars['YEAR'], errors='coerce').astype(int)
cars['MILEAGE'] = pd.to_numeric(cars['MILEAGE'], errors='coerce').astype(int)

categorical_cols = [ 'STATUS', 'COLOR', 'ENV_LABEL', 'FUEL_TYPE']
df_encoded = pd.get_dummies(cars, columns=categorical_cols)

train_size = 0.8  # Adjust for desired split (e.g., 0.8 for 80%)
train_data, test_data = train_test_split(df_encoded, test_size=1 - train_size, random_state=12)

# Get all unique models in the entire data
all_models = df_encoded['MODEL'].unique()

# Check for models in test but not in train
missing_models_in_train = test_data['MODEL'].isin(all_models).notna().sum() - test_data['MODEL'].nunique()

if missing_models_in_train > 0:
  # Identify rows with missing models in train
  missing_in_train = test_data[~test_data['MODEL'].isin(train_data['MODEL'])]

  # Update train and test data (add missing models to train, remove from test)
  train_data = pd.concat([train_data, missing_in_train], ignore_index=True)
  test_data = test_data[test_data['MODEL'].isin(train_data['MODEL'])]

  print(f"Added {missing_models_in_train} models from test to training data.")

else:
  print("All models in test data are present in training data.")

formula = 'last_financed_price ~ YEAR + HP + C(MODEL) + C(TRANSMISSION) + YEAR:MILEAGE'  # Add other variables as needed
model = sm.MixedLM.from_formula(formula, data=train_data, groups=train_data['MANUFACTURER'])
result = model.fit()

predicted_prices = result.predict(train_data)
r2 = r2_score(train_data['last_financed_price'], predicted_prices)


predicted_prices = result.predict(test_data)
r2 = r2_score(test_data['last_financed_price'], predicted_prices)

test_data['Predicted_Price'] = predicted_prices  # Add predicted prices to DataFrame
results_test = test_data[['MANUFACTURER', 'MODEL', 'HP', 'YEAR', 'MILEAGE', 'last_financed_price', 'Predicted_Price']]

formula = 'last_financed_price ~ YEAR + HP + C(MODEL) + C(TRANSMISSION) + YEAR:MILEAGE'

def predict_car_price():
  """Prompts user for car data and predicts the price using a trained model.

  This function assumes you have a trained model stored in a variable called 'model'
  and that the model expects a DataFrame with specific column names.
  """

  # Get user input for car data
  manufacturer = input("Enter Manufacturer: ")
  model = input("Enter Model: ")
  hp = float(input("Enter Horsepower (HP): "))
  transmission = input("Enter Transmission (e.g., Automatic, Manual): ")
  year = int(input("Enter Year: "))
  mileage = int(input("Enter Mileage: "))

  # Create a DataFrame from user input
  data = {
      'Manufacturer': [manufacturer],
      'MODEL': [model],
      'HP': [hp],
      'TRANSMISSION': [transmission],
      'YEAR': [year],
      'MILEAGE': [mileage]
  }
  df = pd.DataFrame(data)

  predicted_price = result.predict(data)



  # Print the predicted price
  print(f"Predicted price for a {predicted_price}")

  print(f"Predicted price for a {year} {manufacturer} {model} with {hp} HP and {mileage} miles: ${predicted_price.values[0]:.2f}")


def predict_car_price_test():
    manufacturer = request.form['manufacturer']
    model = request.form['model']
    hp = float(request.form['hp'])
    transmission = request.form['transmission']
    year = int(request.form['year'])
    mileage = int(request.form['mileage'])

    # Call your prediction logic using these variables
    predicted_price = result.predict([[manufacturer, model, hp, transmission, year, mileage]])  # Assuming result expects a 2D array
    print(f"Price: ${predicted_price.values[0]:.2f}")


#predict_car_price()

def predict_car_price_off(manufacturer, model, hp, transmission, year, mileage):
    # Create a DataFrame with the input variables
    input_data = pd.DataFrame({
        'MANUFACTURER': [manufacturer],
        'MODEL': [model],
        'HP': [hp],
        'TRANSMISSION': [transmission],
        'YEAR': [year],
        'MILEAGE': [mileage]
    })

    # Predict from the model using the input DataFrame
    predicted_price = result.predict(input_data)  # Assuming 'result' is your model
    print(predicted_price)
    return predicted_price.values[0]

predict_car_price_off('BMW','X2', 250, 'manual', 2000, 12000)
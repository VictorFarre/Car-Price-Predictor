from flask import Flask, render_template, request
from car_prediction import predict_car_price_off


app = Flask(__name__)

# Define routes and logic for your application here
@app.route('/')
def home():
    return "Welcome to your car price prediction website Now!"

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/predict_price', methods=['GET', 'POST'])
def predict_car_price():
    if request.method == 'GET':
        return render_template('predict_price_form.html')
    else:  # Assuming POST request
        return handle_prediction_form()
    
def handle_prediction_form():
    print("Request received!")
    error_message = None
    
    try:
        print("Form data:", request.form)
        # Extract user input from the form (adjust field names based on your HTML form)
        year = int(request.form['year'])
        manufacturer = request.form['manufacturer']
        model = request.form['model']
        hp = float(request.form['hp'])
        mileage = int(request.form['mileage'])
        transmission = request.form['transmission']

        if model.isdigit():
            model = int(model)  # Convert to integer if it contains only numeric characters
        else:
            model = str(model)

        predicted_price = predict_car_price_off(manufacturer, model, hp, transmission, year, mileage)

        # Add print statements here
        print(f"year: {year}")
        print(f"manufacturer: {manufacturer}")
        print(f"model: {model}")
        print(f"hp: {hp}")
        print(f"mileage: {mileage}")
        print(f"transmission: {transmission}")
        print(f"Price: {predicted_price}")

        input_values = {
            'year': year,
            'manufacturer': manufacturer,
            'model': model,
            'hp': hp,
            'mileage': mileage,
            'transmission': transmission
        }

    

    except Exception as e:
        # Handle prediction errors gracefully
        input_values = None
        predicted_price = None
        error_message = str(e)  # Capture error message

    # Render the result template with predicted price (or error message)
    return render_template('predict_price_result.html', input_values=input_values, predicted_price=predicted_price, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict_car_price_new():
    if request.method == 'GET':
        return render_template('Predict.html')
    else:  # Assuming POST request
        return handle_prediction_form()
    
def handle_prediction_form():
    print("Request received!")
    error_message = None
    
    try:
        print("Form data:", request.form)
        # Extract user input from the form (adjust field names based on your HTML form)
        year = int(request.form['year'])
        manufacturer = request.form['manufacturer']
        model = request.form['model']
        hp = float(request.form['hp'])
        mileage = int(request.form['mileage'])
        transmission = request.form['transmission']

        if model.isdigit():
            model = int(model)  # Convert to integer if it contains only numeric characters
        else:
            model = str(model)

        predicted_price = predict_car_price_off(manufacturer, model, hp, transmission, year, mileage)

        # Add print statements here
        print(f"year: {year}")
        print(f"manufacturer: {manufacturer}")
        print(f"model: {model}")
        print(f"hp: {hp}")
        print(f"mileage: {mileage}")
        print(f"transmission: {transmission}")
        print(f"Price: {predicted_price}")

        input_values = {
            'year': year,
            'manufacturer': manufacturer,
            'model': model,
            'hp': hp,
            'mileage': mileage,
            'transmission': transmission
        }

    

    except Exception as e:
        # Handle prediction errors gracefully
        input_values = None
        predicted_price = None
        error_message = str(e)  # Capture error message

    # Render the result template with predicted price (or error message)
    return render_template('Predict_Result.html', input_values=input_values, predicted_price=predicted_price, error_message=error_message)
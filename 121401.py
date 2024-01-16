import joblib

# Load the model
model = joblib.load("2023121401.pkl")

# Collect input from the user
gender = input("Enter gender (0 for Male, 1 for Female): ")
married = input("Are you married? (0 for No, 1 for Yes): ")
dependents = input("Number of dependents: ")
education = input("Education level (0 for Not Graduate, 1 for Graduate): ")
self_employed = input("Are you self-employed? (0 for No, 1 for Yes): ")
property_area = input("Enter property area (0 for Urban, 1 for Semiurban, 2 for Rural): ")

# Convert input to numerical format
input_data = [int(gender), int(married), int(dependents), int(education), int(self_employed), int(property_area)]

# Make predictions
pred = model.predict([input_data])
print(pred)
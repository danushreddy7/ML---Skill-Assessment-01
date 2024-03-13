# ML---Skill-Assessment-01
## Developed By : Abbu Rehan
## Register Number : 212223240165
## Department : AIML
## Objective 1 : 
To Create a scatter plot between cylinder vs Co2Emission (green color).
## Code : 
```
Developed by : SD.ABBU REHAN
Register Number : 212223240165

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()
```
## Output :
![ml ws 1 1](https://github.com/Abburehan/ML---Skill-Assessment-01/assets/138849336/f1789226-9288-489f-b2c3-9ca7d748b2af)
## Objective 2 : 
Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors.
## Code :
```
Developed by : SD.ABBU REHAN
Register Number : 212223240165

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='red', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='yellow', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()
```
## Output :
![ml ws 1 2](https://github.com/Abburehan/ML---Skill-Assessment-01/assets/138849336/58f81faf-8bba-42aa-8fbc-6ed8ebbcb74a)
## Objective 3 :
Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors.
## Code :
```
Developed by : SD.ABBU REHAN
Register Number : 212223240165

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('FuelConsumption.csv')

plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='brown', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='green', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()
```
## Output :
![image](https://github.com/Abburehan/ML---Skill-Assessment-01/assets/138849336/4c4297d3-4723-40e7-8149-c92da08413e4)
## Objective 4 :
Train your model with independent variable as cylinder and dependent variable as Co2Emission.
## Code :
```
Developed by : SD.ABBU REHAN
Register Number : 212223240165

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FuelConsumption.csv')

X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']

X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)

model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)
```
## Output :
![image](https://github.com/Abburehan/ML---Skill-Assessment-01/assets/138849336/2e41fdd4-5d8d-4f56-8205-4f4d3d3cdc78)
## Objective 5 :
Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission.
## Code :
```
Developed by : SD.ABBU REHAN
Register Number : 212223240165

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('FuelConsumption.csv')

X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']

X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)

model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)
```
## Output :
![image](https://github.com/Abburehan/ML---Skill-Assessment-01/assets/138849336/a84c5e9a-41f6-4479-b243-3ec9e5250739)
## Objective 6 :
Train your model on different train test ratio and train the models and note down their accuracies.
## Code :
```
Developed by : SD.ABBU REHAN
Register Number : 212223240165

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
ratios = [0.1, 0.4, 0.5, 0.8]

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')
```
## Output :
![image](https://github.com/Abburehan/ML---Skill-Assessment-01/assets/138849336/3bec1d51-ca13-4c08-b690-18854c09fe5d)
## Result :
All the programs executed successfully.

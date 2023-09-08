import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#Create a coloumn transformer
ct = make_column_transformer(
    (StandardScaler(),["age","bmi","children"]),#Turn the values between -3 and 3
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
    )
#Create X and Y
X = insurance.drop("charges",axis =1)
y = insurance["charges"]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Fit column transformer to out training data
ct.fit(X_train)

#Transform training and test data with normalization (Standard)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

X_train_normal

#EarlyStoppingCallback
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
tf.random.set_seed(42)
#1. Creating the model
insurance_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])

#Compile the model
insurance_model_3.compile(loss = tf.keras.losses.mae,
                          optimizer = tf.keras.optimizers.Adam(lr = 0.01),
                          metrics =["mae"])
history = insurance_model_3.fit(X_train_normal,y_train,epochs=2000,callbacks=[callback],verbose=1)

#Evaluate model 3
insurance_model_3.evaluate(X_test_normal,y_test)

#Plot history (loss curve)
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

# Predicting on the test data
y_pred = insurance_model_3.predict(X_test_normal)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot actual values
ax1.scatter(range(len(y_test)), y_test, c='blue', label='Actual Charges')
ax1.set_xlabel('Data Point')
ax1.set_ylabel('Actual Charges')
ax1.set_title('Actual Insurance Charges')
ax1.grid(True)

# Plot predicted values
ax2.scatter(range(len(y_pred)), y_pred, c='green', label='Predicted Charges')
ax2.set_xlabel('Data Point')
ax2.set_ylabel('Predicted Charges')
ax2.set_title('Predicted Insurance Charges')
ax2.grid(True)

plt.tight_layout()
plt.show()


import pandas as pd

# Create a DataFrame to hold actual and predicted charges
results_df = pd.DataFrame({'Actual Charges': y_test, 'Predicted Charges': y_pred.flatten()})

# Display the DataFrame
print(results_df)




# Create a DataFrame for actual and predicted charges
results_df = pd.DataFrame({'Actual Charges': y_test, 'Predicted Charges': y_pred.flatten()})

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Actual Charges', y='Predicted Charges', data=results_df, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs. Predicted Insurance Charges with Regression Line')
plt.grid(True)
plt.show()

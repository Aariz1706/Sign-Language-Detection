import pandas as pd # Helps work with csv files 
from sklearn.model_selection import train_test_split # spliting data one for training and one for testing
from sklearn.neighbors import KNeighborsClassifier #importing KNN algorithm where it recognizes which letter your hand is showing by comparing it to other examples
from sklearn.metrics import accuracy_score #How many correct guesses it makes
import joblib #saving trained model in a file
data = pd.read_csv("hand_data.csv")
# Split into features (X) and labels (y)
X = data.iloc[:, 1:]  # All columns except the first one
y = data.iloc[:, 0]   # First column is the label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate and print accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "sign_knn_model.pkl")
# X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42) # Splits your data: 80% is used to train the model (X_train, y_train) and 20% is used to test how well it works (X_test, y_test). same split - random_state=42
# knn = KNeighborsClassifier(n_neighbors = 3) #Creates a knn model. Looks at 3 closest examples to guess which letter your hand is showing.
# knn.fit(X_train , y_train) # Trains the model, it learns what hand landmarks match what letters.
# accuracy = knn.score(X_test , y_test) #Tells us how many correct guesses the model makes when looking at the test data
# print(F"Model Accuracy: {accuracy *100:.2f}%") #Shows the accuracy as a percentage
# joblib.dump(knn , "sign_knn_model.pkl") #Saves trained model to a file so we can use it later for real-time prediction 


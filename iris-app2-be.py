import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].apply(lambda x: iris.target_names[x])
df['target'] = iris.target


print("Complete Iris Dataset:")
print(df)


g = sns.pairplot(df, hue='species_name')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Pairplot of Iris Features", y=0.99)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.iloc[:, :4].values  
y = df['species'].values    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

for name, model in models.items():
    print(f"\nTraining and evaluating model: {name}")
    model.fit(X_train, y_train)                    
    y_pred = model.predict(X_test)                  
    acc = accuracy_score(y_test, y_pred)           
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
example_sample = [[5.1, 3.5, 1.4, 0.2]]  
predicted_class = models['Random Forest'].predict(example_sample)
print(f"\nPrediction for sample {example_sample}: {iris.target_names[predicted_class][0]}")
print("\nExamples of predictions for all 3 species:")
test_samples = [
    [5.1, 3.5, 1.4, 0.2],   
    [6.0, 2.9, 4.5, 1.5],   
    [6.5, 3.0, 5.8, 2.2]    
]

for sample in test_samples:
    pred = models['Random Forest'].predict([sample])[0]
    print(f"{sample} → {iris.target_names[pred]}")










 




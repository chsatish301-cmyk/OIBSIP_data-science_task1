import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Create output folder
os.makedirs("output", exist_ok=True)


# Load dataset
df = pd.read_csv("Iris.csv")

print(df.head())


# Save dataset information
with open("output/dataset_info.txt", "w") as f:
    f.write(str(df.info()))
    f.write("\n\n")
    f.write(str(df.describe()))
    f.write("\n\nMissing Values:\n")
    f.write(str(df.isnull().sum()))


# Remove Id column

df.drop("Id", axis=1, inplace=True)



# Species count graph

sns.countplot(x="Species", data=df)

plt.title("Species Distribution")

plt.savefig("output/species_count.png")

plt.show()



# Pair plot

sns.pairplot(df, hue="Species")

plt.savefig("output/pairplot.png")

plt.show()



# Correlation heatmap

plt.figure(figsize=(6,5))

sns.heatmap(
    df.iloc[:,0:4].corr(),
    annot=True
)

plt.title("Feature Correlation")

plt.savefig("output/correlation_heatmap.png")

plt.show()



# Split data

X = df.drop("Species", axis=1)

y = df["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train model

model = RandomForestClassifier()

model.fit(X_train, y_train)



# Prediction

prediction = model.predict(X_test)



# Accuracy

accuracy = accuracy_score(
    y_test,
    prediction
)


print("Accuracy:", accuracy)


with open("output/accuracy.txt","w") as f:
    f.write("Accuracy:\n")
    f.write(str(accuracy))



# Classification report

report = classification_report(
    y_test,
    prediction
)


print(report)


with open("output/classification_report.txt","w") as f:
    f.write(report)



# Confusion matrix

cm = confusion_matrix(
    y_test,
    prediction
)


sns.heatmap(
    cm,
    annot=True
)


plt.title("Confusion Matrix")

plt.savefig(
    "output/confusion_matrix.png"
)

plt.show()



# New flower prediction


new_flower = pd.DataFrame(
    [[5.1,3.5,1.4,0.2]],
    columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)


result = model.predict(new_flower)


print(
    "Predicted Species:",
    result
)



with open("output/prediction_output.txt","w") as f:

    f.write(
        "Input:\n"
    )

    f.write(
        str(new_flower)
    )

    f.write(
        "\n\nPredicted Species:\n"
    )

    f.write(
        str(result)
    )
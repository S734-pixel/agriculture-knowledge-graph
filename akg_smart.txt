# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:/Users/sidds/Downloads/Crop_recommendationV2.csv.xlsx'
dataset = pd.read_excel(file_path)
numeric_dataset = dataset.select_dtypes(include=['int64', 'float64'])
# Data Exploration
print("Dataset Info:")
print(dataset.info())

print("\nFirst few rows of the dataset:")
print(dataset.head())

# Descriptive statistics
descriptive_stats = dataset.describe()
print("\nDescriptive Statistics:")
print(descriptive_stats)

# Correlation matrix
correlation_matrix = numeric_dataset.corr()

# Visualizing Histograms for key features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histogram for Temperature, Humidity, pH, Rainfall, Soil Moisture, and Crop Density
sns.histplot(dataset['temperature'], ax=axes[0, 0], kde=True).set(title='Temperature Distribution')
sns.histplot(dataset['humidity'], ax=axes[0, 1], kde=True).set(title='Humidity Distribution')
sns.histplot(dataset['ph'], ax=axes[0, 2], kde=True).set(title='pH Distribution')
sns.histplot(dataset['rainfall'], ax=axes[1, 0], kde=True).set(title='Rainfall Distribution')
sns.histplot(dataset['soil_moisture'], ax=axes[1, 1], kde=True).set(title='Soil Moisture Distribution')
sns.histplot(dataset['crop_density'], ax=axes[1, 2], kde=True).set(title='Crop Density Distribution')

plt.tight_layout()
plt.show()

# Heatmap for Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
plt.show()

# Crop-wise analysis: Calculate mean for each crop type
crop_analysis = dataset.groupby('label').mean()

# Display crop-wise summary (you can also use print to output in the console)
#import ace_tools as tools; tools.display_dataframe_to_user(name="Crop-wise Analysis", dataframe=crop_analysis)

# Boxplot for visualizing the distribution of key features by crop type
plt.figure(figsize=(12, 6))
sns.boxplot(x='label', y='temperature', data=dataset)
plt.title('Temperature Distribution by Crop Type')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='label', y='rainfall', data=dataset)
plt.title('Rainfall Distribution by Crop Type')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='label', y='soil_moisture', data=dataset)
plt.title('Soil Moisture Distribution by Crop Type')
plt.xticks(rotation=90)
plt.show()

# Further Analysis could include:
# - Identifying outliers or trends for certain crops
# - Predictive modeling, e.g., classification to predict crop type based on features

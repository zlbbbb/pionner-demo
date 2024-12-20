import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset (you can replace this with your own data file)
data = {
    'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Sales': [100, 200, 150, 300, 250, 180, 200, 350, 220],
    'Price': [10, 15, 12, 10, 15, 12, 11, 15, 13],
    'Date': ['2024-01-01', '2024-01-01', '2024-01-01',
             '2024-01-02', '2024-01-02', '2024-01-02',
             '2024-01-03', '2024-01-03', '2024-01-03']
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Basic Data Inspection
print("=== Data Overview ===")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

# 2. Data Cleaning
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Data Analysis
# Group by Product and calculate statistics
product_analysis = df.groupby('Product').agg({
    'Sales': ['mean', 'sum'],
    'Price': 'mean'
}).round(2)

print("\nProduct Analysis:")
print(product_analysis)

# Calculate daily sales
daily_sales = df.groupby('Date')['Sales'].sum()
print("\nDaily Sales:")
print(daily_sales)

# 4. Data Visualization
plt.figure(figsize=(12, 5))

# Create subplots
plt.subplot(1, 2, 1)
sns.boxplot(x='Product', y='Sales', data=df)
plt.title('Sales Distribution by Product')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Price', y='Sales', hue='Product', data=df)
plt.title('Price vs Sales by Product')

plt.tight_layout()
plt.show()

# 5. Additional Analysis: Sales Trends
print("\nSales Trends:")
sales_trends = df.pivot_table(
    values='Sales',
    index='Date',
    columns='Product',
    aggfunc='sum'
)
print(sales_trends)
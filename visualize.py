import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv('books_data_with_ratings.csv')  # Replace with your actual CSV file name

# Convert 'Price' column to numeric (remove the £ symbol)
df['Price'] = df['Price'].replace('£', '', regex=True).astype(float)

# Define price ranges
price_ranges = ['<=20', '>20 & <=40', '>40 & <=60', '>60']
counts = [0, 0, 0, 0]

# Categorize prices into ranges
for price in df['Price']:
    if price <= 20:
        counts[0] += 1
    elif 20 < price <= 40:
        counts[1] += 1
    elif 40 < price <= 60:
        counts[2] += 1
    else:
        counts[3] += 1

# Bar Graph
plt.figure(figsize=(8, 6))
plt.bar(price_ranges, counts, color=['skyblue', 'lightgreen', 'orange', 'pink'], edgecolor='black')
plt.title('Books Distribution by Price Range', fontsize=16)
plt.xlabel('Price Range (£)', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=price_ranges, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'orange', 'pink'])
plt.title('Books Price Range Distribution', fontsize=16)
plt.tight_layout()
plt.show()

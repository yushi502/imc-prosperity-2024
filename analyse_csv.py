import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file into a DataFrame
df = pd.read_csv("R1.csv", delimiter=";")

# Display the first few rows of the DataFrame
print(df.head())

# Display basic information about the DataFrame
print(df.info())

# Summary statistics of numerical columns
print(df.describe())

# Plot bid prices over time for each product
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

for product, ax in zip(df['product'].unique(), axes):
    product_df = df[df['product'] == product]
    ax.plot(product_df['timestamp'], product_df['bid_price_1'], label='Bid Price 1')
    ax.set_title(f'Bid Prices Over Time - {product}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Bid Price')
    ax.legend()

plt.figure(figsize=(10, 6))
plt.plot(df[df['product'] == 'AMETHYSTS']['timestamp'], df[df['product'] == 'AMETHYSTS']['bid_price_1'], label='Bid Price 1', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Bid Price')
plt.title('Bid Prices Over Time for AMETHYSTS')
plt.grid(True)

# Adjust y-axis limits to zoom out the signal amplitude
plt.ylim(0, 12000)  # Adjust the upper limit as needed to zoom out

plt.legend()
plt.show()
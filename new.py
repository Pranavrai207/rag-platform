import pandas as pd
import random
from datetime import datetime, timedelta

# Configuration
num_entries = 100
categories = {
    "Electronics": ["Voltex Headphones", "SwiftTab Tablet", "Lumina Smart Bulb", "HyperLink Router"],
    "Office Supplies": ["Ergonomic Chair", "Bamboo Desk Organizer", "Mechanical Keyboard"],
    "Apparel": ["WeatherGuard Jacket", "Breathable Mesh Sneakers", "Cotton Slim-Fit Tee"],
    "Fitness": ["IronGrip Dumbbells", "Yoga Mat Pro", "SpeedJump Rope"]
}
regions = ["North", "South", "East", "West"]
payments = ["Credit Card", "UPI", "PayPal", "Debit Card"]

data = []

for i in range(num_entries):
    category = random.choice(list(categories.keys()))
    product = random.choice(categories[category])
    date = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 75))
    
    data.append({
        "Transaction_ID": f"TSX-{1000 + i}",
        "Date": date.strftime("%Y-%m-%d"),
        "Customer_Name": random.choice(["Amit Singh", "John Doe", "Sara Khan", "Emily Blunt", "Vikram Rao"]),
        "Product_Category": category,
        "Product_Name": product,
        "Quantity": random.randint(1, 5),
        "Unit_Price": random.randint(20, 500),
        "Region": random.choice(regions),
        "Payment_Method": random.choice(payments)
    })

df = pd.DataFrame(data)
df.to_csv("sales_test_data.csv", index=False)
print("File 'sales_test_data.csv' created with 100 entries.")

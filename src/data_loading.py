import pandas as pd

def load_data():
    # read the CSV file into a table
    data = pd.read_csv("data/raw/creditcard.csv")
    return data

if __name__ == "__main__":
    data = load_data()

    print("Rows and columns:", data.shape)
    print("Number of fraud cases:", data["Class"].sum())
    print("Fraud rate (%):", round(data["Class"].mean() * 100, 2))
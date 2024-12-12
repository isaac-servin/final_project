import sqlite3
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import os
from datetime import datetime
import time

def fetch_eia_data(url, table_name, num_cols):
    conn = sqlite3.connect('project_database.db')

    cur = conn.cursor()
    # Create a table to track the last processed index for different tables
    cur.execute('''
    CREATE TABLE IF NOT EXISTS LastProcessed (
        table_name TEXT PRIMARY KEY,
        last_index INTEGER DEFAULT 0
    )''')
    conn.commit()

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    cells = soup.find_all('td', class_='r t sel-2 sel-13 data sel-22')
    
    # Retrieve the last saved index from which to start processing data
    last_index = get_last_saved_index(table_name, conn)
    data = []

    # Process data in chunks based on the number of columns in the data set
    for i in range(last_index * num_cols, min(len(cells), (last_index + 25) * num_cols), num_cols):
        row_data = [cell.get_text(strip=True) for cell in cells[i:i + num_cols]]

        # Validate if the fetched row has the correct number of columns
        if len(row_data) == num_cols:
            data.append(row_data)
        else:
            print(f"Skipping row due to mismatched columns: {row_data}")

    # Update the last processed index 
    update_last_index(table_name, last_index + len(data), conn)
    conn.close()
    return data

# Retrieve the last saved index for a specific table 
def get_last_saved_index(table_name, conn):
    cur = conn.cursor()
    cur.execute("SELECT last_index FROM LastProcessed WHERE table_name = ?", (table_name,))
    result = cur.fetchone()
    if result is None:
        # Initialize last index if not present
        cur.execute("INSERT INTO LastProcessed (table_name, last_index) VALUES (?, ?)", (table_name, 0))
        conn.commit()
        return 0
    return result[0]

# Update the last index 
def update_last_index(table_name, last_index, conn):
    cur = conn.cursor()
    cur.execute("UPDATE LastProcessed SET last_index = ? WHERE table_name = ?", (last_index, table_name))
    conn.commit()
    
def store_data(data, table_name):
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()
    # Ensure tables exist
    if table_name == 'eia_data':
        cur.execute('''
        CREATE TABLE IF NOT EXISTS eia_data (
            id INTEGER PRIMARY KEY,
            Year INTEGER,
            Coal REAL,
            Petroleum REAL,
            Natural_Gas REAL,
            Other_Fossil_Gas REAL,
            Nuclear REAL,
            Hydroelectric_Conventional REAL,
            Other_Renewable_Sources REAL,
            Hydroelectric_Pumped_Storage REAL,
            Other_Energy_Sources REAL,
            Utility_Total REAL,
            Estimated_Photovoltaic REAL
        )
        ''')
        
        for row in data:
            cur.execute('''
            INSERT OR IGNORE INTO eia_data (Year, Coal, Petroleum, Natural_Gas, Other_Fossil_Gas, Nuclear, 
            Hydroelectric_Conventional, Other_Renewable_Sources, Hydroelectric_Pumped_Storage, 
            Other_Energy_Sources, Utility_Total, Estimated_Photovoltaic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)
    
    elif table_name == 'electricity_data':
        cur.execute('''
        CREATE TABLE IF NOT EXISTS electricity_data (
            id INTEGER PRIMARY KEY,
            Year INTEGER,
            Residential REAL,
            Commercial REAL,
            Industrial REAL,
            Transportation REAL,
            Total REAL,
            Direct_Use REAL,
            Total_End_Use REAL
        )
        ''')
        
        for row in data:
            cur.execute('''
            INSERT OR IGNORE INTO electricity_data (Year, Residential, Commercial, Industrial, Transportation, 
            Total, Direct_Use, Total_End_Use)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)
    
    conn.commit()
    conn.close()

def visualize_data():
    conn = sqlite3.connect('project_database.db')

    # Fetch only relevant columns
    eia_query = "SELECT Year, Petroleum FROM eia_data"
    electricity_query = "SELECT Year, Residential, Commercial, Industrial, Transportation FROM electricity_data"

    # Data into Pandas DataFrames
    eia_df = pd.read_sql_query(eia_query, conn)
    electricity_df = pd.read_sql_query(electricity_query, conn)

    conn.close()

    # Convert columns to numeric
    eia_df['Year'] = pd.to_numeric(eia_df['Year'], errors='coerce')
    eia_df['Petroleum'] = pd.to_numeric(eia_df['Petroleum'], errors='coerce')

    electricity_df['Year'] = pd.to_numeric(electricity_df['Year'], errors='coerce')
    numeric_columns = ['Residential', 'Commercial', 'Industrial', 'Transportation']
    electricity_df[numeric_columns] = electricity_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    eia_df = eia_df.dropna(subset=['Petroleum'])

    # Merge on Year
    merged_data = pd.merge(eia_df, electricity_df, on='Year', how='inner')

    # Plot Petroleum over the years
    plt.figure(figsize=(10, 6))
    plt.bar(merged_data['Year'], merged_data['Petroleum'], color='orange', alpha=0.8)
    plt.title('Petroleum Production Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Energy Production (Megawatts)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('petroleum_production_bar.png')
    plt.show()

visualize_data()
if __name__ == "__main__":
    urls = [
        'https://www.eia.gov/electricity/annual/html/epa_04_02_a.html',
        'https://www.eia.gov/electricity/annual/html/epa_02_02.html'
    ]
    table_names = ['eia_data', 'electricity_data']
    num_cols = [12, 8]
    
    for url, table_name, cols in zip(urls, table_names, num_cols):
        data = fetch_eia_data(url, table_name, cols)
        store_data(data, table_name)
    
    print("Data gathering and storage complete.")

def fetch_data_from_db():

    conn = sqlite3.connect('project_database.db')
    
    eia_query = "SELECT * FROM eia_data"
    electricity_query = "SELECT * FROM electricity_data"
    
    eia_df = pd.read_sql_query(eia_query, conn)
    electricity_df = pd.read_sql_query(electricity_query, conn)
    
    conn.close()

    # Convert numeric columns in electricity_df
    numeric_columns = ['Residential', 'Commercial', 'Industrial', 'Transportation']
    electricity_df[numeric_columns] = electricity_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate total consumption
    total_consumption = electricity_df[numeric_columns].sum()

    return eia_df, electricity_df, total_consumption

def calculate_and_visualize(eia_df, electricity_df):
    # Convert numeric columns to float, excluding 'id' and 'Year'
    numeric_columns = eia_df.columns.drop(['id', 'Year'])
    eia_df[numeric_columns] = eia_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Calculation: Average energy production by source
    avg_production = eia_df[numeric_columns].mean()

    # Visualization 1: Bar chart of average energy production by source
    plt.figure(figsize=(12, 6))
    avg_production.plot(kind='bar')
    plt.title('Average Energy Production by Source')
    plt.ylabel('Megawatts')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('avg_energy_production.png')
    plt.close()

    # Convert numeric columns in electricity_df
    numeric_columns = ['Residential', 'Commercial', 'Industrial', 'Transportation']
    electricity_df[numeric_columns] = electricity_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Calculation: Total electricity consumption by sector
    total_consumption = electricity_df[numeric_columns].sum().dropna()

    # Visualization 2: Pie chart of total electricity consumption by sector
    if not total_consumption.empty and total_consumption.sum() > 0:
        plt.figure(figsize=(10, 10))
        plt.pie(total_consumption, labels=total_consumption.index, autopct='%1.1f%%')
        plt.title('Total Electricity Consumption by Sector')
        plt.savefig('electricity_consumption_by_sector.png')
        plt.close()
    else:
        print("No valid data for pie chart visualization")

    # Write calculations to a file
    with open('calculations.txt', 'w') as f:
        f.write("Average Energy Production by Source:\n")
        f.write(str(avg_production))
        f.write("\n\nTotal Electricity Consumption by Sector:\n")
        f.write(str(total_consumption))
eia_df, electricity_df, total_consumption = fetch_data_from_db()


if __name__ == "__main__":
    # Fetched individual tables 
    eia_df, electricity_df, total_consumption = fetch_data_from_db()
    

    calculate_and_visualize(eia_df, electricity_df)
    print("Data processing and visualization complete.")

    # Use total_consumption directly for the pie chart
    if not total_consumption.empty and total_consumption.sum() > 0:
        plt.figure(figsize=(10, 10))
        plt.pie(total_consumption, labels=total_consumption.index, autopct='%1.1f%%')
        plt.title('Total Electricity Consumption by Sector')
        plt.savefig('electricity_consumption_by_sector.png')
        plt.close()
    else:
        print("No valid data for pie chart visualization")

# In your main code
eia_df, electricity_df, total_consumption = fetch_data_from_db()
calculate_and_visualize(eia_df, electricity_df)
print("Data processing and visualization complete.")















#MATTS DATA

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": 40.71,
    "longitude": 74,
    "start_date": "2021-01-01",
    "end_date": "2024-01-01",
    "daily": ["temperature_2m_max", "temperature_2m_min"],
    "timezone": "America/New_York"
}

response = requests.get(API_URL, params=PARAMS)

if response.status_code == 200:
    data = response.json()
    
    with open('weather_data.json', 'w') as f:
        json.dump(data, f, indent=4)
    
    print("Weather data has been saved to weather_data.json")
else:
    print(f"Error: {response.status_code}, {response.text}")




def load_weather_data(json_file, max_items=25):
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS weather_data (
        date TEXT PRIMARY KEY,
        temperature_2m_max REAL,
        temperature_2m_min REAL
    )''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS last_inserted_date (
        id INTEGER PRIMARY KEY,
        date TEXT
    )''')

    cur.execute("SELECT date FROM last_inserted_date WHERE id = 1")
    result = cur.fetchone()
    last_date = result[0] if result else None

    with open(json_file, 'r') as f:
        data = json.load(f)

    daily_data = data['daily']
    dates = daily_data['time']
    max_temps = daily_data['temperature_2m_max']
    min_temps = daily_data['temperature_2m_min']

    items_inserted = 0
    start_index = dates.index(last_date) + 1 if last_date in dates else 0

    for i in range(start_index, len(dates)):
        if items_inserted >= max_items:
            break
        cur.execute('''
        INSERT OR IGNORE INTO weather_data VALUES (?, ?, ?)
        ''', (dates[i], max_temps[i], min_temps[i]))
        items_inserted += 1
        last_date = dates[i]

    if items_inserted > 0:
        cur.execute('''
        INSERT OR REPLACE INTO last_inserted_date (id, date) VALUES (1, ?)
        ''', (last_date,))

    conn.commit()
    conn.close()

    print(f"Inserted {items_inserted} items into weather_data")

# Call the function to load data
load_weather_data('weather_data.json')


def calculate_and_write_results():
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()

    # Calculate average temperatures
    cur.execute("SELECT AVG(temperature_2m_max), AVG(temperature_2m_min) FROM weather_data")
    result = cur.fetchone()
    avg_max, avg_min = result if result else (None, None)

    # Calculate total residential electricity usage
    cur.execute("SELECT SUM(Residential) FROM electricity_data")
    total_residential = cur.fetchone()[0]

    # Join weather_data and electricity_data
    cur.execute("""
    SELECT w.date, w.temperature_2m_max, e.Residential
    FROM weather_data w
    JOIN electricity_data e ON substr(w.date, 1, 4) = e.Year
    LIMIT 10
    """)
    joined_data = cur.fetchall()

    conn.close()

    # Write results to a file
    with open('weather_calculation_results.txt', 'w') as f:
        if avg_max is not None and avg_min is not None:
            f.write(f"Average max temperature: {avg_max:.2f}\n")
            f.write(f"Average min temperature: {avg_min:.2f}\n")
        else:
            f.write("No temperature data available\n")
        
        f.write(f"Total residential electricity usage: {total_residential:.2f}\n\n")
        
        f.write("Date, Max Temperature, Residential Electricity Usage\n")
        for row in joined_data:
            f.write(f"{row[0]}, {row[1]}, {row[2]}\n")

calculate_and_write_results()

















#JOES DATA

API_URL = "https://api.electricitymap.org/v3/power-breakdown/history"
TOKEN = "M4sgDfSxIUYYx"
ZONES = ["US-MIDW-MISO", "US-MIDA-PJM", "US-NY-NYIS", "US-NE-ISNE", "US-CENT-SWPP"]

def fetch_electricity_map_data(max_items_per_zone=24):
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()

    # Create tables if they don't exist
    cur.execute('''
    CREATE TABLE IF NOT EXISTS electricity_map_data (
        zone TEXT,
        datetime TEXT,
        nuclear REAL,
        geothermal REAL,
        biomass REAL,
        coal REAL,
        wind REAL,
        solar REAL,
        hydro REAL,
        gas REAL,
        oil REAL,
        unknown REAL,
        PRIMARY KEY (zone, datetime)
    )''')

    # Create table to track last inserted datetime for each zone
    cur.execute('''
    CREATE TABLE IF NOT EXISTS last_inserted_datetime (
        zone TEXT PRIMARY KEY,
        datetime TEXT
    )''')

    total_items_inserted = 0
    for zone in ZONES:
        # Retrieve the last datetime data was inserted for the zone - problem was our zones were overlapping + a lack of data (24hrs)
        cur.execute("SELECT datetime FROM last_inserted_datetime WHERE zone = ?", (zone,))
        result = cur.fetchone()
        last_datetime = result[0] if result else None

        headers = {"auth-token": TOKEN}
        params = {"zone": zone}
        if last_datetime:
            cur.execute('''
            INSERT OR REPLACE INTO last_inserted_datetime (zone, datetime) VALUES (?, ?)
            ''', (zone, last_datetime))
            # Modify the query paramter if a last datetime exists 

        response = requests.get(API_URL, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json().get("history", [])
            zone_items_inserted = 0

            # Reverse to handle from oldest to newest
            for entry in reversed(data):
                if zone_items_inserted >= max_items_per_zone:
                    break
                
                current_datetime = entry.get("datetime")
                if not current_datetime or (last_datetime and current_datetime <= last_datetime):
                    continue

                power = entry.get("powerConsumptionBreakdown", {})
                try:
                    # Insert data into data table, ignoring duplicates
                    cur.execute('''
                    INSERT OR IGNORE INTO electricity_map_data
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        zone,
                        current_datetime,
                        power.get("nuclear", 0),
                        power.get("geothermal", 0),
                        power.get("biomass", 0),
                        power.get("coal", 0),
                        power.get("wind", 0),
                        power.get("solar", 0),
                        power.get("hydro", 0),
                        power.get("gas", 0),
                        power.get("oil", 0),
                        power.get("unknown", 0)
                    ))
                    if cur.rowcount > 0:
                        zone_items_inserted += 1
                        total_items_inserted += 1
                        last_datetime = current_datetime
                except sqlite3.Error as e:
                    print(f"Database error for {zone}, {current_datetime}: {e}")

            # Update last inserted datetime after processing data
            if data:
                cur.execute('''
                INSERT OR REPLACE INTO last_inserted_datetime (zone, datetime) VALUES (?, ?)
                ''', (zone, data[0].get("datetime")))

            print(f"Inserted {zone_items_inserted} items for zone {zone}")
        else:
            print(f"Error fetching data for zone {zone}: {response.status_code} - {response.text}")

    conn.commit()
    conn.close()
    print(f"Inserted {total_items_inserted} items into electricity_map_data")

if __name__ == "__main__":
    # Initate fetching and processing 
    print("Fetching electricity map data...")
    fetch_electricity_map_data() 
    print("Electricity map data processing complete.")
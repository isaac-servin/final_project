import sqlite3
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import os

def fetch_eia_data(url, table_name, num_cols):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    cells = soup.find_all('td', class_='r t sel-2 sel-13 data sel-22')
    
    data = []
    for i in range(0, min(25 * num_cols, len(cells)), num_cols):
        row_data = [cell.get_text(strip=True) for cell in cells[i:i + num_cols]]
        data.append(row_data)
    
    return data

def store_data(data, table_name):
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()
    
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
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS last_inserted_date (
        id INTEGER PRIMARY KEY,
        date TEXT
    )
    ''')

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

    for i in range(start_index, min(len(dates), start_index + max_items)):
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
ZONES = ["US-MIDW-MISO", "US-MIDA-PJM","US-NY-NYIS","US-NE-ISNE","US-CENT-SWPP"]  # Add more zones as needed

def fetch_electricity_data(max_items=25):
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
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS last_inserted_datetime (
        zone TEXT PRIMARY KEY,
        datetime TEXT
    )
    ''')

    items_inserted = 0

    for zone in ZONES:
        if items_inserted >= max_items:
            break
        
        # Get last inserted datetime for this zone
        cur.execute("SELECT datetime FROM last_inserted_datetime WHERE zone = ?", (zone,))
        result = cur.fetchone()
        last_datetime = result[0] if result else None
        print(f"Last datetime for {zone}: {last_datetime}")

        # Fetch data from API
        headers = {"auth-token": TOKEN}
        params = {"zone": zone}
        response = requests.get(API_URL, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json().get("history", [])
            print(f"Fetched {len(data)} entries for zone {zone}")

            # Insert new data into database
            for entry in reversed(data):  # Reverse to process oldest first
                if items_inserted >= max_items:
                    break
                
                current_datetime = entry.get("datetime")
                if last_datetime is None or current_datetime > last_datetime:
                    power = entry.get("powerConsumptionBreakdown", {})
                    print(f"Inserting: {zone}, {current_datetime}")
                    
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
                    items_inserted += 1
                    last_datetime = current_datetime

            # Update last inserted datetime
            cur.execute('''
            INSERT OR REPLACE INTO last_inserted_datetime (zone, datetime) VALUES (?, ?)
            ''', (zone, last_datetime))
        
        else:
            print(f"Error fetching data for zone {zone}: {response.status_code}")

    conn.commit()
    conn.close()
    print(f"Inserted {items_inserted} items into electricity_map_data")
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

#DON"T TOUCH 
def fetch_eia_data(url, table_name, num_cols):
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS LastProcessed (
        table_name TEXT PRIMARY KEY,
        last_index INTEGER DEFAULT 0
    )''')
    conn.commit()

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    cells = soup.find_all('td', class_='r t sel-2 sel-13 data sel-22')

    # Retrieve the last processed index
    last_index = get_last_saved_index(table_name, conn)
    print(f"Last processed index for {table_name}: {last_index}")  # Debugging output
    data = []
    for i in range(last_index * num_cols, min(len(cells), (last_index + 11) * num_cols), num_cols):
        # Process data in rows starting from the last saved index
        row_data = []
        for cell in cells[i:i + num_cols]:
            try:
                numeric_value = float(cell.get_text(strip=True).replace(',', ''))
                row_data.append(numeric_value)
            except ValueError:
                row_data.append(None)
        if len(row_data) == num_cols:
            data.append(row_data)
        else:
            print(f"Skipping row due to mismatched columns: {row_data}")
        
        if len(data) >= 11:
            break
    
    # Update the last processed index
    update_last_index(table_name, last_index + len(data), conn)
    print(f"New last index for {table_name}: {last_index + len(data)}")  # Debugging output
    conn.close()
    return data

def get_last_saved_index(table_name, conn):
    cur = conn.cursor()
    cur.execute("SELECT last_index FROM LastProcessed WHERE table_name = ?", (table_name,))
    result = cur.fetchone()
    if result is None:
        cur.execute("INSERT INTO LastProcessed (table_name, last_index) VALUES (?, ?)", (table_name, 0))
        conn.commit()
        return 0
    return result[0]

def update_last_index(table_name, last_index, conn):
    cur = conn.cursor()
    cur.execute("UPDATE LastProcessed SET last_index = ? WHERE table_name = ?", (last_index, table_name))
    conn.commit()

def store_data(data, table_name):
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()
    print(f"Storing data for {table_name}: {data}")  # Debugging output

    try:
        # Ensure the Years table exists
        cur.execute('''
        CREATE TABLE IF NOT EXISTS Years (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER UNIQUE
        )''')

        # Ensure the energy_data table exists
        cur.execute('''
        CREATE TABLE IF NOT EXISTS energy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year_id INTEGER,
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
            Estimated_Photovoltaic REAL,
            Residential REAL,
            Commercial REAL,
            Industrial REAL,
            Transportation REAL,
            Total REAL,
            Direct_Use REAL,
            Total_End_Use REAL,
            FOREIGN KEY (year_id) REFERENCES Years(id),
            UNIQUE(year_id, Coal, Petroleum, Natural_Gas, Other_Fossil_Gas)
        )''')

        for row in data:
            # Insert year and get year_id
            year = int(row[0])
            cur.execute('INSERT OR IGNORE INTO Years (year) VALUES (?)', (year,))
            cur.execute('SELECT id FROM Years WHERE year = ?', (year,))
            year_id = cur.fetchone()[0]

            if table_name == 'eia_data':
                values = (year_id, *row[1:], None, None, None, None, None, None, None)
                placeholders = ", ".join("?" * len(values))
                cur.execute(f'''
                INSERT OR IGNORE INTO energy_data (year_id, Coal, Petroleum, Natural_Gas, Other_Fossil_Gas, Nuclear,
                Hydroelectric_Conventional, Other_Renewable_Sources, Hydroelectric_Pumped_Storage,
                Other_Energy_Sources, Utility_Total, Estimated_Photovoltaic, Residential, Commercial,
                Industrial, Transportation, Total, Direct_Use, Total_End_Use)
                VALUES ({placeholders})
                ''', values)

            elif table_name == 'electricity_data':
                values = (year_id, None, None, None, None, None, None, None, None, None, None, None, *row[1:])
                placeholders = ", ".join("?" * len(values))
                cur.execute(f'''
                INSERT OR IGNORE INTO energy_data (year_id, Coal, Petroleum, Natural_Gas, Other_Fossil_Gas, Nuclear,
                Hydroelectric_Conventional, Other_Renewable_Sources, Hydroelectric_Pumped_Storage,
                Other_Energy_Sources, Utility_Total, Estimated_Photovoltaic, Residential, Commercial,
                Industrial, Transportation, Total, Direct_Use, Total_End_Use)
                VALUES ({placeholders})
                ''', values)

        conn.commit()
    finally:
        conn.close()

# DON'T TOUCH
def fetch_data_from_db():
    conn = sqlite3.connect('project_database.db')
    try:
        # Fetch all data from the unified energy_data table
        energy_df = pd.read_sql_query('''
        SELECT y.year, e.*
        FROM energy_data e
        JOIN Years y ON e.year_id = y.id
        ''', conn)

        # Columns specific to each dataset
        eia_columns = [
            'Coal', 'Petroleum', 'Natural_Gas', 'Other_Fossil_Gas', 'Nuclear',
            'Hydroelectric_Conventional', 'Other_Renewable_Sources',
            'Hydroelectric_Pumped_Storage', 'Other_Energy_Sources',
            'Utility_Total', 'Estimated_Photovoltaic'
        ]
        electricity_columns = [
            'Residential', 'Commercial', 'Industrial', 'Transportation',
            'Total', 'Direct_Use', 'Total_End_Use'
        ]

        # Separate data for eia and electricity
        eia_df = energy_df[['year'] + eia_columns]
        electricity_df = energy_df[['year'] + electricity_columns]

        return eia_df, electricity_df
    finally:
        conn.close()

def write_calculations(eia_df, electricity_df):
    if eia_df.empty or electricity_df.empty:
        print("DataFrames are empty. Skipping writing to file.")
        return

    # Specifying columns for energy production and consumption
    energy_columns = ['Coal', 'Petroleum', 'Natural_Gas', 'Other_Fossil_Gas', 'Nuclear', 
                      'Hydroelectric_Conventional', 'Other_Renewable_Sources', 'Hydroelectric_Pumped_Storage', 
                      'Other_Energy_Sources', 'Utility_Total', 'Estimated_Photovoltaic']
    consumption_columns = ['Residential', 'Commercial', 'Industrial', 'Transportation', 
                           'Total', 'Direct_Use', 'Total_End_Use']

    # Converting energy and consumption data to numeric
    eia_df[energy_columns] = eia_df[energy_columns].apply(pd.to_numeric, errors='coerce')
    electricity_df[consumption_columns] = electricity_df[consumption_columns].apply(pd.to_numeric, errors='coerce')

    # Calculating averages and sums
    avg_production = eia_df[energy_columns].mean(skipna=True)
    total_consumption = electricity_df[consumption_columns].sum(skipna=True)

    with open('calculations.txt', 'w') as f:
        f.write("Average Energy Production by Source:\n")
        for index, value in avg_production.items():
            f.write(f"{index}: {value:.2f}\n")

        f.write("Total Electricity Consumption by Sector:\n")
        for index, value in total_consumption.items():
            f.write(f"{index}: {value:.2f}\n")

def visualize_data(eia_df, electricity_df):
    conn = sqlite3.connect('project_database.db')

    # Updated queries to use energy_data
    eia_query = """
        SELECT y.year AS Year, e.Petroleum 
        FROM energy_data e
        JOIN Years y ON e.year_id = y.id
    """
    electricity_query = """
        SELECT y.year AS Year, e.Residential, e.Commercial, e.Industrial, e.Transportation 
        FROM energy_data e 
        JOIN Years y ON e.year_id = y.id
    """

    # Data into Pandas DataFrames
    try:
        eia_df = pd.read_sql_query(eia_query, conn)
        electricity_df = pd.read_sql_query(electricity_query, conn)
    except Exception as e:
        print("Error fetching data for visualization:", e)
        conn.close()
        return

    conn.close()

    # Visualization code remains unchanged
    # Convert columns to numeric
    eia_df['Year'] = pd.to_numeric(eia_df['Year'], errors='coerce')
    eia_df['Petroleum'] = pd.to_numeric(eia_df['Petroleum'], errors='coerce').fillna(0)  # Fill NaN with 0

    electricity_df['Year'] = pd.to_numeric(electricity_df['Year'], errors='coerce')
    numeric_columns = ['Residential', 'Commercial', 'Industrial', 'Transportation']
    electricity_df[numeric_columns] = electricity_df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)  # Fill NaN with 0

    # Merge on Year
    merged_data = pd.merge(eia_df, electricity_df, on='Year', how='inner')

    # Plot Petroleum over the years
    plt.figure(figsize=(12, 8))
    sns.barplot(data=merged_data, x='Year', y='Petroleum', color='orange')
    plt.title('Petroleum Production Over the Years', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Energy Production (Megawatts)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('petroleum_production_bar_seaborn.png')
    plt.show()

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

    # Check total entries in energy_data
    conn = sqlite3.connect('project_database.db')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM energy_data")
    print(f"Total entries in energy_data: {cur.fetchone()[0]}")
    conn.close()

    eia_df, electricity_df = fetch_data_from_db()
    visualize_data(eia_df, electricity_df)
    write_calculations(eia_df, electricity_df)
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

def load_weather_data(json_file, max_items=11):
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
    cur.execute("SELECT SUM(Residential) FROM energy_data")
    total_residential = cur.fetchone()[0] or 0.00  # Default to 0.00 if None

    # Join weather_data and energy_data
    cur.execute("""
    SELECT w.date, w.temperature_2m_max, e.Residential
    FROM weather_data w
    JOIN Years y ON substr(w.date, 1, 4) = y.year
    JOIN energy_data e ON y.id = e.year_id
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

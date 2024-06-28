import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
from time import sleep
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import altair as alt

# Function to fetch F1 data
@st.cache_data
def fetch_f1_data(session_key):
    base_url = "https://api.openf1.org/v1"
    
    # Fetch session info
    session_url = f"{base_url}/sessions?session_key={session_key}"
    with urlopen(session_url) as response:
        session = json.loads(response.read().decode('utf-8'))[0]
    sleep(1)  # Delay to respect rate limits
    
    # Fetch lap times
    laps_url = f"{base_url}/laps?session_key={session_key}"
    with urlopen(laps_url) as response:
        lap_times = json.loads(response.read().decode('utf-8'))
    sleep(1)  # Delay to respect rate limits
    
    # Fetch intervals
    intervals_url = f"{base_url}/intervals?session_key={session_key}"
    with urlopen(intervals_url) as response:
        intervals = json.loads(response.read().decode('utf-8'))
    sleep(1)  # Delay to respect rate limits
    
    # Fetch pit stops
    pits_url = f"{base_url}/pit?session_key={session_key}"
    with urlopen(pits_url) as response:
        pit_stops = json.loads(response.read().decode('utf-8'))
    
    return session, lap_times, intervals, pit_stops

# Function to process driver data
def process_driver_data(lap_times, intervals, pit_stops, driver_number):
    driver_laps = [lap for lap in lap_times if lap['driver_number'] == driver_number]
    driver_intervals = [interval for interval in intervals if interval['driver_number'] == driver_number]
    driver_pits = [pit for pit in pit_stops if pit['driver_number'] == driver_number]
    
    df = pd.DataFrame(driver_laps)
    df['date'] = pd.to_datetime(df['date_start'])
    df = df.dropna(subset=['date'])  # Remove rows with null dates
    df = df.sort_values('date')
    df['lap_time'] = df['lap_duration'].fillna(0)
    df['cumulative_time'] = df['lap_time'].cumsum()
    
    # Add interval data
    interval_df = pd.DataFrame(driver_intervals)
    interval_df['date'] = pd.to_datetime(interval_df['date'])
    interval_df = interval_df.dropna(subset=['date'])  # Remove rows with null dates
    interval_df = interval_df.sort_values('date')
    
    # Merge lap data with interval data
    df = pd.merge_asof(df, interval_df[['date', 'interval']], on='date', direction='nearest')
    
    # Add pit stop data
    for pit in driver_pits:
        pit_date = pd.to_datetime(pit['date'])
        closest_lap = df.loc[df['date'] >= pit_date].iloc[0] if not df[df['date'] >= pit_date].empty else None
        if closest_lap is not None:
            df.loc[closest_lap.name, 'pit_duration'] = pit['pit_duration']
    
    df['pit_duration'] = df['pit_duration'].fillna(0)
    df['lap_time_without_pit'] = df['lap_time'] - df['pit_duration']
    
    return df

# Function to visualize comparison
def visualize_comparison(driver1_df, driver2_df, driver1_name, driver2_name):
    fig, ax1 = plt.subplots(figsize=(15, 10))
    
    # Plot lap times
    ax1.plot(driver1_df['lap_number'], driver1_df['lap_time_without_pit'], label=f'{driver1_name} Lap Time', color='blue')
    ax1.plot(driver2_df['lap_number'], driver2_df['lap_time_without_pit'], label=f'{driver2_name} Lap Time', color='orange')
    
    # Plot pit stops
    driver1_pits = driver1_df[driver1_df['pit_duration'] > 0]
    driver2_pits = driver2_df[driver2_df['pit_duration'] > 0]
    ax1.scatter(driver1_pits['lap_number'], driver1_pits['lap_time'], color='blue', s=100, marker='^', label=f'{driver1_name} Pit Stop')
    ax1.scatter(driver2_pits['lap_number'], driver2_pits['lap_time'], color='orange', s=100, marker='^', label=f'{driver2_name} Pit Stop')
    
    ax1.set_xlabel('Lap Number')
    ax1.set_ylabel('Lap Time (seconds)')
    ax1.set_title(f'Lap-by-Lap Pace Comparison: {driver1_name} vs {driver2_name}')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Add a secondary y-axis for the interval if data is available
    if 'interval' in driver2_df.columns and not driver2_df['interval'].isnull().all():
        ax2 = ax1.twinx()
        ax2.plot(driver2_df['lap_number'], driver2_df['interval'], color='green', linestyle='--', label=f'{driver2_name} Interval to Leader')
        ax2.set_ylabel('Interval to Leader (seconds)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# Function to analyze results
def analyze_results(driver1_df, driver2_df, driver1_name, driver2_name):
    analysis = f"\nAnalysis Results:\n"
    analysis += f"{driver1_name} completed {len(driver1_df)} laps\n"
    analysis += f"{driver2_name} completed {len(driver2_df)} laps\n\n"
    
    analysis += f"{driver1_name}'s average lap time (excluding pits): {driver1_df['lap_time_without_pit'].mean():.3f} seconds\n"
    analysis += f"{driver2_name}'s average lap time (excluding pits): {driver2_df['lap_time_without_pit'].mean():.3f} seconds\n\n"
    
    analysis += f"{driver1_name}'s fastest lap: {driver1_df['lap_time_without_pit'].min():.3f} seconds\n"
    analysis += f"{driver2_name}'s fastest lap: {driver2_df['lap_time_without_pit'].min():.3f} seconds\n\n"
    
    analysis += f"{driver1_name}'s total pit time: {driver1_df['pit_duration'].sum():.3f} seconds\n"
    analysis += f"{driver2_name}'s total pit time: {driver2_df['pit_duration'].sum():.3f} seconds\n\n"
    
    time_difference = driver1_df['cumulative_time'].iloc[-1] - driver2_df['cumulative_time'].iloc[-1]
    analysis += f"Final time difference: {abs(time_difference):.3f} seconds\n"
    winner = driver1_name if time_difference < 0 else driver2_name
    analysis += f"{winner} finished ahead\n\n"
    
    # Analyze pace
    driver1_pace = driver1_df['lap_time_without_pit'].median()
    driver2_pace = driver2_df['lap_time_without_pit'].median()
    if driver2_pace < driver1_pace:
        analysis += f"{driver2_name} had a faster median pace than {driver1_name}.\n"
    else:
        analysis += f"{driver1_name} had a faster median pace than {driver2_name}.\n"
    
    # Analyze pit strategy impact
    pit_difference = driver2_df['pit_duration'].sum() - driver1_df['pit_duration'].sum()
    if abs(pit_difference) > abs(time_difference):
        analysis += "The pit strategy had a significant impact on the final result.\n"
    else:
        analysis += "The pit strategy did not significantly impact the final result.\n"
    
    return analysis

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'session' not in st.session_state:
    st.session_state.session = None
if 'lap_times' not in st.session_state:
    st.session_state.lap_times = None
if 'intervals' not in st.session_state:
    st.session_state.intervals = None
if 'pit_stops' not in st.session_state:
    st.session_state.pit_stops = None
if 'driver_numbers' not in st.session_state:
    st.session_state.driver_numbers = []

def prepare_data_for_ml(lap_times, intervals, pit_stops):
    all_driver_data = []
    for driver_number in set(lap['driver_number'] for lap in lap_times):
        driver_df = process_driver_data(lap_times, intervals, pit_stops, driver_number)
        if not driver_df.empty:
            driver_stats = {
                'driver_number': driver_number,
                'avg_lap_time': driver_df['lap_time_without_pit'].mean(),
                'fastest_lap': driver_df['lap_time_without_pit'].min(),
                'total_pit_time': driver_df['pit_duration'].sum(),
                'laps_completed': len(driver_df),
                'avg_interval': driver_df['interval'].mean() if 'interval' in driver_df.columns else np.nan
            }
            all_driver_data.append(driver_stats)
    return pd.DataFrame(all_driver_data)

# New function to train ML model and make predictions
def train_and_predict(historical_data, next_race_data):
    # Prepare the data
    X = historical_data.drop(['driver_number', 'position'], axis=1)
    y = historical_data['position']
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    # Make predictions for the next race
    next_race_predictions = model.predict_proba(next_race_data.drop('driver_number', axis=1))
    
    # Get the driver numbers and their win probabilities
    driver_probs = list(zip(next_race_data['driver_number'], next_race_predictions[:, 0]))
    
    # Sort by win probability (highest first) and exclude 0% probabilities
    driver_probs = [(driver, prob) for driver, prob in driver_probs if prob > 0]
    driver_probs.sort(key=lambda x: x[1], reverse=True)
    
    return driver_probs

def create_lap_time_distribution(driver1_df, driver2_df, driver1_name, driver2_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(driver1_df['lap_time_without_pit'], kde=True, color='blue', label=driver1_name, ax=ax)
    sns.histplot(driver2_df['lap_time_without_pit'], kde=True, color='orange', label=driver2_name, ax=ax)
    ax.set_title('Lap Time Distribution')
    ax.set_xlabel('Lap Time (seconds)')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig

def create_pace_evolution(driver1_df, driver2_df, driver1_name, driver2_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.regplot(x='lap_number', y='lap_time_without_pit', data=driver1_df, scatter=True, color='blue', label=driver1_name, ax=ax)
    sns.regplot(x='lap_number', y='lap_time_without_pit', data=driver2_df, scatter=True, color='orange', label=driver2_name, ax=ax)
    ax.set_title('Pace Evolution Throughout the Race')
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Lap Time (seconds)')
    ax.legend()
    return fig

def create_interval_heatmap(driver1_df, driver2_df, driver1_name, driver2_name):
    interval_diff = driver2_df['interval'] - driver1_df['interval']
    interval_diff = interval_diff.fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(interval_diff.to_frame().T, cmap='RdYlGn_r', center=0, ax=ax)
    ax.set_title(f'Interval Difference: {driver2_name} vs {driver1_name}')
    ax.set_xlabel('Lap Number')
    ax.set_yticklabels([])
    plt.colorbar(ax.collections[0], label='Interval Difference (seconds)', ax=ax)
    return fig

# Modify the main function to include ML prediction
def main():
    st.title("F1 H2H Race Analysis and Prediction")
    
    # Session key input
    session_key = st.number_input("Enter the session key for the race:", value=9539, step=1)
    
    if st.button("Fetch Data"):
        # Fetch data
        with st.spinner("Fetching data..."):
            st.session_state.session, st.session_state.lap_times, st.session_state.intervals, st.session_state.pit_stops = fetch_f1_data(session_key)
        st.session_state.data_fetched = True
        st.session_state.driver_numbers = sorted(list(set(lap['driver_number'] for lap in st.session_state.lap_times)))
        st.success(f"Data fetched for {st.session_state.session['session_name']} at {st.session_state.session['circuit_short_name']}")
        
        # Prepare data for ML model
        st.session_state.ml_data = prepare_data_for_ml(st.session_state.lap_times, st.session_state.intervals, st.session_state.pit_stops)
    
    if st.session_state.data_fetched:
        # Driver selection for analysis
        st.subheader("Driver Comparison Analysis")
        col1, col2 = st.columns(2)
        with col1:
            driver1 = st.selectbox("Select first driver:", st.session_state.driver_numbers, index=0, key="driver1")
        with col2:
            driver2 = st.selectbox("Select second driver:", st.session_state.driver_numbers, index=1, key="driver2")
        
        if st.button("Analyze"):
            # Process data for selected drivers
            driver1_df = process_driver_data(st.session_state.lap_times, st.session_state.intervals, st.session_state.pit_stops, driver1)
            driver2_df = process_driver_data(st.session_state.lap_times, st.session_state.intervals, st.session_state.pit_stops, driver2)
            
            # Visualize comparison
            st.pyplot(visualize_comparison(driver1_df, driver2_df, f"Driver {driver1}", f"Driver {driver2}"))
            
            # Display analysis
            st.text(analyze_results(driver1_df, driver2_df, f"Driver {driver1}", f"Driver {driver2}"))

            # New Seaborn visualizations
            st.subheader("Detailed Analysis")

            # Lap Time Distribution
            st.pyplot(create_lap_time_distribution(driver1_df, driver2_df, f"Driver {driver1}", f"Driver {driver2}"))
            st.write("This plot shows the distribution of lap times for both drivers. It helps to understand the consistency and overall pace of each driver.")

            # Pace Evolution
            st.pyplot(create_pace_evolution(driver1_df, driver2_df, f"Driver {driver1}", f"Driver {driver2}"))
            st.write("This plot shows how the pace of each driver evolved throughout the race. The trend lines help to identify if a driver was getting faster or slower as the race progressed.")

            # Interval Heatmap
            st.pyplot(create_interval_heatmap(driver1_df, driver2_df, f"Driver {driver1}", f"Driver {driver2}"))
            st.write("This heatmap shows the difference in interval times between the two drivers throughout the race. Green indicates Driver 1 is ahead, while red indicates Driver 2 is ahead.")

            # Interactive Altair chart for lap times
            lap_times_df = pd.melt(pd.DataFrame({
                'Lap': driver1_df['lap_number'],
                f'Driver {driver1}': driver1_df['lap_time_without_pit'],
                f'Driver {driver2}': driver2_df['lap_time_without_pit']
            }), id_vars=['Lap'], var_name='Driver', value_name='Lap Time')

            chart = alt.Chart(lap_times_df).mark_line().encode(
                x='Lap:Q',
                y='Lap Time:Q',
                color='Driver:N',
                tooltip=['Lap', 'Driver', 'Lap Time']
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
            st.write("This interactive chart allows you to hover over the lines to see exact lap times for each driver.")

        
        # ML Prediction for next race
        st.subheader("Prediction for Next Race")
        if st.button("Predict Next Race"):
            # For this example, we'll use the current race data to predict itself
            # In a real scenario, you'd have historical data from multiple races
            historical_data = st.session_state.ml_data.copy()
            historical_data['position'] = range(1, len(historical_data) + 1)  # Assign positions based on performance
            
            next_race_data = st.session_state.ml_data.copy()
            
            predictions = train_and_predict(historical_data, next_race_data)
            
            st.write("Predicted Race Results (excluding drivers with 0% chance):")
            if predictions:
                for i, (driver, prob) in enumerate(predictions, 1):
                    st.write(f"{i}. Driver {driver}: {prob*100:.2f}% chance of winning")
            else:
                st.write("No drivers predicted to have a chance of winning. This might indicate an issue with the model or data.")

            # Display a pie chart of the predictions
            if predictions:
                fig, ax = plt.subplots()
                drivers, probs = zip(*predictions)
                ax.pie(probs, labels=[f'Driver {d}' for d in drivers], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title("Predicted Win Probabilities")
                st.pyplot(fig)

if __name__ == "__main__":
    main()

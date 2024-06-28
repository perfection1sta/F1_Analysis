# F1 Race Analysis and Prediction

This Streamlit application provides in-depth analysis and predictions for Formula 1 races using data from the OpenF1 API. It offers detailed comparisons between drivers, race visualizations, and machine learning predictions for race outcomes.

## Features

- Fetch and analyze real-time F1 race data
- Compare performance between two selected drivers
- Visualize lap times, pit stops, and intervals
- Analyze race strategies and their impact
- Predict race outcomes using machine learning
- Interactive charts and heatmaps for detailed insights

## Installation

1. Clone this repository:
git clone https://github.com/{yourusername}/f1_compare_streamlit.git
cd f1_compare_streamlit

2. Install the required packages:
pip install -r requirements.txt

## Usage

1. Run the Streamlit app:
run f1_compare_streamlit.py

2. Open your web browser and go to `http://localhost:8501`

3. Enter the session key for the race you want to analyze

4. Use the interface to select drivers and analyze their performance

## Data Source

This application uses data from the [OpenF1 API](https://openf1.org/). Please refer to their documentation for more information on available data and usage limits.

## Features in Detail

### Data Fetching
- Retrieves session info, lap times, intervals, and pit stop data for a given race

### Driver Comparison
- Allows selection of two drivers for detailed comparison
- Visualizes lap-by-lap pace, pit stops, and intervals

### Race Analysis
- Provides insights on average lap times, fastest laps, pit strategies, and overall performance

### Visualizations
- Lap time distribution plots
- Pace evolution throughout the race
- Interval heatmaps
- Interactive lap time charts

### Machine Learning Predictions
- Uses Random Forest Classifier to predict race outcomes
- Visualizes win probabilities for drivers

## Contributing

Contributions to improve the application are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [OpenF1 API](https://openf1.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

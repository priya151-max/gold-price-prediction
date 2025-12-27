from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

app = Flask(__name__)

# Load data
def load_data():
    df = pd.read_csv("gold_rate_inr_preprocessed.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df = df.dropna(subset=["Date", "Price"])
    df = df.sort_values("Date")
    df["Price"] = df["Price"] / 31.1035  # Convert to INR/gram
    return df

df = load_data()

# KMeans Clustering Function
def apply_kmeans(data, num_clusters=3):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    return kmeans, clusters, scaler

@app.route('/anomaly-detection')
def anomaly_detection():
    if df.empty:
        return jsonify({"error": "No data available for anomaly detection."}), 500

    prices = df[['Date', 'Price']].copy()
    kmeans, clusters, scaler = apply_kmeans(prices[['Price']])
    prices['Cluster'] = clusters
    prices['ScaledPrice'] = scaler.transform(prices[['Price']])
    centroids = kmeans.cluster_centers_

    prices['Distance'] = np.linalg.norm(prices[['ScaledPrice']] - centroids[clusters], axis=1)
    threshold = prices['Distance'].mean() + prices['Distance'].std()
    prices['Anomaly'] = prices['Distance'] > threshold

    high_outliers = prices[prices['Anomaly']]
    low_outliers = prices[~prices['Anomaly']]

    return jsonify({
        'high_outliers': high_outliers[['Date', 'Price']].to_dict(orient='records'),
        'low_outliers': low_outliers[['Date', 'Price']].to_dict(orient='records'),
        'outlier_count': {
            'high': len(high_outliers),
            'low': len(low_outliers)
        }
    })

@app.route('/anomaly-plot')
def anomaly_plot():
    if df.empty:
        return "No data available", 500

    # Perform KMeans clustering and anomaly detection
    prices = df[['Date', 'Price']].copy()
    kmeans, clusters, scaler = apply_kmeans(prices[['Price']])
    prices['Cluster'] = clusters
    prices['ScaledPrice'] = scaler.transform(prices[['Price']])
    centroids = kmeans.cluster_centers_
    prices['Distance'] = np.linalg.norm(prices[['ScaledPrice']] - centroids[clusters], axis=1)
    threshold = prices['Distance'].mean() + prices['Distance'].std()
    prices['Anomaly'] = prices['Distance'] > threshold

    # Plotting Clusters with Anomalies
    plt.figure(figsize=(12, 6))

    # Scatter plot for the clusters
    scatter = plt.scatter(prices['Date'], prices['Price'], c=prices['Cluster'], cmap='viridis', label='Cluster', marker='o', alpha=0.6)

    # Mark anomalies with a different color
    plt.scatter(prices[prices['Anomaly']]['Date'], prices[prices['Anomaly']]['Price'], color='red', label='Anomalies', marker='x', s=100)

    # Add centroids to the plot
    plt.scatter(prices['Date'].iloc[0], centroids[0], color='blue', marker='*', edgecolors='black', s=200, label='Centroids')
    for i, centroid in enumerate(centroids[1:]):
        plt.scatter(prices['Date'].iloc[i + 1], centroid, color='blue', edgecolors='black', s=200, marker='*')

    # Add labels and title
    plt.title("Gold Price Clustering and Anomaly Detection (KMeans)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (INR/gram)", fontsize=12)
    plt.legend()

    # Improve the layout and tick formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

@app.route('/data')
def get_today_data():
    if df.empty:
        return jsonify({"error": "Data not available."}), 500
    
    today = pd.Timestamp.today().date()
    today_data = df[df['Date'] == today]

    if today_data.empty:
        return jsonify({"error": "No data available for today."}), 404
    
    today_row = today_data.iloc[0]

    price = today_row['Price']
    high = today_data['High'].max()
    low = today_data['Low'].min()
    open_price = today_row.get('Open', price)
    
    return jsonify({
        'date': today.strftime('%d-%m-%Y'),
        'price': round(price, 2),
        'high': round(high, 2),
        'low': round(low, 2),
        'change_rs': round(price - open_price, 2),
        'change_percent': round(((price - open_price) / open_price * 100), 2) if open_price else 0
    })

@app.route('/yesterday-data')
def get_yesterday_data():
    if df.empty:
        return jsonify({"error": "Data not available."}), 500

    # Getting data for yesterday
    yesterday = pd.Timestamp.today() - pd.Timedelta(days=1)
    yesterday_data = df[df['Date'] == yesterday.date()].copy()
    
    if not yesterday_data.empty:
        yesterday_row = yesterday_data.iloc[0]
        return jsonify({
            'change_rs': round(yesterday_row['Price'] - yesterday_row.get('Open', yesterday_row['Price']), 2)
        })

    return jsonify({'change_rs': 0})

@app.route('/gold-data', methods=['GET'])
def get_gold_data():
    today = pd.Timestamp.today()
    recent_data = df[df['Date'] <= today].copy()
    today_row = recent_data.iloc[-1] if not recent_data.empty else None

    if today_row is not None:
        today_price = today_row['Price']
        high_price = recent_data['Price'].max()
        low_price = recent_data['Price'].min()
        previous_price = recent_data.iloc[-2]['Price'] if len(recent_data) > 1 else today_price
        change_in_rupes = today_price - previous_price
        change_in_percent = (change_in_rupes / previous_price) * 100 if previous_price else 0
    else:
        today_price = high_price = low_price = change_in_rupes = change_in_percent = 0

    return jsonify({
        'today_price': round(today_price, 2),
        'high_price': round(high_price, 2),
        'low_price': round(low_price, 2),
        'change_in_rupes': round(change_in_rupes, 2),
        'change_in_percent': round(change_in_percent, 2),
        'today_date': str(today.date()),
        'all_data': df.to_dict(orient='records')
    })

@app.route('/gold-data/<timeframe>', methods=['GET'])
def get_gold_data_filtered(timeframe):
    today = pd.Timestamp.today()
    if timeframe == 'week':
        start_date = today - pd.Timedelta(days=7)
    elif timeframe == 'month':
        start_date = today - pd.DateOffset(months=1)
    elif timeframe == '1y':
        start_date = today - pd.DateOffset(years=1)
    elif timeframe == '10y':
        start_date = today - pd.DateOffset(years=10)
    elif timeframe == '25y':
        start_date = today - pd.DateOffset(years=25)
    else:
        return jsonify({"error": "Invalid timeframe"}), 400

    filtered_data = df[df['Date'] >= start_date]
    return jsonify(filtered_data.to_dict(orient='records'))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        future_date_str = request.form.get('future_date')
        try:
            future_date = datetime.strptime(future_date_str, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."})

        today = datetime.today().date()
        if future_date <= today:
            return jsonify({"error": "Please enter a future date only."})

        if future_date.weekday() == 5:
            future_date = future_date - timedelta(days=1)
        elif future_date.weekday() == 6:
            future_date = future_date - timedelta(days=2)

        df_features = df.copy()
        df_features['DayOfYear'] = df_features['Date'].dt.dayofyear
        df_features['Year'] = df_features['Date'].dt.year
        df_features['Month'] = df_features['Date'].dt.month
        df_features['Day'] = df_features['Date'].dt.day
        df_features['Weekday'] = df_features['Date'].dt.weekday
        df_features['Index'] = range(len(df_features))

        X = df_features[['DayOfYear', 'Year', 'Month', 'Day', 'Weekday', 'Index']]
        y = df_features['Price']

        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X, y)

        input_data = pd.DataFrame([{
            'DayOfYear': future_date.timetuple().tm_yday,
            'Year': future_date.year,
            'Month': future_date.month,
            'Day': future_date.day,
            'Weekday': future_date.weekday(),
            'Index': len(df_features)
        }])

        predicted_price = model.predict(input_data)[0]
        today_price = df[df['Date'] <= pd.Timestamp.today()].iloc[-1]['Price']

        if abs(predicted_price - today_price) < 0.01:
            category = "Neutral"
        elif predicted_price > today_price:
            category = "High"
        else:
            category = "Low"

        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'predicted_category': category,
            'future_date': future_date.strftime('%Y-%m-%d')
        })

    return render_template('prediction.html')

@app.route('/analysis')
def analysis_page():
    return render_template('analysis.html')

@app.route('/conversion')
def conversion():
    return render_template('conversion.html')

@app.route('/')
def home():
    today = pd.Timestamp.today().date()
    today_data = df[df['Date'] == today]
    
    if today_data.empty:
        return render_template('index.html', error="No data available for today.")
    
    today_row = today_data.iloc[0]

    return render_template('index.html', 
                           date=today.strftime('%d-%m-%Y'),
                           price=round(today_row['Price'], 2),
                           high=round(today_row['High'], 2),
                           low=round(today_row['Low'], 2),
                           change_rs=round(today_row['Price'] - today_row.get('Open', today_row['Price']), 2),
                           change_percent=round(((today_row['Price'] - today_row.get('Open', today_row['Price'])) / today_row.get('Open', today_row['Price']) * 100) if today_row.get('Open', today_row['Price']) else 0, 2))

if __name__ == '__main__':
    app.run(debug=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def load_data():
    try:
        daily_url = "https://gist.githubusercontent.com/Assafdek/30ae463f3c736f0abd631aab3d25a16d/raw/33c9305aed76a63d62cace175b972890c5e04fa7/SampleDaily.csv"
        monthly_url = "https://gist.githubusercontent.com/Assafdek/d7cc0b9ec36d5e04dcf4e1f6b2f54221/raw/3a7235476d0a7833154223a73b28c4eaecfbe5c1/SampleMonthly.csv"
      
        # Load data as tab-separated
        daily = pd.read_csv(daily_url, sep='\t')
        monthly = pd.read_csv(monthly_url, sep='\t')
        
        print("Daily data columns:", daily.columns)
        print("Monthly data columns:", monthly.columns)
        
        for df in [daily, monthly]:
            # Convert 'Date' column to datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            
            # Process numeric columns
            for col in df.columns:
                if col != 'Date':
                    df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
        
        return daily, monthly
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    
# Calculate portfolio statistics
def calculate_portfolio_stats(df, stocks, weights):
    returns = df[stocks].pct_change().dropna()
    
    # Expected returns
    expected_returns = returns.mean() * 252  # Annualized
    
    # Standard deviations
    std_devs = returns.std() * np.sqrt(252)  # Annualized
    
    # Correlation matrix
    correlation_matrix = returns.corr()
    
    # Portfolio expected return
    portfolio_return = np.sum(expected_returns * weights)
    
    # Portfolio standard deviation
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    return expected_returns, std_devs, correlation_matrix, portfolio_return, portfolio_std_dev

# Initialize session state
if 'weights' not in st.session_state:
    st.session_state.weights = []

# App title
st.title('Stock Portfolio Analyzer')

# Load data
daily_data, monthly_data = load_data()

if daily_data is None or monthly_data is None:
    st.stop()

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Company selection
companies = [col for col in daily_data.columns if col.lower().endswith('close')]
num_stocks = st.sidebar.number_input('Number of stocks in portfolio (1-3)', min_value=1, max_value=3, value=1)

selected_stocks = []
weights = []

for i in range(num_stocks):
    available_companies = [c for c in companies if c not in selected_stocks]
    stock = st.sidebar.selectbox(f'Select Stock {i+1}', available_companies, key=f'stock_{i}')
    
    # Use session state to store weights
    if i >= len(st.session_state.weights):
        st.session_state.weights.append(1.0 / num_stocks)
    
    weight = st.sidebar.slider(f'Weight for {stock}', min_value=0.0, max_value=1.0, value=st.session_state.weights[i], step=0.01, key=f'weight_{i}')
    st.session_state.weights[i] = weight
    
    selected_stocks.append(stock)
    weights.append(weight)

# Normalize weights
weights = np.array(weights) / np.sum(weights)
st.session_state.weights = weights.tolist()

# Frequency selection
frequency = st.sidebar.radio('Select Frequency', ['Daily', 'Monthly'])

# Date range selection
df = daily_data if frequency == 'Daily' else monthly_data
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
start_date = st.sidebar.date_input('Start Date', min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input('End Date', max_date, min_value=min_date, max_value=max_date)

# Filter data based on user input
filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

# Calculate portfolio statistics
expected_returns, std_devs, correlation_matrix, portfolio_return, portfolio_std_dev = calculate_portfolio_stats(filtered_df, selected_stocks, weights)

# Display portfolio composition
st.subheader('Portfolio Composition')
for stock, weight in zip(selected_stocks, weights):
    st.write(f"{stock}: {weight:.2%}")

# Display correlation matrix
if len(selected_stocks) > 1:
    st.subheader('Correlation Matrix')
    fig_corr = px.imshow(correlation_matrix, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr)
else:
    st.write("Correlation matrix is not available for a single stock.")

# Display expected returns and standard deviations
st.subheader('Expected Annual Returns and Standard Deviations')
stats_df = pd.DataFrame({
    'Expected Return': expected_returns,
    'Standard Deviation': std_devs
})
st.dataframe(stats_df)

# Display portfolio statistics
st.subheader('Portfolio Statistics')
st.write(f"Expected Annual Return: {portfolio_return:.2%}")
st.write(f"Annual Standard Deviation: {portfolio_std_dev:.2%}")

# Plot stock prices
st.subheader('Stock Prices')
fig = go.Figure()
for stock in selected_stocks:
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[stock], mode='lines', name=stock))
fig.update_layout(title='Stock Prices Over Time', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)

# Plot cumulative returns
st.subheader('Cumulative Returns')
cumulative_returns = (1 + filtered_df[selected_stocks].pct_change()).cumprod()
fig_cumulative = go.Figure()
for stock in selected_stocks:
    fig_cumulative.add_trace(go.Scatter(x=filtered_df['Date'], y=cumulative_returns[stock], mode='lines', name=stock))
fig_cumulative.update_layout(title='Cumulative Returns Over Time', xaxis_title='Date', yaxis_title='Cumulative Return')
st.plotly_chart(fig_cumulative)
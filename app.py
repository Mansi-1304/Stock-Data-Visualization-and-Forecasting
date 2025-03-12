import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import sqlite3
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 🔹 Initialize session state variables (Add this block here)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'stock_data' not in st.session_state:  # Ensure stock_data is initialized
    st.session_state.stock_data = None
if 'signup_mode' not in st.session_state:
    st.session_state.signup_mode = False
if 'USER_CREDENTIALS' not in st.session_state:
    st.session_state.USER_CREDENTIALS = {}  # Ensure credentials are stored properly


# 🔹 Database Setup (Replaces JSON)
def init_db():
    """Initialize the SQLite database for user authentication."""
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

# 🔹 User Signup
def signup():
    """Handles user signup and stores credentials in SQLite."""
    st.sidebar.header('Sign Up')
    new_username = st.sidebar.text_input('Choose a Username', '')
    new_password = st.sidebar.text_input('Choose a Password', type='password')
    confirm_password = st.sidebar.text_input('Confirm Password', type='password')

    if st.sidebar.button('Sign Up'):
        if new_username and new_password:
            if new_password == confirm_password:
                conn = sqlite3.connect("users.db")
                cursor = conn.cursor()
                try:
                    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                                   (new_username, new_password))
                    conn.commit()
                    st.sidebar.success('✅ Account created successfully! Please log in.')
                    st.session_state.signup_mode = False
                except sqlite3.IntegrityError:
                    st.sidebar.error('❌ Username already exists! Choose a different one.')
                finally:
                    conn.close()
            else:
                st.sidebar.error('❌ Passwords do not match!')
        else:
            st.sidebar.error('❌ Please enter all fields.')

# 🔹 User Login
def login():
    """Handles user login and authentication via SQLite."""
    st.sidebar.header('Login')
    username = st.sidebar.text_input('Username', '')
    password = st.sidebar.text_input('Password', type='password')

    if st.sidebar.button('Login'):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            st.session_state.logged_in = True
            st.sidebar.success(f'✅ Welcome {username}!')
        else:
            st.sidebar.error('❌ Invalid credentials. Please try again.')

    if st.sidebar.button('Create an Account'):
        st.session_state.signup_mode = True

# 🔹 Logout
def logout():
    """Handles user logout."""
    st.sidebar.info('Welcome to the Stock Visualizzation And Forecasting App')
    if st.sidebar.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.stock_data = None  # Reset stock data
        st.sidebar.success('✅ You have logged out.')

# 🔹 Download Stock Data
def download_data():
    """Download stock data using Yahoo Finance."""
    option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
    today = datetime.date.today()
    duration = st.sidebar.number_input('Enter duration (days)', value=3000)
    start_date = today - datetime.timedelta(days=duration)
    start_date = st.sidebar.date_input('Start Date', value=start_date)
    end_date = st.sidebar.date_input('End Date', value=today)

    if st.sidebar.button('Download'):
        if start_date < end_date:
            st.sidebar.success(f'Downloading data from `{start_date}` to `{end_date}`...')
            df = yf.download(option, start=start_date, end=end_date, progress=False)

            if df.empty:
                st.sidebar.error("❌ No data downloaded. Check stock symbol or API.")
            else:
                st.sidebar.success("✅ Data downloaded successfully!")
                st.session_state.stock_data = df  # Store in session state
        else:
            st.sidebar.error('❌ End date must be after start date.')

# 🔹 Display Recent Stock Data
def dataframe():
    """Show recent stock data."""
    if st.session_state.stock_data is None:
        st.error("No stock data available. Please download first.")
        return
    st.header('Recent Data')
    st.dataframe(st.session_state.stock_data.tail(10))

# 🔹 Technical Indicators
def tech_indicators():
    """Visualize technical indicators"""
    if st.session_state.stock_data is None:
        st.error("No stock data available. Please download the data first.")
        return

    st.header('Technical Indicators')
    option = st.radio('Choose an Indicator', ['Close', 'MACD', 'RSI', 'SMA', 'EMA'])

    # ✅ Ensure Close Price is a 1D series
    close_price = st.session_state.stock_data['Close']
    if isinstance(close_price, pd.DataFrame):
        close_price = close_price.squeeze()  # Convert DataFrame to Series

    # 🚨 Debugging Output
    st.write(f"Debug - Close Price Shape: {close_price.shape}")
    st.write(f"Debug - Close Price First 5 Rows:\n{close_price.head()}")

    # Compute and display indicators
    if option == 'Close':
        st.write('Closing Price')
        st.line_chart(close_price)

    elif option == 'MACD':
        macd = MACD(close=close_price).macd()
        if macd.isna().all():
            st.error("MACD cannot be calculated. Try using more historical data.")
        else:
            st.write('MACD Indicator')
            st.line_chart(macd)

    elif option == 'RSI':
        rsi = RSIIndicator(close=close_price).rsi()
        if rsi.isna().all():
            st.error("RSI cannot be calculated. Ensure enough historical data.")
        else:
            st.write('RSI Indicator')
            st.line_chart(rsi)

    elif option == 'SMA':
        sma = SMAIndicator(close=close_price, window=14).sma_indicator()
        if sma.isna().all():
            st.error("SMA cannot be calculated. Try increasing data points.")
        else:
            st.write('Simple Moving Average')
            st.line_chart(sma)

    elif option == 'EMA':
        ema = EMAIndicator(close=close_price).ema_indicator()
        if ema.isna().all():
            st.error("EMA cannot be calculated. Ensure enough data is available.")
        else:
            st.write('Exponential Moving Average')
            st.line_chart(ema)


# 🔹 Stock Price Prediction
def predict():
    """Predict stock prices using different models."""
    if st.session_state.stock_data is None:
        st.error("No stock data available. Please download first.")
        return

    st.header('Stock Price Prediction')
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = int(st.number_input('How many days forecast?', value=5))

    if st.button('Predict'):
        df = st.session_state.stock_data.copy()

        if 'Close' not in df:
            st.error("Error: 'Close' column is missing in stock data.")
            return

        x = StandardScaler().fit_transform(df[['Close']].values)
        y = df[['Close']].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        engines = {
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBoostRegressor': XGBRegressor()
        }

        selected_model = engines[model]
        selected_model.fit(x_train, y_train)
        y_pred = selected_model.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f'R² Score: {r2:.2f}')
        st.write(f'Mean Absolute Error: {mae:.2f}')
        st.success("Prediction completed successfully!")

        future_x = x[-num:]  
        future_predictions = selected_model.predict(future_x)
        st.write("Future Predictions:", future_predictions)

# 🔹 Main Function
def main():
    """Main app logic."""
    if st.session_state.signup_mode:
        signup()
    elif st.session_state.logged_in:
        st.title('Stock Price Predictions')
        option = st.sidebar.selectbox('Choose an Option', ['Visualize', 'Recent Data', 'Predict'])
        if option == 'Visualize':
            tech_indicators()
        elif option == 'Recent Data':
            dataframe()
        else:
            predict()
    else:
        st.title('Login')
        login()

# 🔹 Run the App
if __name__ == '__main__':
    init_db()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'signup_mode' not in st.session_state:
        st.session_state.signup_mode = False

    if st.session_state.logged_in:
        logout()
        download_data()
        main()
    else:
        main()

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from PIL import Image

# Load data
df = pd.read_csv('gld_price_data.csv')

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop rows with invalid data
df = df.dropna()

# Split into X and Y
X = df.drop(['Date', 'GLD'], axis=1)
Y = df['GLD']
print(X.shape, "\n", Y.shape)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2)
print(X_train.shape, X_test.shape)

reg = RandomForestRegressor()
reg.fit(X_train, Y_train)
pred = reg.predict(X_test)
score = r2_score(Y_test, pred)

# Prediction tab
def prediction_tab():
    st.title('Halo Future Gold Explorer!')

    # Change background and add more styling
    st.markdown(
        """
        <style>
            body { }
            .stApp {
                background-color: burlywood;
                color: black;
                text-align:justify;
            }
            p {
            font-size: 18px;  /* Adjust the font size as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        \nWelcome to the Prediction Zone! Here, you hold the power to shape future insights on gold prices. Input your predictions by providing key variables such as the date, stock market indices (SPX), oil prices (USO), silver prices (SLV), and currency exchange rates (EUR/USD). As you submit these inputs, our advanced model works its magic, crunching the data to forecast the future price of gold. Explore different scenarios, experiment with varying inputs, and witness how your unique insights influence the predictions. Make informed decisions and embark on your journey as a gold price forecaster in this interactive and exploratory Prediction Zone.
        """,
        unsafe_allow_html=True,
    )

    # Find the earliest date in the training dataset
    earliest_date = df['Date'].min()

    # Calculate the number of days since the earliest date in the training dataset
    df['Days_Since_Earliest'] = (df['Date'] - earliest_date).dt.days

    # Take user input for features
    st.sidebar.title("Enter Feature Values")
    selected_date = st.sidebar.text_input("Date (YYYY-MM-DD):")
    selected_spx = st.sidebar.number_input("SPX:")
    selected_uso = st.sidebar.number_input("USO:")
    selected_slv = st.sidebar.number_input("SLV:")
    selected_eur_usd = st.sidebar.number_input("EUR/USD:")

    # Convert user input date to datetime
    user_input_date = pd.to_datetime(selected_date, errors='coerce')
    if not user_input_date:
        st.sidebar.error("Please enter a valid date in the format YYYY-MM-DD.")
        st.stop()

    # Calculate the number of days since the earliest date in the training dataset for user input
    days_since_earliest = (user_input_date - earliest_date).days

    # Create user input DataFrame
    user_input = pd.DataFrame({
        'SPX': [selected_spx],
        'USO': [selected_uso],
        'SLV': [selected_slv],
        'EUR/USD': [selected_eur_usd],
        'Days_Since_Earliest': [days_since_earliest]
    })

    # Features and target variables
    X = df[['SPX', 'USO', 'SLV', 'EUR/USD', 'Days_Since_Earliest']]
    Y = df['GLD']

    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

    # Train the model
    reg = RandomForestRegressor()
    reg.fit(X_train, Y_train)

    # Make prediction for user input
    user_pred = reg.predict(user_input)

    # Convert predicted price to Indian Rupees (assuming an exchange rate of 74.5)
    exchange_rate_usd_to_inr = 74.5
    predicted_price_inr = user_pred[0] * exchange_rate_usd_to_inr

    # Display prediction
    st.subheader('User Input Features:')
    st.write(user_input)

    st.subheader('Predicted Gold Price in US Dollar:')
    st.write(f"${user_pred[0]:.2f}")

    st.subheader('Predicted Gold Price in Indian Rupees:')
    st.write(f"₹{predicted_price_inr:.2f}")

# About tab
def about_tab():
    st.title('About the Prediction Model!')

    # Change background and add more styling
    st.markdown(
        """
        <style>
            body {
                
            }
            .stApp {
                background-color: rgb(178, 215, 232);
                color: black;
                text-align: justify;
            }
            p {
            font-size: 18px;  /* Adjust the font size as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        \nThe centerpiece of this project is the Random Forest Regressor, a robust machine learning algorithm renowned for its versatility and accuracy in predictive modeling. In this context, the model is trained on a dataset containing features such as stock market indices, oil and silver prices, and currency exchange rates. The algorithm learns the underlying patterns and relationships within this historical data, allowing it to make accurate predictions about future gold prices.
        \nThe Random Forest Regressor is an ensemble learning method that builds a multitude of decision trees during training and outputs the average prediction of the individual trees for regression tasks. It excels in capturing complex relationships in data, handling outliers, and providing robust predictions. Its ability to mitigate overfitting and generalize well to new data makes it a suitable choice for predicting the intricate dynamics of gold prices.
        \nThe prediction process involves training the Random Forest Regressor on historical data, allowing it to learn the patterns and relationships within the features. Once trained, the model is ready to make predictions based on user inputs. Users provide specific values for the date, stock market indices, oil and silver prices, and currency exchange rates, and the model processes this input to generate a prediction for the future gold price. The result is a powerful tool that empowers users to explore, analyze, and anticipate the fascinating world of gold prices with confidence and insight.
        """,
        unsafe_allow_html=True,
    )

# Display model performance
    st.subheader('Using Random Forest Regressor')
    st.write(df)
    st.subheader('Model Performance:')
    st.write(score)

    # Display interactive plot
    st.subheader('Gold Price Over Time')
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['GLD'], label='Actual Gold Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Gold Price')
    ax.set_title('Gold Price Over Time')
    ax.legend()
    st.pyplot(fig)

# Home tab
def home_tab():
    st.title('Welcome to Gold Price Explorer!')

    # Change background and add more styling
    st.markdown(
        """
        <style>
            body { }
            .stApp {
                background-color: rgb(188, 188, 124);
                color: black;
                text-align: justify;
                
            }
            p {
            font-size: 18px;  /* Adjust the font size as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        \nGold Price Explorer is an immersive platform that offers users a unique journey through the intricate world of gold prices. It goes beyond traditional data analysis, providing a dynamic experience where enthusiasts can explore historical trends, predict future prices, and uncover valuable insights. The platform seamlessly combines informative visualizations, predictive modeling, and historical data, transforming gold price exploration into a compelling narrative.
        """,
        unsafe_allow_html=True,
    )

    # Display image
    img = Image.open('gold.jpng.webp')
    st.image(img, width=200, use_column_width=True)

    st.markdown(
        """
        The model employed in Gold Price Explorer leverages the power of the Random Forest Regressor, a robust machine learning algorithm. This model is adept at handling complex relationships within the data, making it particularly suitable for predicting gold prices influenced by various factors. The Random Forest Regressor works by constructing multiple decision trees during the training phase and outputting the average prediction of the individual trees for more accurate and reliable results.

        \nTo predict future gold prices, users input key variables such as the date, stock market indices (SPX), oil prices (USO), silver prices (SLV), and currency exchange rates (EUR/USD). The Random Forest Regressor then processes this input, analyzing historical patterns and relationships between the variables to generate a forecast. This allows users to explore different scenarios, experiment with inputs, and gain valuable insights into potential gold price movements.

        \nIn summary, Gold Price Explorer amalgamates historical data, advanced machine learning techniques, and user inputs to create an interactive platform where users can navigate through time, explore trends, and make informed predictions about the future of gold prices.
        """,
        unsafe_allow_html=True
    )

    
# Create tabs
tabs = ["Home","About","Future Gold Explorer"]
selected_tab = st.sidebar.radio("Welcome :)", tabs)

# Show the selected tab
if selected_tab == "Home":
    home_tab()
elif selected_tab == "About":
    about_tab()
else:
    prediction_tab()

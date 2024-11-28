import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors

# Initialize Spark session (if needed)
spark = SparkSession.builder.appName("CustomerBehaviorVisualization").getOrCreate()

# Load the trained model (replace 'model_path' with the path to your model)
model = LogisticRegressionModel.load("model_path")

# Load the dataset for visualization (replace 'dataset_path' with the path to your dataset)
data_path = "./datasets/PHVN_DATest_Dataset.json"
df = spark.read.json(data_path)
pandas_df = df.toPandas()

# Streamlit app begins
st.title("Pizza Hut Customer Behavior Analysis")

# Sidebar menu
menu = ["Overview", "Visualizations", "Predictive Modeling"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(pandas_df.head(10))
    
    st.write("### Basic Statistics")
    st.write(pandas_df.describe())

elif choice == "Visualizations":
    st.header("Interactive Visualizations")

    # Traffic Source Distribution
    st.subheader("Traffic Source Distribution")
    traffic_source = pandas_df["Source"].value_counts()
    fig, ax = plt.subplots()
    traffic_source.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Traffic Source Distribution")
    ax.set_xlabel("Source")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Event Name Distribution
    st.subheader("Event Name Distribution")
    event_name = pandas_df["EventName"].value_counts()
    fig, ax = plt.subplots()
    event_name.plot(kind='bar', ax=ax, color='green')
    ax.set_title("Event Name Distribution")
    ax.set_xlabel("Event Name")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Peak Hour Analysis
    st.subheader("Peak Hour Analysis")
    pandas_df["Hour"] = pd.to_datetime(pandas_df["EventDateTime"]).dt.hour
    peak_hours = pandas_df["Hour"].value_counts().sort_index()
    fig, ax = plt.subplots()
    peak_hours.plot(kind='line', ax=ax, marker='o', color='purple')
    ax.set_title("Peak Hour Activity")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif choice == "Predictive Modeling":
    st.header("Purchase Prediction")

    st.write("Input session-level features to predict the likelihood of a purchase.")

    # User inputs for prediction
    session_start = st.number_input("Number of Session Start Events", min_value=0, value=1)
    view_cart = st.number_input("Number of View Cart Events", min_value=0, value=0)
    add_to_cart = st.number_input("Number of Add to Cart Events", min_value=0, value=0)
    purchase = st.number_input("Number of Purchase Events", min_value=0, value=0)
    session_duration = st.number_input("Session Duration (seconds)", min_value=0, value=60)

    # Predict button
    if st.button("Predict"):
        # Prepare input data
        features = Vectors.dense([session_start, view_cart, add_to_cart, purchase, session_duration])
        input_df = spark.createDataFrame([(features,)], ["features"])

        # Predict using the trained model
        prediction = model.transform(input_df).select("probability").collect()[0][0]
        st.write(f"### Purchase Probability: {round(prediction[1] * 100, 2)}%")

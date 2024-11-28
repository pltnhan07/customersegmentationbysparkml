from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, lag, unix_timestamp
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler

def preprocess_data(df):
    # Step 1: Parse EventDateTime
    df = df.withColumn("EventDateTime", to_timestamp("EventDateTime", "dd-MM-yyyy HH:mm:ss"))
    
    # Step 2: Normalize Location
    df = df.selectExpr(
        "AppVersion", 
        "EventDateTime", 
        "EventName", 
        "Item",
        "Location.City as City", 
        "Location.Country as Country", 
        "Location.Region as Region",
        "MobileBrandName", 
        "SessionID", 
        "Source"
    )
    
    # Step 3: Filter Country == Vietnam
    df = df.filter(col("Country") == "Vietnam")
    
    # Step 4: Drop rows with missing values
    df = df.dropna()

    # Extract additional features: Hour and DayOfWeek
    df = df.withColumn("Hour", hour(col("EventDateTime"))).withColumn("DayOfWeek", dayofweek(col("EventDateTime")))
    
    # Calculate session duration
    window_spec = Window.partitionBy("SessionID").orderBy("EventDateTime")
    df = df.withColumn("PrevEventTime", lag("EventDateTime").over(window_spec))
    df = df.withColumn("EventDuration", unix_timestamp(col("EventDateTime")) - unix_timestamp(col("PrevEventTime")))
    session_durations = df.groupBy("SessionID").agg({"EventDuration": "sum"}).withColumnRenamed("sum(EventDuration)", "SessionDuration")
    
    # Pivot EventName for event counts
    session_features = df.groupBy("SessionID").pivot("EventName").count().fillna(0)
    
    # Merge session features and durations
    final_data = session_features.join(session_durations, on="SessionID")
    
    return final_data

def scale_features(data, input_cols, output_col="scaled_features"):
    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    data = assembler.transform(data)
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol=output_col)
    data = scaler.fit(data).transform(data)
    
    return data

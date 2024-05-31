from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

# Define the data path
data_path = "hdfs:///user/maria_dev/lxt/iris.data"

# Define the schema for the DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

schema = StructType([
    StructField("sepal_length", DoubleType(), True),
    StructField("sepal_width", DoubleType(), True),
    StructField("petal_length", DoubleType(), True),
    StructField("petal_width", DoubleType(), True),
    StructField("species", StringType(), True)
])

# Load the dataset
iris_df = spark.read.csv(data_path, schema=schema, header=False)
iris_df.show()

# Index laels, adding metadata to the label column
labelIndexer = StringIndexer(inputCol='species', outputCol='indexedLabel').fit(iris_df)

# Split the data into training and test sets (70% training, 30% testing)
train_data, test_data = iris_df.randomSplit([0.7, 0.3])

# Assemble features
feature_columns = iris_df.columns[:-1]  # Assuming the last column is the target
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Initialize the classifier
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")

# Create a Pipeline
pipeline = Pipeline(stages=[labelIndexer,assembler,rf])

# Build the parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Create evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

# Cross-validator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train_data)

# Make predictions on test data
predictions = cvModel.transform(test_data)

# Evaluate the model
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

# Show predicted and actual labels
predictions.select("prediction", "indexedLabel").show()

# Stop the Spark session
spark.stop()

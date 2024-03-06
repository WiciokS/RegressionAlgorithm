// Create MLContext
using Microsoft.ML;

MLContext mlContext = new MLContext();

// Load data
var trainingData = mlContext.Data.LoadFromTextFile<HouseData>("C:\\Users\\skude\\source\\repos\\RegressionAlgorithm\\RegressionAlgorithm\\house_data.csv", hasHeader: true, separatorChar: ',');

// Data process configuration with pipeline data transformations
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size", "NumBedrooms", "Age", "LocationIndex" })
    .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Price")); // Using FastTree for potentially better performance

// Train the model
var model = pipeline.Fit(trainingData);

// Example prediction
var sampleHouse = new HouseData() { Size = 2000f, NumBedrooms = 4, Age = 5, LocationIndex = 2 };
var pricePrediction = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(sampleHouse);

Console.WriteLine($"Predicted price: {pricePrediction.Price}");

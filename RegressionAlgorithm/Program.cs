// Create MLContext
using Microsoft.ML;

MLContext mlContext = new MLContext();

// Load data
var trainingData = mlContext.Data.LoadFromTextFile<HouseData>("C:\\Users\\skude\\source\\repos\\RegressionAlgorithm\\RegressionAlgorithm\\house_data.csv", hasHeader: true, separatorChar: ',');

// Data process configuration with pipeline data transformations 
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

// Train the model
var model = pipeline.Fit(trainingData);

// Use the model for a single prediction
var size = new HouseData() { Size = 750f };
var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

Console.WriteLine($"Predicted price for size {size.Size}: {price.Price}");
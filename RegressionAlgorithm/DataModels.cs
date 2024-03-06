using Microsoft.ML;
using Microsoft.ML.Data;

public class HouseData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1)]
    public float NumBedrooms { get; set; }

    [LoadColumn(2)]
    public float Age { get; set; }

    [LoadColumn(3)]
    public float LocationIndex { get; set; }

    [LoadColumn(4)]
    public float Price { get; set; }
}


public class Prediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

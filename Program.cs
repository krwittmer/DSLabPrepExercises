
namespace DSLabPrepExercises
{
    internal class Program
    {
        static void Main(string[] args)
        {
            goto KNNRegression;

        KNNClassification:
            Console.WriteLine("Launching the KNN classification demo program");
            KNNClassification.KNNClassificationProgram.MainKNN(args);
            return;

        KNNRegression:
            Console.WriteLine("Launching the KNN regression demo program");
            KNNRegression.KNNRegressionProgram.MainKNN(args);
            return;

        GradientDescent:
            Console.WriteLine("Launching the Gradient Descent regression (for logistics) program");
            LogisticGradientDescent.LogisticGradientProgram.MainGD(args);
            return;

        PrincipalComponentsClassic:
            Console.WriteLine("Launching the Principal Component Analysis (classical) program");
            PrincipalComponentsClassic.PrincipalClassicProgram.MainPCA(args);
            return;

        SupportVectorMachine:
            Console.WriteLine("Hello, World! Firing off on Support Vector Machine");
            SupportVectorMachine.SupportVectorMachineProgram.MainSVM(args);
            return;

        NeuralNetworkRegression:
            Console.WriteLine("Hello, World! Firing off on Neural Network");
            NeuralNetworkRegression.NeuralRegressionProgram.MainNN(args);
            return;
        }
    }
}

namespace Athenna
{
    public enum Activations : short
    {
        Sigmoid = 0,
        TanH = 1,
        ReLU = 2,
        LeakyReLU = 3,
        Linear = 4,
        Softmax = 5,
    }

    public static class Utils
    {
        public static Activations StringToActivation(string action)
        {
            switch (action)
            {
                case "sigmoid": return Activations.Sigmoid;
                case "tanh": return Activations.TanH;
                case "relu": return Activations.ReLU;
                case "leakyrelu": return Activations.LeakyReLU;
                case "linear": return Activations.Linear;
                case "softmax": return Activations.Softmax;
                default: return Activations.ReLU;
            }
        }
    }
}

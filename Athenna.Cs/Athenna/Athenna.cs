using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Athenna
{
    public class Athenna
    {
        private Activations[] activations;
        private int[] layers;
        private float[][] neurons;
        private float[][] biases;
        private float[][][] weights;

        public float fitness = 0;
        public float learningRate = 0.01f;
        public float cost = 0;
        public string title;
        public string filePath;

        public Athenna(int[] layers, Activations[] activations)
        {
            this.layers = layers;
            this.activations = activations;

            InitNeurons();
            InitBiases();
            InitWeights();
        }


        private void InitNeurons()
        {
            neurons = new float[layers.Length][];
            for (int i = 0; i < layers.Length; i++)
            {
                neurons[i] = new float[layers[i]];
            }
        }

        private void InitBiases()
        {
            biases = new float[layers.Length-1][];
            for (int i = 1; i < layers.Length; i++)
            {
                biases[i-1] = new float[layers[i]];
                for (int j = 0; j < layers[i]; j++)
                {
                    biases[i - 1][j] = RandRng(-0.5f, 0.5f) / layers[i];
                }
            }
        }

        private void InitWeights()
        {
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> layerWeightsList = new List<float[]>();
                int neuronsInPreviousLayer = layers[i - 1];
                for (int j = 0; j < layers[i]; j++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer];
                    for (int k = 0; k < neuronsInPreviousLayer; k++)
                    {
                        neuronWeights[k] = RandRng(-0.5f, 0.5f) / (float)neuronsInPreviousLayer;
                    }
                    layerWeightsList.Add(neuronWeights);
                }
                weightsList.Add(layerWeightsList.ToArray());
            }
            weights = weightsList.ToArray();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] FeedForward(float[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                neurons[0][i] = inputs[i];
            }
            for (int i = 1; i < layers.Length; i++)
            {
                int layerIdx = i - 1;
                for (int j = 0; j < layers[i]; j++)
                {
                    float value = 0f;
                    for (int k = 0; k < layers[i - 1]; k++)
                    {
                        value += weights[i - 1][j][k] * neurons[i - 1][k];
                    }
                    neurons[i][j] = activate(value + biases[i - 1][j], layerIdx, j);
                }
                if (activations[layerIdx] == Activations.Softmax)
                {
                    var sigma = 0.0f;
                    for (int j = 0; j < layers[i]; j++)
                    {
                        sigma += neurons[i][j];
                    }
                    for (int j = 0; j < layers[i]; j++)
                    {
                        neurons[i][j] /= sigma;
                    }
                }
            }

            //var lastLayerIdx = layers[i]


            return neurons[layers.Length - 1];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float activate(float value, int layer, int idx = 0)//all activation functions
        {
            switch (activations[layer])
            {
                case Activations.Sigmoid:
                    return sigmoid(value);
                case Activations.TanH:
                    return tanh(value);
                case Activations.ReLU:
                    return relu(value);
                case Activations.LeakyReLU:
                    return leakyrelu(value);
                case Activations.Linear:
                    return value;
                case Activations.Softmax:
                    return softmax(value);
                default:
                    return relu(value);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float activateDer(float value, int layer, int neuronIdx)
        {
            switch (activations[layer])
            {
                case Activations.Sigmoid:
                    return sigmoidDer(value);
                case Activations.TanH:
                    return tanhDer(value);
                case Activations.ReLU:
                    return reluDer(value);
                case Activations.LeakyReLU:
                    return leakyreluDer(value);
                case Activations.Linear:
                    return 1.0f;
                case Activations.Softmax:
                    return softmaxDer(value);
                default:
                    return reluDer(value);
            }
        }

        private static Random rnd = new Random();

        public float RandRng(float min, float max)
        {
            float range = max - min;
            return ((float)rnd.NextDouble() * range) + min;
        }

        public float sigmoid(float x)
        {
            float k = (float)Math.Exp(x);
            return k / (1.0f + k);
        }
        public float tanh(float x)
        {
            return (float)Math.Tanh(x);
        }
        public float relu(float x)
        {
            return (0 >= x) ? 0 : x;
        }
        public float leakyrelu(float x)
        {
            return (0 >= x) ? 0.01f * x : x;
        }
        public float bileakyrelu(float x)
        {
            return -((0 >= -x) ? 0.01f * -x : -x);
        }
        public float sigmoidDer(float x)
        {
            return x * (1 - x);
        }
        public float tanhDer(float x)
        {
            return 1 - (x * x);
        }
        public float reluDer(float x)
        {
            return (0 >= x) ? 0 : 1;
        }
        public float leakyreluDer(float x)
        {
            return (0 >= x) ? 0.01f : 1;
        }
        public float bileakyreluDer(float x)
        {
            return -((0 >= -x) ? 0.01f : 1);
        }

        public float softmax(float x)
        {
            return MathF.Exp(x);
        }
        public float softmaxDer(float x)
        {
            return x * (1 - x);
        }
        public float softmaxDerDiff(float x, float y)
        {
            return -x * y;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void BackPropagate(float[] inputs, float[] expected)
        {
            float[] output = FeedForward(inputs);

            cost = 0;
            for (int i = 0; i < output.Length; i++)
            {
                cost += (float)Math.Pow(output[i] - expected[i], 2);
            }
            cost = cost / 2;

            float[][] gamma;

            List<float[]> gammaList = new List<float[]>();
            for (int i = 0; i < layers.Length; i++)
            {
                gammaList.Add(new float[layers[i]]);
            }
            gamma = gammaList.ToArray();

            int layer = layers.Length - 2;
            var lastLayerIdx = layers.Length - 1;

            if (activations[layer] == Activations.Softmax)
            {
                for (int i = 0; i < output.Length; i++)
                {
                    gamma[lastLayerIdx][i] = (output[i] - expected[i]) * (output[i] * (1.0f - output[i]));
                }
            }
            else if (activations[layer] == Activations.Linear)
            {
                for (int i = 0; i < output.Length; i++)
                {
                    gamma[lastLayerIdx][i] = output[i] - expected[i];
                }
            }
            else
            {
                for (int i = 0; i < output.Length; i++)
                {
                    gamma[lastLayerIdx][i] = (output[i] - expected[i]) * activateDer(output[i], layer, i);
                }
            }

            for (int i = 0; i < layers[lastLayerIdx]; i++)
            {
                biases[layers.Length - 2][i] -= gamma[layers.Length - 1][i] * learningRate;
                for (int j = 0; j < layers[layers.Length - 2]; j++)
                {
                    weights[layers.Length - 2][i][j] -= gamma[layers.Length - 1][i] * neurons[layers.Length - 2][j] * learningRate;//*learning 
                }
            }

            for (int i = layers.Length - 2; i > 0; i--)
            {
                layer = i - 1;
                for (int j = 0; j < layers[i]; j++)
                {
                    gamma[i][j] = 0;
                    for (int k = 0; k < gamma[i + 1].Length; k++)
                    {
                        gamma[i][j] += gamma[i + 1][k] * weights[i][k][j];
                    }
                    gamma[i][j] *= activateDer(neurons[i][j], layer, j);
                }
                for (int j = 0; j < layers[i]; j++)
                {
                    biases[i - 1][j] -= gamma[i][j] * learningRate;
                    for (int k = 0; k < layers[i - 1]; k++)
                    {
                        weights[i - 1][j][k] -= gamma[i][j] * neurons[i - 1][k] * learningRate;
                    }
                }
            }
        }

        public void Mutate(int high, float val)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                for (int j = 0; j < biases[i].Length; j++)
                {
                    biases[i][j] = (RandRng(0f, (float)high) <= 2) ? biases[i][j] += RandRng(-val, val) : biases[i][j];
                }
            }
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        weights[i][j][k] = (RandRng(0f, (float)high) <= 2) ? weights[i][j][k] += RandRng(-val, val) : weights[i][j][k];
                    }
                }
            }
        }

        public byte[] ComputeNetworkHash()
        {
            double sum = 0.0;
            for (int i = 0; i < biases.Length; i++)
            {
                for (int j = 0; j < biases[i].Length; j++)
                {
                    sum += biases[i][j] * 7;
                }
            }
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        sum += weights[i][j][k] * 3;
                    }
                }
            }
            return BitConverter.GetBytes(sum);
        }

        public int CompareTo(Athenna other)
        {
            if (other == null) return 1;

            if (fitness > other.fitness)
                return 1;
            else if (fitness < other.fitness)
                return -1;
            else
                return 0;
        }

        public Athenna Copy(Athenna nn)
        {
            for (int i = 0; i < biases.Length; i++)
            {
                for (int j = 0; j < biases[i].Length; j++)
                {
                    nn.biases[i][j] = biases[i][j];
                }
            }
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        nn.weights[i][j][k] = weights[i][j][k];
                    }
                }
            }
            return nn;
        }

        public static Athenna Load(string path)
        {
            var labelIdx = 0;
            var activations = new Activations[] { Activations.Linear };
            var layers = new int[] { 1, 1 };
            var weights = new List<float>();
            var biases = new List<float>();
            int layersIdx = 0;
            int activationsIdx = 0;

            Athenna nn = new Athenna(layers, activations);

            using (var stream = new StreamReader(path))
            {
                while (!stream.EndOfStream)
                {
                    var line = stream.ReadLine();

                    switch (line) {

                        case "[network.type]":
                            labelIdx = 1;
                            break;
                        case "[layer.neurons]":
                            labelIdx = 2;
                            break;
                        case "[layer.activations]":
                            labelIdx = 3;
                            break;
                        case "[layer.weights]":
                            labelIdx = 4;
                            break;
                        case "[layer.biases]":
                            labelIdx = 5;
                            break;
                        case "[network.learning_rate]":
                            labelIdx = 6;
                            break;
                        case "":
                            if ((labelIdx == 2 || labelIdx == 3)
                                && activations.Length == layers.Length - 1
                                && activations.Length > 0)
                            {
                                nn = new Athenna(layers, activations);
                                labelIdx = 0;
                            }
                            break;
                        default:
                            if (labelIdx == 2) { 

                                layers[layersIdx] = Convert.ToInt32(line);
                                layersIdx++;
                            }
                            else if (labelIdx == 3) {
                                activations[activationsIdx] = Utils.StringToActivation(line);
                                activationsIdx++;
                            }
                            else if (labelIdx == 4)
                            {
                                weights.Add(Convert.ToSingle(line));
                            }
                            else if (labelIdx == 5)
                            {
                                biases.Add(Convert.ToSingle(line));
                            }
                            else if (labelIdx == 6)
                            {
                                nn.learningRate = Convert.ToSingle(line);
                            }
                            break;
                    }
                }

                int biasesIdx = 0;
                for (int i = 0; i < nn.biases.Length; i++)
                {
                    for (int j = 0; j < nn.biases[i].Length; j++)
                    {
                        nn.biases[i][j] = biases[biasesIdx];
                        biasesIdx++;
                    }
                }

                int weightsIdx = 0;
                for (int i = 0; i < nn.weights.Length; i++)
                {
                    for (int j = 0; j < nn.weights[i].Length; j++)
                    {
                        for (int k = 0; k < nn.weights[i][j].Length; k++)
                        {
                            nn.weights[i][j][k] = weights[weightsIdx];
                            weightsIdx++;
                        }
                    }
                }
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced);
            }
            return nn;
        }

        public void Save(string path)
        {
            File.Create(path).Close();
            StreamWriter writer = new StreamWriter(path, true);

            writer.WriteLine("[network.type]\nathenna.v2");
            writer.WriteLine($"\n[dims]\n{layers[0]}\n{layers[layers.Length-1]}");
            writer.WriteLine($"\n[layer.total]\n{layers.Length}");
            writer.WriteLine($"\n[layer.neurons]");
            foreach(var layer in layers)
            {
                writer.WriteLine($"{layer}");
            }
            writer.WriteLine($"\n[layer.activations]");
            foreach (var activation in activations)
            {
                writer.WriteLine($"{activation.ToString().ToLower()}");
            }
            writer.WriteLine($"\n[layer.weights]");
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        writer.WriteLine(weights[i][j][k]);
                    }
                }
            }
            writer.WriteLine($"\n[layer.biases]");
            for (int i = 0; i < biases.Length; i++)
            {
                for (int j = 0; j < biases[i].Length; j++)
                {
                    writer.WriteLine(biases[i][j]);
                }
            }
            writer.Close();
        }
    }
}

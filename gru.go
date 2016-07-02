package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	randomGRUCoefficient = 0.05
)

type GRU struct {
	Block rnn.StackedBlock
}

func DeserializeGRU(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &GRU{Block: block.(rnn.StackedBlock)}, nil
}

func (g *GRU) PrintTrainingUsage() {
	newRNNFlags(g.Name()).FlagSet.PrintDefaults()
}

func (g *GRU) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (g *GRU) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(g.Name())
	flags.FlagSet.Parse(args)
	g.makeNetwork(flags)
	TrainRNN(g.Block, g, seqs, flags)
}

func (g *GRU) Generate(length int, args []string) string {
	return GenerateRNN(g.Block, g, length, args)
}

func (g *GRU) Serialize() ([]byte, error) {
	return g.Block.Serialize()
}

func (g *GRU) SerializerType() string {
	return serializerTypeGRU
}

func (g *GRU) Name() string {
	return "gru"
}

func (g *GRU) makeNetwork(flags *rnnFlags) {
	if g.Block != nil {
		return
	}
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -0.0078125, Scale: 1 / 0.08804240367},
	}
	g.Block = append(g.Block, rnn.NewNetworkBlock(inNet, 0))
	for i := 0; i < flags.Layers; i++ {
		inputSize := ASCIICount
		if i > 0 {
			inputSize = flags.HiddenSize
		}
		layer := rnn.NewGRU(inputSize, flags.HiddenSize)
		g.Block = append(g.Block, layer)

		for i, param := range layer.Parameters() {
			if i%2 == 0 {
				for i := range param.Vector {
					param.Vector[i] = rand.NormFloat64() * randomGRUCoefficient
				}
			}
		}
	}
	outputNet := neuralnet.Network{
		&neuralnet.DropoutLayer{
			KeepProbability: flags.HiddenDropout,
			Training:        true,
		},
		&neuralnet.DenseLayer{
			InputCount:  flags.HiddenSize,
			OutputCount: ASCIICount,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outputNet.Randomize()
	outputBlock := rnn.NewNetworkBlock(outputNet, 0)
	g.Block = append(g.Block, outputBlock)
}

func (g *GRU) toggleTraining(training bool) {
	outBlock := g.Block[len(g.Block)-1].(*rnn.NetworkBlock)
	dropout := outBlock.Network()[0].(*neuralnet.DropoutLayer)
	dropout.Training = training
}

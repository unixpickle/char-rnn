package main

import (
	"fmt"
	"math"
	"os"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	irnnIdentityScale = 0.1
)

type IRNN struct {
	Block rnn.StackedBlock
}

func DeserializeIRNN(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &IRNN{Block: block}, nil
}

func (i *IRNN) PrintTrainingUsage() {
	newRNNFlags(i.Name()).FlagSet.PrintDefaults()
}

func (i *IRNN) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (i *IRNN) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(i.Name())
	flags.FlagSet.Parse(args)
	i.makeNetwork(flags)
	TrainRNN(i.Block, i, seqs, flags)
}

func (i *IRNN) Generate(length int, args []string) string {
	return GenerateRNN(i.Block, i, length, args)
}

func (i *IRNN) Serialize() ([]byte, error) {
	return i.Block.Serialize()
}

func (i *IRNN) SerializerType() string {
	return serializerTypeIRNN
}

func (i *IRNN) Name() string {
	return "irnn"
}

func (i *IRNN) makeNetwork(flags *rnnFlags) {
	if i.Block != nil {
		return
	}
	mean := 1.0 / CharCount
	stddev := math.Sqrt((CharCount-1)*math.Pow(mean, 2) + math.Pow(1-mean, 2))
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -mean, Scale: 1 / stddev},
	}
	i.Block = append(i.Block, rnn.NewNetworkBlock(inNet, 0))
	for j := 0; j < flags.Layers; j++ {
		inputSize := CharCount
		if j > 0 {
			inputSize = flags.HiddenSize
		}
		layer := rnn.NewIRNN(inputSize, flags.HiddenSize, irnnIdentityScale)
		i.Block = append(i.Block, layer)
	}
	outputNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  flags.HiddenSize,
			OutputCount: CharCount,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outputNet.Randomize()
	outputBlock := rnn.NewNetworkBlock(outputNet, 0)
	i.Block = append(i.Block, outputBlock)
}

func (i *IRNN) toggleTraining(training bool) {
	// Simply implemented to support the RNN routines.
}

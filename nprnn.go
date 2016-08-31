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

type NPRNN struct {
	Block rnn.StackedBlock
}

func DeserializeNPRNN(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &NPRNN{Block: block}, nil
}

func (n *NPRNN) PrintTrainingUsage() {
	newRNNFlags(n.Name()).FlagSet.PrintDefaults()
}

func (n *NPRNN) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (n *NPRNN) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(n.Name())
	flags.FlagSet.Parse(args)
	n.makeNetwork(flags)
	TrainRNN(n.Block, n, seqs, flags)
}

func (n *NPRNN) Generate(length int, args []string) string {
	return GenerateRNN(n.Block, n, length, args)
}

func (n *NPRNN) Serialize() ([]byte, error) {
	return n.Block.Serialize()
}

func (n *NPRNN) SerializerType() string {
	return serializerTypeNPRNN
}

func (n *NPRNN) Name() string {
	return "nprnn"
}

func (n *NPRNN) makeNetwork(flags *rnnFlags) {
	if n.Block != nil {
		return
	}
	mean := 1.0 / CharCount
	stddev := math.Sqrt((CharCount-1)*math.Pow(mean, 2) + math.Pow(1-mean, 2))
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -mean, Scale: 1 / stddev},
	}
	n.Block = append(n.Block, rnn.NewNetworkBlock(inNet, 0))
	for j := 0; j < flags.Layers; j++ {
		inputSize := CharCount
		if j > 0 {
			inputSize = flags.HiddenSize
		}
		layer := rnn.NewNPRNN(inputSize, flags.HiddenSize)
		n.Block = append(n.Block, layer)
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
	n.Block = append(n.Block, outputBlock)
}

func (n *NPRNN) toggleTraining(training bool) {
	// Simply implemented to support the RNN routines.
}

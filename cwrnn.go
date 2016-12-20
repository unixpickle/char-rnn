package main

import (
	"fmt"
	"math"
	"os"

	"github.com/unixpickle/clockwork"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

var cwrnnScales = []int{1, 2, 4, 8, 16, 32, 64}

type CWRNN struct {
	Block rnn.StackedBlock
}

func DeserializeCWRNN(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &CWRNN{Block: block}, nil
}

func (i *CWRNN) PrintTrainingUsage() {
	newRNNFlags(i.Name()).FlagSet.PrintDefaults()
}

func (i *CWRNN) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (i *CWRNN) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(i.Name())
	flags.FlagSet.Parse(args)
	i.makeNetwork(flags)
	TrainRNN(i.Block, i, seqs, flags)
}

func (i *CWRNN) Generate(length int, args []string) string {
	return GenerateRNN(i.Block, i, length, args)
}

func (i *CWRNN) Serialize() ([]byte, error) {
	return i.Block.Serialize()
}

func (i *CWRNN) SerializerType() string {
	return serializerTypeIRNN
}

func (i *CWRNN) Name() string {
	return "cwrnn"
}

func (i *CWRNN) makeNetwork(flags *rnnFlags) {
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
			inputSize = flags.HiddenSize * len(cwrnnScales)
		}
		blockSizes := make([]int, len(cwrnnScales))
		for i := range blockSizes {
			blockSizes[i] = flags.HiddenSize
		}
		layer := clockwork.NewBlockFC(inputSize, cwrnnScales, blockSizes)
		i.Block = append(i.Block, layer)
	}
	i.Block = append(i.Block, rnn.NewNetworkBlock(neuralnet.Network{
		neuralnet.NewDenseLayer(flags.HiddenSize*len(cwrnnScales), CharCount),
		&neuralnet.LogSoftmaxLayer{},
	}, 0))
}

func (i *CWRNN) toggleTraining(training bool) {
	// Simply implemented to support the RNN routines.
}

package main

import (
	"fmt"
	"math"
	"os"

	"github.com/unixpickle/hebbnet"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

type HebbNet struct {
	Block rnn.StackedBlock
}

func DeserializeHebbNet(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &HebbNet{Block: block}, nil
}

func (h *HebbNet) PrintTrainingUsage() {
	newRNNFlags(h.Name()).FlagSet.PrintDefaults()
}

func (h *HebbNet) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (h *HebbNet) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(h.Name())
	flags.FlagSet.Parse(args)
	h.makeNetwork(flags)
	TrainRNN(h.Block, h, seqs, flags)
}

func (h *HebbNet) Generate(length int, args []string) string {
	return GenerateRNN(h.Block, h, length, args)
}

func (h *HebbNet) Serialize() ([]byte, error) {
	return h.Block.Serialize()
}

func (h *HebbNet) SerializerType() string {
	return serializerTypeHebbNet
}

func (h *HebbNet) Name() string {
	return "hebbnet"
}

func (h *HebbNet) makeNetwork(flags *rnnFlags) {
	if h.Block != nil {
		return
	}
	mean := 1.0 / CharCount
	stddev := math.Sqrt((CharCount-1)*math.Pow(mean, 2) + math.Pow(1-mean, 2))
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -mean, Scale: 1 / stddev},
	}
	h.Block = append(h.Block, rnn.NewNetworkBlock(inNet, 0))
	for j := 0; j < flags.Layers; j++ {
		inputSize := CharCount
		if j > 0 {
			inputSize = flags.HiddenSize
		}
		layer := hebbnet.NewDenseLayer(inputSize, flags.HiddenSize)
		layer.UseActivation = true
		h.Block = append(h.Block, layer)
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
	h.Block = append(h.Block, outputBlock)
}

func (h *HebbNet) toggleTraining(training bool) {
	// Simply implemented to support the RNN routines.
}

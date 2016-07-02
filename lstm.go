package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	randomLSTMCoefficient = 0.05
)

type LSTM struct {
	Block rnn.StackedBlock
}

func DeserializeLSTM(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeStackedBlock(d)
	if err != nil {
		return nil, err
	}
	return &LSTM{Block: block.(rnn.StackedBlock)}, nil
}

func (l *LSTM) PrintTrainingUsage() {
	newRNNFlags(l.Name()).FlagSet.PrintDefaults()
}

func (l *LSTM) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (l *LSTM) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(l.Name())
	flags.FlagSet.Parse(args)
	l.makeNetwork(flags)
	TrainRNN(l.Block, l, seqs, flags)
}

func (l *LSTM) Generate(length int, args []string) string {
	return GenerateRNN(l.Block, l, length, args)
}

func (l *LSTM) Serialize() ([]byte, error) {
	return l.Block.Serialize()
}

func (l *LSTM) SerializerType() string {
	return serializerTypeLSTM
}

func (l *LSTM) Name() string {
	return "lstm"
}

func (l *LSTM) makeNetwork(flags *rnnFlags) {
	if l.Block != nil {
		return
	}
	mean := 1.0 / CharCount
	stddev := math.Sqrt((CharCount-1)*math.Pow(mean, 2) + math.Pow(1-mean, 2))
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -mean, Scale: 1 / stddev},
	}
	l.Block = append(l.Block, rnn.NewNetworkBlock(inNet, 0))
	for i := 0; i < flags.Layers; i++ {
		inputSize := CharCount
		if i > 0 {
			inputSize = flags.HiddenSize
		}
		layer := rnn.NewLSTM(inputSize, flags.HiddenSize)
		l.Block = append(l.Block, layer)

		for i, param := range layer.Parameters() {
			if i%2 == 0 {
				for i := range param.Vector {
					param.Vector[i] = rand.NormFloat64() * randomLSTMCoefficient
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
			OutputCount: CharCount,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outputNet.Randomize()
	outputBlock := rnn.NewNetworkBlock(outputNet, 0)
	l.Block = append(l.Block, outputBlock)
}

func (l *LSTM) toggleTraining(training bool) {
	outBlock := l.Block[len(l.Block)-1].(*rnn.NetworkBlock)
	dropout := outBlock.Network()[0].(*neuralnet.DropoutLayer)
	dropout.Training = training
}

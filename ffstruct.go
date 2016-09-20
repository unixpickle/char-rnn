package main

import (
	"fmt"
	"math"
	"os"

	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

type FFStruct struct {
	SeqFunc *neuralstruct.SeqFunc
}

func DeserializeFFStruct(d []byte) (serializer.Serializer, error) {
	sf, err := neuralstruct.DeserializeSeqFunc(d)
	if err != nil {
		return nil, err
	}
	return &FFStruct{SeqFunc: sf}, nil
}

func (f *FFStruct) PrintTrainingUsage() {
	newStructRNNFlags(f.Name()).FlagSet.PrintDefaults()
}

func (f *FFStruct) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (f *FFStruct) Train(seqs sgd.SampleSet, args []string) {
	flags := newStructRNNFlags(f.Name())
	flags.FlagSet.Parse(args)
	f.makeNetwork(flags)
	TrainStructRNN(f.SeqFunc, f, seqs, flags)
}

func (f *FFStruct) Generate(length int, args []string) string {
	return GenerateStructRNN(f.SeqFunc, f, length, args)
}

func (f *FFStruct) Serialize() ([]byte, error) {
	return f.SeqFunc.Serialize()
}

func (f *FFStruct) SerializerType() string {
	return serializerTypeFFStruct
}

func (f *FFStruct) Name() string {
	return "ffstruct"
}

func (f *FFStruct) makeNetwork(flags *structRNNFlags) {
	if f.SeqFunc != nil {
		return
	}
	mean := 1.0 / CharCount
	stddev := math.Sqrt((CharCount-1)*math.Pow(mean, 2) + math.Pow(1-mean, 2))
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -mean, Scale: 1 / stddev},
	}

	structure := f.makeStruct()

	var block rnn.StackedBlock
	block = append(block, rnn.NewNetworkBlock(inNet, 0))
	for j := 0; j < flags.Layers; j++ {
		inputSize := CharCount + structure.DataSize()
		if j > 0 {
			inputSize = flags.HiddenSize
		}
		net := neuralnet.Network{
			&neuralnet.DenseLayer{
				InputCount:  inputSize,
				OutputCount: flags.HiddenSize,
			},
			&neuralnet.HyperbolicTangent{},
		}
		net.Randomize()
		block = append(block, rnn.NewNetworkBlock(net, 0))
	}
	outputNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  flags.HiddenSize,
			OutputCount: CharCount + structure.ControlSize(),
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outputNet.Randomize()
	block = append(block, rnn.NewNetworkBlock(outputNet, 0))

	f.SeqFunc = &neuralstruct.SeqFunc{
		Block:  block,
		Struct: structure,
	}
}

func (f *FFStruct) makeStruct() neuralstruct.RStruct {
	return neuralstruct.RAggregate{
		&neuralstruct.Queue{VectorSize: 6},
		&neuralstruct.Queue{VectorSize: 6},
		&neuralstruct.Queue{VectorSize: 6},
		&neuralstruct.Stack{VectorSize: 6, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 6, NoReplace: true},
		&neuralstruct.Stack{VectorSize: 6, NoReplace: true},
	}
}

func (f *FFStruct) toggleTraining(training bool) {
	// Simply implemented to support the RNN routines.
}

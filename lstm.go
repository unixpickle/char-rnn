package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const asciiCount = 128

const (
	defaultLSTMHiddenSize    = 512
	defaultLSTMLayerCount    = 2
	defaultLSTMStepSize      = 0.001
	defaultLSTMInDropout     = 0.9
	defaultLSTMHiddenDropout = 0.5
	defaultLSTMBatchSize     = 100

	validateBatchSize = 20
)

type LSTM struct {
	Block rnn.StackedBlock
}

func DeserializeLSTM(d []byte) (serializer.Serializer, error) {
	block, err := rnn.DeserializeLSTM(d)
	if err != nil {
		return nil, err
	}
	return &LSTM{Block: block.(rnn.StackedBlock)}, nil
}

func (l *LSTM) PrintTrainingUsage() {
	newLSTMFlags().FlagSet.PrintDefaults()
}

func (l *LSTM) PrintGenerationUsage() {
	fmt.Fprintln(os.Stderr, "No generation arguments.")
}

func (l *LSTM) Train(seqs neuralnet.SampleSet, args []string) {
	flags := newLSTMFlags()
	flags.FlagSet.Parse(args)
	l.makeNetwork(flags)
	costFunc := neuralnet.DotCost{}
	gradienter := &neuralnet.RMSProp{
		Gradienter: &rnn.FullRGradienter{
			Learner:  l.Block,
			CostFunc: costFunc,
		},
	}
	var epoch int
	neuralnet.SGDInteractive(gradienter, seqs, flags.StepSize, flags.BatchSize, func() bool {
		runner := &rnn.Runner{Block: l.Block}
		cost := runner.TotalCost(validateBatchSize, seqs, costFunc)
		log.Printf("Epoch %d: cost=%f", epoch, cost)
		epoch++
		return true
	})
}

func (l *LSTM) Generate(length int, args []string) string {
	var res bytes.Buffer
	r := &rnn.Runner{Block: l.Block}
	input := make(linalg.Vector, asciiCount)
	for i := 0; i < length; i++ {
		output := r.StepTime(input)
		idx := chooseLogIndex(output)
		input := make(linalg.Vector, asciiCount)
		input[idx] = 1
		res.WriteByte(byte(idx))
	}
	return res.String()
}

func (l *LSTM) Serialize() ([]byte, error) {
	return l.Block.Serialize()
}

func (l *LSTM) SerializerType() string {
	return serializerTypeLSTM
}

func (l *LSTM) makeNetwork(flags *lstmFlags) {
	if l.Block == nil {
		l.Block = rnn.StackedBlock{
			rnn.NewNetworkBlock(neuralnet.Network{
				&neuralnet.DropoutLayer{
					KeepProbability: flags.InDropout,
					Training:        true,
				},
			}, 0),
		}
		for i := 0; i < flags.Layers; i++ {
			inputSize := asciiCount
			if i > 0 {
				inputSize = flags.HiddenSize
			}
			layer := rnn.NewLSTM(inputSize, flags.HiddenSize)
			l.Block = append(l.Block, layer)
		}
		outputBlock := rnn.NewNetworkBlock(neuralnet.Network{
			neuralnet.Sigmoid{},
			&neuralnet.DropoutLayer{
				KeepProbability: flags.HiddenDropout,
				Training:        true,
			},
			&neuralnet.DenseLayer{
				InputCount:  flags.HiddenSize,
				OutputCount: asciiCount,
			},
			&neuralnet.LogSoftmaxLayer{},
		}, 0)
		l.Block = append(l.Block, outputBlock)
	}
}

type lstmFlags struct {
	FlagSet *flag.FlagSet

	StepSize  float64
	BatchSize int

	HiddenSize    int
	Layers        int
	InDropout     float64
	HiddenDropout float64
}

func newLSTMFlags() *lstmFlags {
	res := &lstmFlags{}
	res.FlagSet = flag.NewFlagSet("lstm", flag.ExitOnError)
	res.FlagSet.Float64Var(&res.StepSize, "stepsize", defaultLSTMStepSize, "step size")
	res.FlagSet.IntVar(&res.BatchSize, "batch", defaultLSTMBatchSize, "mini-batch size")
	res.FlagSet.IntVar(&res.HiddenSize, "hidden", defaultLSTMHiddenSize, "hidden layer size")
	res.FlagSet.IntVar(&res.Layers, "layers", defaultLSTMLayerCount, "number of layers")
	res.FlagSet.Float64Var(&res.InDropout, "dropin", defaultLSTMInDropout, "input dropout")
	res.FlagSet.Float64Var(&res.HiddenDropout, "drophid", defaultLSTMHiddenDropout,
		"hidden dropout")
	return res
}

func chooseLogIndex(logProbs linalg.Vector) int {
	n := rand.Float64()
	var sum float64
	for i, x := range logProbs {
		sum += math.Exp(x)
		if sum > n {
			return i
		}
	}
	return len(logProbs) - 1
}

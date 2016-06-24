package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	defaultLSTMHiddenSize    = 300
	defaultLSTMLayerCount    = 1
	defaultLSTMStepSize      = 0.005
	defaultLSTMHiddenDropout = 0.5
	defaultLSTMBatchSize     = 100
	defaultLSTMHeadSize      = 50
	defaultLSTMTailSize      = 20

	defaultLSTMEquilibrationMemory = 0.9

	randomCoefficient = 0.05

	validateBatchSize = 20
	maxLanes          = 21
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
	newLSTMFlags().FlagSet.PrintDefaults()
}

func (l *LSTM) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (l *LSTM) Train(seqs neuralnet.SampleSet, args []string) {
	flags := newLSTMFlags()
	flags.FlagSet.Parse(args)
	l.makeNetwork(flags)
	costFunc := neuralnet.DotCost{}
	gradienter := &neuralnet.Equilibration{
		RGradienter: &rnn.TruncatedBPTT{
			Learner:  l.Block,
			CostFunc: costFunc,
			MaxLanes: maxLanes,
			HeadSize: flags.HeadSize,
			TailSize: flags.TailSize,
		},
		Learner: l.Block,
		Memory:  flags.EquilibrationMemory,
		Damping: 0.01,
	}

	l.toggleTraining(true)
	log.Println("Training LSTM on", seqs.Len(), "samples...")

	var epoch int
	neuralnet.SGDInteractive(gradienter, seqs, flags.StepSize, flags.BatchSize, func() bool {
		l.toggleTraining(false)
		defer l.toggleTraining(true)

		runner := &rnn.Runner{Block: l.Block}
		cost := runner.TotalCost(validateBatchSize, seqs, costFunc)
		log.Printf("Epoch %d: cost=%f", epoch, cost)

		epoch++
		return true
	})
}

func (l *LSTM) Generate(length int, args []string) string {
	temp := 1.0
	if len(args) == 1 {
		var err error
		temp, err = strconv.ParseFloat(args[0], 64)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid temperature:", args[0])
			os.Exit(1)
		}
	}

	l.toggleTraining(false)

	var res bytes.Buffer
	r := &rnn.Runner{Block: l.Block}
	input := make(linalg.Vector, ASCIICount)
	input[0] = 1
	for i := 0; i < length; i++ {
		output := r.StepTime(input)
		idx := chooseLogIndex(output, temp)
		input = make(linalg.Vector, ASCIICount)
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

func (l *LSTM) Name() string {
	return "lstm"
}

func (l *LSTM) makeNetwork(flags *lstmFlags) {
	if l.Block != nil {
		return
	}
	inNet := neuralnet.Network{
		&neuralnet.RescaleLayer{Bias: -0.0078125, Scale: 1 / 0.08804240367},
	}
	l.Block = append(l.Block, rnn.NewNetworkBlock(inNet, 0))
	for i := 0; i < flags.Layers; i++ {
		inputSize := ASCIICount
		if i > 0 {
			inputSize = flags.HiddenSize
		}
		layer := rnn.NewLSTM(inputSize, flags.HiddenSize)
		l.Block = append(l.Block, layer)

		for i, param := range layer.Parameters() {
			if i%2 == 0 {
				for i := range param.Vector {
					param.Vector[i] = rand.NormFloat64() * randomCoefficient
				}
			}
		}
		inputBiases := layer.Parameters()[3]
		for i := range inputBiases.Vector {
			inputBiases.Vector[i] = -1
		}
		outputBiases := layer.Parameters()[7]
		for i := range outputBiases.Vector {
			outputBiases.Vector[i] = -2
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
	l.Block = append(l.Block, outputBlock)
}

func (l *LSTM) toggleTraining(training bool) {
	outBlock := l.Block[len(l.Block)-1].(*rnn.NetworkBlock)
	dropout := outBlock.Network()[0].(*neuralnet.DropoutLayer)
	dropout.Training = training
}

type lstmFlags struct {
	FlagSet *flag.FlagSet

	StepSize  float64
	BatchSize int

	HiddenSize    int
	Layers        int
	HiddenDropout float64

	HeadSize int
	TailSize int

	EquilibrationMemory float64
}

func newLSTMFlags() *lstmFlags {
	res := &lstmFlags{}
	res.FlagSet = flag.NewFlagSet("lstm", flag.ExitOnError)
	res.FlagSet.Float64Var(&res.StepSize, "stepsize", defaultLSTMStepSize, "step size")
	res.FlagSet.IntVar(&res.BatchSize, "batch", defaultLSTMBatchSize, "mini-batch size")
	res.FlagSet.IntVar(&res.HiddenSize, "hidden", defaultLSTMHiddenSize, "hidden layer size")
	res.FlagSet.IntVar(&res.Layers, "layers", defaultLSTMLayerCount, "number of layers")
	res.FlagSet.Float64Var(&res.HiddenDropout, "dropout", defaultLSTMHiddenDropout,
		"hidden dropout (1=no dropout)")
	res.FlagSet.IntVar(&res.HeadSize, "headsize", defaultLSTMHeadSize, "TBPTT head size")
	res.FlagSet.IntVar(&res.TailSize, "tailsize", defaultLSTMTailSize, "TBPTT tail size")
	res.FlagSet.Float64Var(&res.EquilibrationMemory, "eqmem", defaultLSTMEquilibrationMemory,
		"equilibration memory")
	return res
}

func chooseLogIndex(logProbs linalg.Vector, temp float64) int {
	n := rand.Float64()
	var sum float64
	for _, x := range logProbs {
		sum += math.Exp(x / temp)
	}
	var curSum float64
	for i, x := range logProbs {
		curSum += math.Exp(x / temp)
		if curSum/sum > n {
			return i
		}
	}
	return len(logProbs) - 1
}

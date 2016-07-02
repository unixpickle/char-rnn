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
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	defaultRNNHiddenSize    = 300
	defaultRNNLayerCount    = 1
	defaultRNNStepSize      = 0.005
	defaultRNNHiddenDropout = 0.5
	defaultRNNBatchSize     = 100
	defaultRNNHeadSize      = 50
	defaultRNNTailSize      = 20

	validateBatchSize = 20
	maxLanes          = 21
)

type trainToggler interface {
	toggleTraining(bool)
}

func TrainRNN(b rnn.BlockLearner, t trainToggler, seqs sgd.SampleSet, flags *rnnFlags) {
	costFunc := neuralnet.DotCost{}
	gradienter := &sgd.Adam{
		Gradienter: &rnn.TruncatedBPTT{
			Learner:  b,
			CostFunc: costFunc,
			MaxLanes: maxLanes,
			HeadSize: flags.HeadSize,
			TailSize: flags.TailSize,
		},
	}

	t.toggleTraining(true)
	log.Println("Training model on", seqs.Len(), "samples...")

	var epoch int
	sgd.SGDInteractive(gradienter, seqs, flags.StepSize, flags.BatchSize, func() bool {
		t.toggleTraining(false)
		defer t.toggleTraining(true)

		runner := &rnn.Runner{Block: b}
		cost := runner.TotalCost(validateBatchSize, seqs, costFunc)
		log.Printf("Epoch %d: cost=%f", epoch, cost)

		epoch++
		return true
	})
}

func GenerateRNN(b rnn.Block, t trainToggler, length int, args []string) string {
	temp := 1.0
	if len(args) == 1 {
		var err error
		temp, err = strconv.ParseFloat(args[0], 64)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid temperature:", args[0])
			os.Exit(1)
		}
	}

	t.toggleTraining(false)

	var res bytes.Buffer
	runner := &rnn.Runner{Block: b}
	input := make(linalg.Vector, ASCIICount)
	input[0] = 1
	for i := 0; i < length; i++ {
		output := runner.StepTime(input)
		idx := chooseLogIndex(output, temp)
		input = make(linalg.Vector, ASCIICount)
		input[idx] = 1
		res.WriteByte(byte(idx))
	}
	return res.String()
}

type rnnFlags struct {
	FlagSet *flag.FlagSet

	StepSize  float64
	BatchSize int

	HiddenSize    int
	Layers        int
	HiddenDropout float64

	HeadSize int
	TailSize int
}

func newRNNFlags(cmdName string) *rnnFlags {
	res := &rnnFlags{}
	res.FlagSet = flag.NewFlagSet(cmdName, flag.ExitOnError)
	res.FlagSet.Float64Var(&res.StepSize, "stepsize", defaultRNNStepSize, "step size")
	res.FlagSet.IntVar(&res.BatchSize, "batch", defaultRNNBatchSize, "mini-batch size")
	res.FlagSet.IntVar(&res.HiddenSize, "hidden", defaultRNNHiddenSize, "hidden layer size")
	res.FlagSet.IntVar(&res.Layers, "layers", defaultRNNLayerCount, "number of layers")
	res.FlagSet.Float64Var(&res.HiddenDropout, "dropout", defaultRNNHiddenDropout,
		"hidden dropout (1=no dropout)")
	res.FlagSet.IntVar(&res.HeadSize, "headsize", defaultRNNHeadSize, "TBPTT head size")
	res.FlagSet.IntVar(&res.TailSize, "tailsize", defaultRNNTailSize, "TBPTT tail size")
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

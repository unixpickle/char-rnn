package main

import (
	"flag"
	"log"

	"github.com/unixpickle/neuralstruct"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

func TrainStructRNN(s *neuralstruct.SeqFunc, t trainToggler, seqs sgd.SampleSet,
	flags *structRNNFlags) {
	costFunc := neuralnet.DotCost{}
	gradienter := &sgd.Adam{
		Gradienter: &seqtoseq.SeqFuncGradienter{
			SeqFunc:  s,
			Learner:  s,
			CostFunc: costFunc,
			MaxLanes: maxLanes,
		},
	}

	t.toggleTraining(true)
	log.Println("Training model on", seqs.Len(), "samples...")

	var epoch int
	sgd.SGDInteractive(gradienter, seqs, flags.StepSize, flags.BatchSize, func() bool {
		t.toggleTraining(false)
		defer t.toggleTraining(true)

		cost := seqtoseq.TotalCostSeqFunc(s, validateBatchSize, seqs, costFunc)
		log.Printf("Epoch %d: cost=%f", epoch, cost)

		epoch++
		return true
	})
}

func GenerateStructRNN(s *neuralstruct.SeqFunc, t trainToggler, length int,
	args []string) string {
	step := &neuralstruct.Runner{Block: s.Block, Struct: s.Struct}
	return generateTimeStepper(step, t, length, args)
}

type structRNNFlags struct {
	FlagSet *flag.FlagSet

	StepSize  float64
	BatchSize int

	HiddenSize    int
	Layers        int
	HiddenDropout float64
}

func newStructRNNFlags(cmdName string) *structRNNFlags {
	res := &structRNNFlags{}
	res.FlagSet = flag.NewFlagSet(cmdName, flag.ExitOnError)
	res.FlagSet.Float64Var(&res.StepSize, "stepsize", defaultRNNStepSize, "step size")
	res.FlagSet.IntVar(&res.BatchSize, "batch", defaultRNNBatchSize, "mini-batch size")
	res.FlagSet.IntVar(&res.HiddenSize, "hidden", defaultRNNHiddenSize, "hidden layer size")
	res.FlagSet.IntVar(&res.Layers, "layers", defaultRNNLayerCount, "number of layers")
	res.FlagSet.Float64Var(&res.HiddenDropout, "dropout", defaultRNNHiddenDropout,
		"hidden dropout (1=no dropout)")
	return res
}

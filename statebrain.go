package main

import (
	"fmt"
	"os"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/statebrain"
)

type StateBrain struct {
	Block *statebrain.Block
}

func DeserializeStateBrain(d []byte) (serializer.Serializer, error) {
	block, err := statebrain.DeserializeBlock(d)
	if err != nil {
		return nil, err
	}
	return &StateBrain{Block: block}, nil
}

func (s *StateBrain) PrintTrainingUsage() {
	newRNNFlags(s.Name()).FlagSet.PrintDefaults()
}

func (s *StateBrain) PrintGenerateUsage() {
	fmt.Fprintln(os.Stderr, "Optional value: [temperature]")
}

func (s *StateBrain) Train(seqs sgd.SampleSet, args []string) {
	flags := newRNNFlags(s.Name())
	flags.FlagSet.Parse(args)
	s.makeNetwork(flags)
	TrainRNN(s.Block, s, seqs, flags)
}

func (s *StateBrain) Generate(length int, args []string) string {
	return GenerateRNN(s.Block, s, length, args)
}

func (s *StateBrain) Serialize() ([]byte, error) {
	return s.Block.Serialize()
}

func (s *StateBrain) SerializerType() string {
	return serializerTypeStateBrain
}

func (s *StateBrain) Name() string {
	return "statebrain"
}

func (s *StateBrain) makeNetwork(flags *rnnFlags) {
	if s.Block != nil {
		return
	}
	s.Block = statebrain.NewBlock(CharCount, flags.HiddenSize)
}

func (s *StateBrain) toggleTraining(training bool) {
	// Simply implemented to support the RNN routines.
}

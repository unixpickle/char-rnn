package main

import (
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

type Model interface {
	serializer.Serializer

	PrintTrainingUsage()
	PrintGenerateUsage()

	// Train trains the model using a SampleSet full of
	// rnn.Sequence instances.
	Train(samples neuralnet.SampleSet, arguments []string)

	// Generate generates a new string of text using the
	// model.
	Generate(length int, arguments []string) string
}

package charrnn

import (
	"flag"

	"github.com/unixpickle/serializer"
)

// A Model is a trainable language model for predicting
// characters in a string.
type Model interface {
	serializer.Serializer

	Name() string

	TrainingFlags() *flag.FlagSet
	GenerationFlags() *flag.FlagSet

	Train(samples SampleList)
	Generate()
}

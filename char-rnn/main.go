package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	charrnn "github.com/unixpickle/char-rnn"
	"github.com/unixpickle/serializer"
)

var Models = []charrnn.Model{&charrnn.LSTM{}, &charrnn.Markov{}, &charrnn.HMM{}}

const OutputPermissions = 0755

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) < 2 {
		dieUsage()
	}
	subCmd := os.Args[1]
	switch subCmd {
	case "train":
		trainCommand()
	case "gen":
		genCommand()
	case "help":
		helpCommand()
	default:
		dieUsage()
	}
}

func trainCommand() {
	if len(os.Args) < 5 {
		dieUsage()
	}

	modelFile := os.Args[3]

	model := modelForName(os.Args[2])
	samples := charrnn.ReadSampleList(os.Args[4])
	modelData, err := ioutil.ReadFile(modelFile)

	if err == nil {
		x, desErr := serializer.DeserializeWithType(modelData)
		if desErr != nil {
			fmt.Fprintln(os.Stderr, "Failed to deserialize model:", desErr)
			os.Exit(1)
		}
		var ok bool
		model, ok = x.(charrnn.Model)
		if !ok {
			fmt.Fprintf(os.Stderr, "Loaded type was not a model but a %T\n", x)
			os.Exit(1)
		}
		log.Println("Loaded model from file.")
	} else {
		log.Println("Created new model.")
	}

	model.TrainingFlags().Parse(os.Args[5:])
	model.Train(samples)

	encoded, err := serializer.SerializeWithType(model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize model:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(modelFile, encoded, OutputPermissions); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}

func genCommand() {
	if len(os.Args) < 3 {
		dieUsage()
	}

	modelData, err := ioutil.ReadFile(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read model:", err)
		os.Exit(1)
	}

	x, err := serializer.DeserializeWithType(modelData)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	model, ok := x.(charrnn.Model)
	if !ok {
		fmt.Fprintf(os.Stderr, "Loaded type was not a model but a %T\n", x)
		os.Exit(1)
	}

	model.GenerationFlags().Parse(os.Args[3:])
	model.Generate()
}

func helpCommand() {
	if len(os.Args) != 3 {
		dieUsage()
	}
	m := modelForName(os.Args[2])
	fmt.Fprintf(os.Stderr, "Usage for training:\n\n")
	m.TrainingFlags().PrintDefaults()
	fmt.Fprintf(os.Stderr, "\nUsage for generation:\n\n")
	m.GenerationFlags().PrintDefaults()
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: char-rnn train <model> <rnn-file> <sample dir> [args]\n"+
		"       char-rnn gen <rnn-file> [args]\n"+
		"       char-rnn help <model>\n\n"+
		"Available models:")
	for _, m := range Models {
		fmt.Fprintln(os.Stderr, " "+m.Name())
	}
	fmt.Fprintln(os.Stderr, "\nEnvironment variables:")
	fmt.Fprintf(os.Stderr, " TEXT_CHUNK_SIZE      chars per sample (default %d)\n",
		charrnn.TextChunkSize)
	fmt.Fprintln(os.Stderr, " TEXT_CHUNK_HEAD_ONLY only use heads of samples")
	fmt.Fprintln(os.Stderr)
	os.Exit(1)
}

func modelForName(name string) charrnn.Model {
	for _, m := range Models {
		if m.Name() == name {
			return m
		}
	}
	fmt.Fprintln(os.Stderr, "no such model: "+os.Args[2])
	dieUsage()
	return nil
}

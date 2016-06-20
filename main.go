package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"

	"github.com/unixpickle/serializer"
)

var Models = []Model{&LSTM{}}

const OutputPermissions = 0755

func main() {
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

	modelFile := os.Args[2]

	model := modelForName(modelFile)
	samples := ReadSequences(os.Args[4])
	modelData, err := ioutil.ReadFile(os.Args[3])

	if err == nil {
		x, err := serializer.DeserializeWithType(modelData)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to deserialize model:", err)
			os.Exit(1)
		}
		var ok bool
		model, ok = x.(Model)
		if !ok {
			fmt.Fprintf(os.Stderr, "Loaded type was not a model but a %T\n", x)
			os.Exit(1)
		}
		log.Println("Loaded model from file.")
	} else {
		log.Println("Created new model.")
	}

	model.Train(samples, os.Args[5:])

	encoded, err := model.Serialize()
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
	if len(os.Args) < 4 {
		dieUsage()
	}

	size, err := strconv.Atoi(os.Args[3])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid size:", os.Args[3])
		os.Exit(1)
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

	model, ok := x.(Model)
	if !ok {
		fmt.Fprintf(os.Stderr, "Loaded type was not a model but a %T\n", x)
		os.Exit(1)
	}

	fmt.Println(model.Generate(size, os.Args[4:]))
}

func helpCommand() {
	if len(os.Args) != 3 {
		dieUsage()
	}
	m := modelForName(os.Args[2])
	fmt.Fprintf(os.Stderr, "Usage for training:\n\n")
	m.PrintTrainingUsage()
	fmt.Fprintf(os.Stderr, "\nUsage for generation:\n\n")
	m.PrintGenerateUsage()
}

func dieUsage() {
	fmt.Fprintln(os.Stderr, "Usage: char-rnn train <model> <rnn-file> <sample dir> [args]\n"+
		"                gen <rnn-file> <chars> [args]\n"+
		"                help <model>\n\n"+
		"Available models:")
	for _, m := range Models {
		fmt.Fprintln(os.Stderr, " "+m.Name())
	}
	os.Exit(1)
}

func modelForName(name string) Model {
	for _, m := range Models {
		if m.Name() == name {
			return m
		}
	}
	fmt.Fprintln(os.Stderr, "no such model: "+os.Args[2])
	dieUsage()
	return nil
}

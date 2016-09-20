package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	TextChunkSize = 1 << 10
	CharCount     = 256
)

type SampleSet [][]byte

func ReadSampleSet(dir string) SampleSet {
	contents, err := ioutil.ReadDir(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var result SampleSet

	chunkSize := TextChunkSize
	if csVar := os.Getenv("TEXT_CHUNK_SIZE"); csVar != "" {
		chunkSize, err = strconv.Atoi(csVar)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid TEXT_CHUNK_SIZE value:", csVar)
			os.Exit(1)
		}
	}

	for _, item := range contents {
		if strings.HasPrefix(item.Name(), ".") {
			continue
		}
		p := filepath.Join(dir, item.Name())
		textContents, err := ioutil.ReadFile(p)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		for i := 0; i < len(textContents); i += chunkSize {
			bs := chunkSize
			if bs > len(textContents)-i {
				bs = len(textContents) - i
			}
			result = append(result, textContents[i:i+bs])
		}
	}

	return result
}

func (s SampleSet) Len() int {
	return len(s)
}

func (s SampleSet) Copy() sgd.SampleSet {
	res := make(SampleSet, len(s))
	copy(res, s)
	return res
}

func (s SampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SampleSet) GetSample(idx int) interface{} {
	return seqForChunk(s[idx])
}

func (s SampleSet) Subset(start, end int) sgd.SampleSet {
	return s[start:end]
}

func seqForChunk(chunk []byte) seqtoseq.Sample {
	var res seqtoseq.Sample
	for i, x := range chunk {
		res.Outputs = append(res.Outputs, oneHotAscii(x))
		if i == 0 {
			res.Inputs = append(res.Inputs, oneHotAscii(0))
		} else {
			res.Inputs = append(res.Inputs, oneHotAscii(chunk[i-1]))
		}
	}
	return res
}

func oneHotAscii(b byte) linalg.Vector {
	res := make(linalg.Vector, CharCount)
	res[int(b)] = 1
	return res
}

package charrnn

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

const (
	TextChunkSize = 1 << 10
	CharCount     = 256
)

type SampleList [][]byte

func ReadSampleList(dir string) SampleList {
	contents, err := ioutil.ReadDir(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var result SampleList

	chunkSize := TextChunkSize
	headOnly := false
	if csVar := os.Getenv("TEXT_CHUNK_SIZE"); csVar != "" {
		chunkSize, err = strconv.Atoi(csVar)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid TEXT_CHUNK_SIZE value:", csVar)
			os.Exit(1)
		}
	}
	if os.Getenv("TEXT_CHUNK_HEAD_ONLY") != "" {
		headOnly = true
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
			if headOnly {
				break
			}
		}
	}

	return result
}

func (s SampleList) Len() int {
	return len(s)
}

func (s SampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SampleList) Slice(start, end int) anysgd.SampleList {
	return append(SampleList{}, s[start:end]...)
}

func (s SampleList) LenAt(idx int) int {
	return len(s[idx])
}

func (s SampleList) GetSample(idx int) *anys2s.Sample {
	return seqForChunk(s[idx])
}

func seqForChunk(chunk []byte) *anys2s.Sample {
	var res anys2s.Sample
	for i, x := range chunk {
		res.Output = append(res.Output, oneHotAscii(x))
		if i == 0 {
			res.Input = append(res.Input, oneHotAscii(0))
		} else {
			res.Input = append(res.Input, oneHotAscii(chunk[i-1]))
		}
	}
	return &res
}

func oneHotAscii(b byte) anyvec.Vector {
	res := make([]float32, CharCount)
	res[int(b)] = 1
	return anyvec32.MakeVectorData(res)
}

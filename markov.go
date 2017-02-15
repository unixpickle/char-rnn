package charrnn

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/serializer"
)

func init() {
	var m Markov
	serializer.RegisterTypedDeserializer(m.SerializerType(), DeserializeMarkov)
}

const entropySoftener = 1e-5

// Markov is a Model for a character-level Markov chain.
type Markov struct {
	Table   map[string]map[byte]float64
	History int

	Validation float64 `json:"-"`
}

func DeserializeMarkov(d []byte) (*Markov, error) {
	var res Markov
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (m *Markov) Name() string {
	return "markov"
}

func (m *Markov) TrainingFlags() *flag.FlagSet {
	f := flag.NewFlagSet("markov", flag.ExitOnError)
	f.IntVar(&m.History, "history", 3, "character history size")
	f.Float64Var(&m.Validation, "validation", 0.1, "validation fraction")
	return f
}

func (m *Markov) GenerationFlags() *flag.FlagSet {
	return flag.NewFlagSet("markov", flag.ExitOnError)
}

func (m *Markov) Train(s SampleList) {
	validation, training := anysgd.HashSplit(s, m.Validation)
	log.Printf("Training: %d samples (%d bytes)", training.Len(),
		training.(SampleList).Bytes())
	log.Printf("Validation: %d samples (%d bytes)", validation.Len(),
		validation.(SampleList).Bytes())

	m.Table = map[string]map[byte]float64{}
	totals := map[string]float64{}

	log.Println("Producing chain...")
	for _, sample := range training.(SampleList) {
		stateBytes := []byte{}
		for _, ch := range append(append([]byte{}, sample...), 0) {
			stateStr := string(stateBytes)
			if m.Table[stateStr] == nil {
				m.Table[stateStr] = map[byte]float64{}
			}
			m.Table[stateStr][ch]++
			totals[stateStr]++

			stateBytes = m.appendState(stateBytes, ch)
		}
	}

	log.Println("Normalizing chain...")
	for state, total := range totals {
		for k, v := range m.Table[state] {
			m.Table[state][k] = v / total
		}
	}

	log.Println("Computing cross-entropy...")

	log.Println("Training entropy:", m.averageEntropy(training.(SampleList)))
	log.Println("Validation entropy:", m.averageEntropy(validation.(SampleList)))
}

func (m *Markov) Generate() {
	state := []byte{}
	for {
		next := m.selectRandom(state)
		if next == 0 {
			break
		}
		fmt.Print(string(next))
		state = m.appendState(state, next)
	}
	fmt.Println()
}

func (m *Markov) SerializerType() string {
	return "github.com/unixpickle/char-rnn.Markov"
}

func (m *Markov) Serialize() ([]byte, error) {
	return json.Marshal(m)
}

func (m *Markov) averageEntropy(s SampleList) float64 {
	var totalEntropy float64
	var charCount float64
	for _, sample := range s {
		totalEntropy += m.sampleEntropy(sample)
		charCount += float64(len(sample))
	}
	return totalEntropy / charCount
}

func (m *Markov) sampleEntropy(sample []byte) float64 {
	entropy := 0.0
	state := []byte{}
	for _, b := range sample {
		p := m.Table[string(state)][b]
		if p == 0 {
			p = entropySoftener
		}
		entropy += math.Log(p)
		state = m.appendState(state, b)
	}
	return -entropy
}

func (m *Markov) selectRandom(state []byte) byte {
	next := m.Table[string(state)]
	if len(next) == 0 {
		return 0
	}
	selection := rand.Float64()
	for b, prob := range next {
		selection -= prob
		if selection < 0 {
			return b
		}
	}
	return 0
}

func (m *Markov) appendState(state []byte, b byte) []byte {
	state = append(state, b)
	if len(state) > m.History {
		copy(state, state[1:])
		state = state[:len(state)-1]
	}
	return state
}

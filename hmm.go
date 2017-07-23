package charrnn

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"runtime"
	"sync"

	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/hmm"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func init() {
	var h HMM
	serializer.RegisterTypedDeserializer(h.SerializerType(), DeserializeHMM)

	gob.Register(hmm.TabularEmitter{})
}

// HMM is a Model for a character-level hidden Markov
// model.
type HMM struct {
	HMM       *hmm.HMM
	NumStates int

	Validation float64
}

func DeserializeHMM(d []byte) (*HMM, error) {
	dec := gob.NewDecoder(bytes.NewReader(d))
	var res *HMM
	if err := dec.Decode(&res); err != nil {
		return nil, essentials.AddCtx("deserialize HMM", err)
	}
	return res, nil
}

func (h *HMM) Name() string {
	return "hmm"
}

func (h *HMM) TrainingFlags() *flag.FlagSet {
	f := flag.NewFlagSet("hmm", flag.ExitOnError)
	f.IntVar(&h.NumStates, "states", 200, "number of hidden states")
	f.Float64Var(&h.Validation, "validation", 0.1, "validation fraction")
	return f
}

func (h *HMM) GenerationFlags() *flag.FlagSet {
	return flag.NewFlagSet("hmm", flag.ExitOnError)
}

func (h *HMM) Train(s SampleList) {
	validation, training := anysgd.HashSplit(s, h.Validation)
	log.Printf("Training: %d samples (%d bytes)", training.Len(),
		training.(SampleList).Bytes())
	log.Printf("Validation: %d samples (%d bytes)", validation.Len(),
		validation.(SampleList).Bytes())

	if h.HMM == nil {
		h.initModel()
	}

	log.Println("Computing initial loss...")
	log.Printf("initial: train_loss=%f val_loss=%f", h.meanLoss(training),
		h.meanLoss(validation))

	log.Println("Training (press ctrl+c to terminate)...")
	r := rip.NewRIP()
	var iter int
	for !r.Done() {
		h.HMM = hmm.BaumWelch(h.HMM, h.samplesToChan(training), 0)
		log.Printf("iter %d: train_loss=%f val_loss=%f", iter,
			h.meanLoss(training), h.meanLoss(validation))
		iter++
	}
}

func (h *HMM) Generate() {
	_, seq := h.HMM.Sample(nil)
	for _, character := range seq {
		fmt.Print(string([]byte{character.(byte)}))
	}
	fmt.Println()
}

func (h *HMM) SerializerType() string {
	return "github.com/unixpickle/char-rnn.HMM"
}

func (h *HMM) Serialize() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(h); err != nil {
		return nil, essentials.AddCtx("serialize HMM", err)
	}
	return buf.Bytes(), nil
}

func (h *HMM) initModel() {
	var states []hmm.State
	for i := 0; i < h.NumStates; i++ {
		states = append(states, i)
	}
	var obses []hmm.Obs
	for i := 0; i < 0x100; i++ {
		obses = append(obses, byte(i))
	}
	h.HMM = hmm.RandomHMM(nil, states, 0, obses)
}

func (h *HMM) meanLoss(samples anysgd.SampleList) float64 {
	var total float64
	var divisor int

	var lock sync.Mutex
	var wg sync.WaitGroup

	ch := h.samplesToChan(samples)
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			for sample := range ch {
				loss := hmm.NewForwardBackward(h.HMM, sample).LogLikelihood()

				lock.Lock()
				// Add 1 for the terminal symbol.
				divisor += len(sample) + 1
				total += loss
				lock.Unlock()
			}
			wg.Done()
		}()
	}

	wg.Wait()

	return total / float64(divisor)
}

func (h *HMM) samplesToChan(samples anysgd.SampleList) <-chan []hmm.Obs {
	res := make(chan []hmm.Obs, 1)
	go func() {
		for _, seq := range samples.(SampleList) {
			var obses []hmm.Obs
			for _, b := range seq {
				obses = append(obses, b)
			}
			res <- obses
		}
		close(res)
	}()
	return res
}

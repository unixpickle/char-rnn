package charrnn

import (
	"flag"
	"fmt"
	"log"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/rwa"
	"github.com/unixpickle/serializer"
)

func init() {
	var l RWA
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeRWA)
}

// RWA is a Model for long short-term memory RNNs.
type RWA struct {
	rwaTrainingFlags
	rwaGenerationFlags

	Block anyrnn.Block
}

func DeserializeRWA(d []byte) (*RWA, error) {
	var b anyrnn.Block
	if err := serializer.DeserializeAny(d, &b); err != nil {
		return nil, err
	}
	return &RWA{Block: b}, nil
}

func (l *RWA) Train(samples SampleList) {
	if l.Block == nil {
		l.createModel()
	}

	validation, training := anysgd.HashSplit(samples, l.Validation)

	t := &anys2s.Trainer{
		Func: func(s anyseq.Seq) anyseq.Seq {
			return anyrnn.Map(s, l.Block)
		},
		Cost:    anynet.DotCost{},
		Params:  l.Block.(anynet.Parameterizer).Parameters(),
		Average: true,
	}

	log.Printf("Training: %d samples (%d bytes)", training.Len(),
		training.(SampleList).Bytes())
	log.Printf("Validation: %d samples (%d bytes)", validation.Len(),
		validation.(SampleList).Bytes())

	var iter int
	sgd := &anysgd.SGD{
		Fetcher:     t,
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples: &anys2s.SortSampleList{
			SortableSampleList: training.(SampleList),
			BatchSize:          l.SortBatch,
		},
		Rater: anysgd.ConstRater(l.StepSize),
		StatusFunc: func(b anysgd.Batch) {
			if validation.Len() == 0 {
				log.Printf("iter %d: cost=%v", iter, t.LastCost)
				return
			}

			vSize := l.BatchSize
			if vSize > validation.Len() {
				vSize = validation.Len()
			}
			anysgd.Shuffle(validation)
			validationBatch, _ := t.Fetch(validation.Slice(0, vSize))
			v := anyvec.Sum(t.TotalCost(validationBatch.(*anys2s.Batch)).Output())

			log.Printf("iter %d: cost=%v validation=%v", iter, t.LastCost, v)
			iter++
		},
		BatchSize: l.BatchSize,
	}

	log.Println("Training (ctrl+c to stop)...")
	l.setDropout(true)
	defer l.setDropout(false)
	sgd.Run(rip.NewRIP().Chan())
}

func (l *RWA) Generate() {
	state := l.Block.Start(1)

	last := oneHotAscii(0)
	seedBytes := []byte(l.Seed)
	for i := 0; i < l.Length; i++ {
		res := l.Block.Step(state, last)
		ch := sampleSoftmax(res.Output())
		if i < len(seedBytes) {
			ch = int(seedBytes[i])
		}
		fmt.Print(string([]byte{byte(ch)}))

		v := make([]float32, CharCount)
		v[ch] = 1
		last = anyvec32.MakeVectorData(v)
		state = res.State()
	}

	fmt.Println()
}

func (l *RWA) Name() string {
	return "rwa"
}

func (l *RWA) SerializerType() string {
	return "github.com/unixpickle/char-rnn.RWA"
}

func (l *RWA) Serialize() ([]byte, error) {
	return serializer.SerializeAny(l.Block)
}

func (l *RWA) createModel() {
	block := anyrnn.Stack{}
	inCount := CharCount
	scaler := anyvec32.MakeNumeric(16)
	for i := 0; i < l.Layers; i++ {
		rwa := rwa.NewRWA(anyvec32.CurrentCreator(), inCount, l.Hidden)
		dropout := &anyrnn.LayerBlock{Layer: &anynet.Dropout{KeepProb: l.Dropout}}
		block = append(block, rwa.ScaleInWeights(scaler), dropout)
		inCount = l.Hidden
	}
	block = append(block, &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(anyvec32.CurrentCreator(), inCount, CharCount),
			anynet.LogSoftmax,
		},
	})
	var size int
	for _, p := range block.Parameters() {
		size += p.Vector.Len()
	}
	l.Block = block
}

func (l *RWA) setDropout(enabled bool) {
	for _, block := range l.Block.(anyrnn.Stack) {
		if block, ok := block.(*anyrnn.LayerBlock); ok {
			if do, ok := block.Layer.(*anynet.Dropout); ok {
				do.Enabled = enabled
			}
		}
	}
}

type rwaTrainingFlags struct {
	StepSize   float64
	Validation float64
	Dropout    float64
	Hidden     int
	Layers     int
	BatchSize  int
	SortBatch  int
}

func (l *rwaTrainingFlags) TrainingFlags() *flag.FlagSet {
	res := flag.NewFlagSet("rwa", flag.ExitOnError)
	res.IntVar(&l.Hidden, "hidden", 512, "hidden neuron count")
	res.IntVar(&l.Layers, "layers", 2, "RWA layer count")
	res.Float64Var(&l.StepSize, "step", 0.001, "step size")
	res.Float64Var(&l.Validation, "validation", 0.1, "validation fraction")
	res.Float64Var(&l.Dropout, "dropout", 0.6, "dropout remain probability")
	res.IntVar(&l.BatchSize, "batch", 32, "SGD batch size")
	res.IntVar(&l.SortBatch, "sortbatch", 128, "sample sort batch size")
	return res
}

type rwaGenerationFlags struct {
	Length int
	Seed   string
}

func (l *rwaGenerationFlags) GenerationFlags() *flag.FlagSet {
	res := flag.NewFlagSet("rwa", flag.ExitOnError)
	res.IntVar(&l.Length, "length", 100, "generated string length")
	res.StringVar(&l.Seed, "seed", "", "starting text")
	return res
}

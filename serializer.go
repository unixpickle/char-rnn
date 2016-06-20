package main

import "github.com/unixpickle/serializer"

const (
	serializerTypePrefix = "github.com/unixpickle/char-rnn"
	serializerTypeLSTM   = serializerTypePrefix + "LSTM"
)

func init() {
	serializer.RegisterDeserializer(serializerTypeLSTM, DeserializeLSTM)
}

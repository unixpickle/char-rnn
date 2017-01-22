// +build cuda

package main

import (
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/anyvec/cuda"
)

func init() {
	handle, err := cuda.NewHandle()
	if err != nil {
		panic(err)
	}
	anyvec32.Use(cuda.NewCreator32(handle))
}

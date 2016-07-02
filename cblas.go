// +build cblas

package main

import (
	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
)

func init() {
	blas64.Use(cgo.Implementation{})
}

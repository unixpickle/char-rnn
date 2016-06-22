# char-rnn

This is an attempt to recreate [karpathy/char-rnn](https://github.com/karpathy/char-rnn) with my own RNN package. Wish me luck.

# Usage

First, gather a folder with a bunch of text files in it (or with one big text file in it). Let's call this `path/to/text`.

Now, install [Go](https://golang.org/doc/install) and setup a GOPATH. Once this is done, you are ready to install char-rnn itself:

```
$ go get -u github.com/unixpickle/char-rnn
```

## Training

You can train char-rnn on some data as follows:

```
$ cd $GOPATH/src/github.com/unixpickle/char-rnn
$ go run *.go train lstm /path/to/lstm /path/to/text
2016/06/22 17:52:53 Loaded model from file.
2016/06/22 17:52:53 Training LSTM on 22308 samples...
2016/06/22 17:59:29 Epoch 0: cost=1857807.936353
...
```

You can set the `GOMAXPROCS` environment variable to specify how many CPU threads to use for training.

If the `/path/to/lstm` file already exists, it will be loaded as an LSTM and training will resume where it left off. Otherwise, a new LSTM will be created.

It may take a while to train the LSTM reasonably well. On karpathy's [tinyshakespeare](https://github.com/karpathy/char-rnn/tree/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare), it took my Intel NUC (quad-core i3 with 1.7GHz) roughly 18 hours to train reasonably well (although for much of that time I was only using a single CPU core).

To pause or stop training, press Ctrl+C exactly once. This will finish the current epoch and then terminate the program (which may take somewhere on the magnitude of a few hours, if your data set is large). Once the program has terminated, a trained LSTM will be saved to `/path/to/lstm`. **Note:** if you hit Ctrl+C more than once, the program will terminate without saving.

## Generating text

Once you have trained an LSTM, you can use it to generate a block of text. You must decide how much text to generate (e.g. 1000 characters, like below):

```
$ go run *.go gen /path/to/lstm 1000
```

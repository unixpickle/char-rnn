# char-rnn

I have recreated [karpathy/char-rnn](https://github.com/karpathy/char-rnn) with my own RNN package. It works fairly well, and I have used it to generate [some cool results](#example).

# Usage

First, gather a folder with a bunch of text files in it (or with one big text file in it). Let's call this `path/to/text`.

Now, install [Go](https://golang.org/doc/install) and setup a GOPATH. Once this is done, you are ready to install char-rnn itself:

```
$ go get -u github.com/unixpickle/char-rnn
$ cd $GOPATH/src/github.com/unixpickle/char-rnn
$ go build
```

This will generate an executable called `char-rnn` in your current directory. If you have the Go BLAS package setup to use a C implementation of BLAS, you can access that by building with `go build -tags cblas`.

## Training

You can train char-rnn on some data as follows:

```
$ ./char-rnn train lstm /path/to/lstm /path/to/sample/directory
2016/06/22 17:52:53 Loaded model from file.
2016/06/22 17:52:53 Training LSTM on 22308 samples...
2016/06/22 17:59:29 Epoch 0: cost=1857807.936353
...
```

You can set the `GOMAXPROCS` environment variable to specify how many CPU threads to use for training.

If the `/path/to/lstm` file already exists, it will be loaded as an LSTM and training will resume where it left off. Otherwise, a new LSTM will be created.

It may take a while to train the LSTM reasonably well. On karpathy's [tinyshakespeare](https://github.com/karpathy/char-rnn/tree/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare), it took my Intel NUC (quad-core i3 with 1.7GHz) roughly 18 hours to train reasonably well (although for much of that time I was only using a single CPU core).

To pause or stop training, press Ctrl+C exactly once. This will finish the current mini-batch and then terminate the program. Once the program has terminated, a trained LSTM will be saved to `/path/to/lstm`. **Note:** if you hit Ctrl+C more than once, the program will terminate without saving.

## Generating text

Once you have trained an LSTM, you can use it to generate a block of text. You must decide how much text to generate (e.g. 1000 characters, like below):

```
$ ./char-rnn gen /path/to/lstm 1000
```

# Example

I ran a GRU on the output of `ls -l /usr/bin` and then generated some dir listings:

```
-rwxr-xr-x  35 root   wheel       821 Aug 23  2015 iptab5.18
-r-xr-xr-x   1 root   wheel      3659 Sep 28  2015 instmodse
-rwxr-xr-x   1 root   wheel        75 Oct 25  2015 info3eal -> /System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/rmic
lrwxr-xr-x   1 root   wheel        84 Oct 25  2015 javmap -> /System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/kchase
-rwxr-xr-x   1 root   wheel     59576 Oct 17  2015 anplrac
-rwxr-xr-x   1 root   wheel        77 Oct 25  2015 edbsc -> cling
-r-xr-xr-x   1 root   wheel     18176 Oct 17  2015 nv5.16
-rwxr-xr-x   1 root   wheel     17204 Aug 22  2015 pod2readme5.16
-rwxr-xr-x  35 root   wheel       811 Aug 23  2015 lwp-download5.16
-r-xr-xr-x   1 root   wheel      3573 Aug 22  2015 dbiprof5.18
-rwxr-xr-x   1 root   wheel     23368 Oct 17  2015 enice
-rwxr-xr-x   1 root   wheel        43 Oct 25  2015 jstat -> /System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/intext
-rwxr-xr-x   1 root   wheel        77 Oct 25  2015 netalloc.5 -> ../../System/Library/Frameworks/Python.framework/Versions/-arwervim
lrwxr-xr-x   1 root   wheel        82 Oct 25  2015 j0 -> vmeadsrad
-rwxr-xr-x   1 root   wheel     18176 Oct 17  2015 gzeratex
-rwxr-xr-x   1 root   wheel      1947 Aug 22  2015 config_data5.16
-rwxr-xr-x   1 root   wheel      9151 Aug 23  2015 ifstroc5.16
-rwxr-xr-x   1 root   wheel         2 Oct 25  2015 viaevketat-cvisthar -> 2toc2.6
```

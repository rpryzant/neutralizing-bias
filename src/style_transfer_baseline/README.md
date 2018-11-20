## Annotation

- x = all words in a sentence
- c(x) the _content_ words of a sentence, those that went unchanged
- m(x) the _modified_ words of a sentence, those that *were* changed
- v (= pre/post), the attribute of that sentence

## Delete
Train:
```
e = [ RNN( c(x_v) ), embedding_v ]
x_hat = RNN( e )
```

Test:
```
e = [ RNN( c(x_pre) ), embedding_post ]
x_post_hat = RNN(e)
```


## Delete AND Retrieve

Train
```
a'(x_v) = {
	 0.9: a(x_v)
	 0.1: another a(x'_v) within word-edit distance 1
}
e = [ RNN(c(x_v)), RNN(a'(x_v)) ]
x_v_hat = RNN(e)
```


Test
```
x'_post = argmin_{ x'_post (possibly excluding the true match) } ( tfidf( c(x'_post), c(x_pre) ) )
e = [ RNN( c(x_pre) ), RNN( a(x'_post) ) ]
x_post_hat = RNN(e)
```
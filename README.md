# Setup

```sh
$ brew install bazel
$ virtualenv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

# References

## Tensorflow deprecated

- [tensorflow - What are the differences between tf.initialize_all_variables() and tf.global_variables_initializer() - Stack Overflow](https://stackoverflow.com/questions/41439254/what-are-the-differences-between-tf-initialize-all-variables-and-tf-global-var)
- [python - tensorflow:AttributeError: 'module' object has no attribute 'mul' - Stack Overflow](https://stackoverflow.com/questions/42217059/tensorflowattributeerror-module-object-has-no-attribute-mul)
- [tf.complex_abs not supported in r1.0.0 · Issue #7405 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/7405)
- [tensorflow/RELEASE.md at master · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md)

## Matplotlib

- [Working with Matplotlib on OSX — Matplotlib 2.1.0 documentation](https://matplotlib.org/faq/osx_framework.html)

```sh
$ PYTHONHOME=$VIRTUAL_ENV /usr/local/bin/python2.7 image.py
```

or

```sh
$ source frameworkpython
$ frameworkpython image.py
```

## MNIST

- [Typo in slim's download and convert mnist, `Size` should be `size` · Issue #668 · tensorflow/models](https://github.com/tensorflow/models/issues/668)

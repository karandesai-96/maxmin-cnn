MaxMin Convolutional Neural Networks
====================================

This repository contains implementation of the paper [MaxMin Convolutional Neural Networks for Image Classification](http://webia.lip6.fr/~thomen/papers/Blot_ICIP_2016.pdf).


How to walk through this repository ?
-------------------------------------

MaxMin CNN performs better than a classical CNN and it should take hardly 30 minutes to understand why and how to use them for your advantage. This is also meant to be a reference for learners, even beginners to Deep Learning. As a pre requisite you should know working of a simple convolutional layer, pooling layer and the meaning of ReLU activation.

1. Have a look through this _README_ completely to get the information about what all has been provided.  
2. Accompanying this repository, [I have written a blog post](https://karandesai-96.github.io/2016/12/21/maxminify-your-cnns/) about the design and working of a MaxMin Convolution Neural Network, before starting on this repository, you may like to have a quick read there - should hardly take 10 minutes or so.  
3. Next up, refer the implementation of MaxMin Convolution Layer written in **`keras_maxmin_impl.py`**.
4. Check out the notebook, which contains implementation of the whole training routine and performance comparison on baseline models trained using **CIFAR 10 dataset**.
5. You may like to "Watch" this repository as I plan to / encourage contributions on thngs mentioned in _TODO_ section below.


Implementation Details
----------------------

The complete implementation is done using **Keras** library of Python. One does not need to have much familiarity with Keras API - the design is quite intuitive, with suitable comments included in the notebook as well as script. I have configured **Tensorflow (GPU) as Keras backend**. Training occurs on my machine with 4GB NVidia GeForce 940 MX card.

**NOTE**: I have modified the default Keras Logger and then built Keras from source, so the verbose outputs are a bit different than those obtained by using versioned release of Keras. It is just to keep this notebook less cluttered and does not affect the results in any way. Do not worry if you are not able to reproduce the same output log on your machine.


TO-DOs
------

* Add in more notebooks showing performance on bigger standard models such as ZFNet and VGG-16, 19.  
* Add in minimal implementation of MaxMin Convolution Layer using a different library or language (such as Caffe, Torch, Tensorflow or CNTK)


Contributing
------------

Issues related to any queries in the blog post or notebook explanation and Pull Requests related to current bug fixes or new additions related to TODOs are most welcome!


LICENSE
-------

```
MIT License, Copyright (c) 2016 Karan Desai
```  
Feel free to use any part of my implementation as per requirement. Reach out to me if you make something using it, I would love to hear from you.

# loaders

## ImageNet
### references

- [How to reproduce ImageNet validation results](http://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html) and its source code on [github](https://github.com/calebrob6/imagenet_validation)

### prerequisites

- Validateion set: [ILSVRC2012_img_val.tar](!) from [ImageNet website]()
- Development toolket for Task 1 and 2: [ILSVRC2012_devkit_t12.tar](!)
- [synset_words.txt]() from [`Caffe` github repo]()

Execute below commands to prepare downloaded data.
```shell
cd loaders
tar xvf /path/to/ILSVRC2012_devkit_t12.tar -C imagenet/downloads
tar xvf /path/to/ILSVRC2012_img_val.tar -C imagenet/downloads/ILSVRC2012_img_val
```
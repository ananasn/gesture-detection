> **_ВАЖНО:_** Все компоненты очень чувствительны к версиям, поэтому необходимо использовать только указанные версии!


# Установка TensorFlow для CPU (это для работы уже с готовой моделью)

1. Установить Python 3.7.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```

**Возможно придется изменить симлинк на python3**

2. Установить TensorFlow 1.13.

```
pip3 install --ignore-installed --upgrade tensorflow==1.13
```

__Если потребуется переустановка, пакеты вначале удалять, иначе будет две разные версии пакета! Например:__

```
pip3 uninstall tensorflow
pip3 install --ignore-installed --upgrade tensorflow==1.13
```

3. Проверяем работу.

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

__Возможно, будут выведены какие-то предупреждения, но если они с символом I, то все нормально__

#Установка TensorFlow для GPU (для обучения модели)

1. Установить CUDA Toolkit 10.0.

Скачать и установить [отcюда.](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)

2. Установить CUDNN.

* Перейти [вот сюда](https://developer.nvidia.com/rdp/cudnn-download) (понадобится регистрация).
* Выбрать пункт *Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.0*.
* Скачать установочный файл для нужно версии ОС.

3. Добавить переменные окружения

В `~\.bashrc` добавить следующие строки, перезапустить сессию (сделать logout или закрыть эмулятор терминала).

```
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```




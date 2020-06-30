> **_ВАЖНО:_** Все компоненты очень чувствительны к версиям, поэтому необходимо использовать только указанные версии!


# Установка TensorFlow для CPU (это для работы уже с готовой моделью)

1. Установить Python 3.7.

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.7
```

> **_ВАЖНО:_** Возможно придется изменить симлинк на python3

2. Установить TensorFlow 1.13.

```
pip3 install --ignore-installed --upgrade tensorflow==1.13
```

> **_ВАЖНО:_** Если потребуется переустановка, пакеты вначале удалять, иначе будет две разные версии пакета! Например:

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

> **_ВАЖНО:_** Возможно, будут выведены какие-то предупреждения, но если они с символом I, то все нормально

# Установка TensorFlow для GPU (для обучения модели)

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

4. Установить TensorFlow GPU 1.13.

```
pip3 install --ignore-installed --upgrade tensorflow-gpu==1.13
```

5. Аналогично версии для CPU проверяем работоспособность, на этот раз вывод будет более подробным и в нем тоже должны быть только предупреждения с символом I:

```
2020-06-28 19:39:57.763993: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-06-28 19:39:57.767488: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-28 19:39:57.771466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0
2020-06-28 19:39:57.774334: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N
2020-06-28 19:39:57.776874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3009 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
```

# Установка моделей обучения для TensorFlow

1. В данном примере используются модели версии 1.13, все необходимые файлы уже лежат в этом репозитарии в папке `models`. В случае необходимости скачать их можно [отсюда](https://github.com/tensorflow/models/releases/tag/v1.13.0)

2. Установить необходимые для обучения пакеты

```
pip3 install --ignore-installed --upgrade pillow==6.2.1
pip3 install --ignore-installed --upgrade lxml==4.4.1
pip3 install --ignore-installed --upgrade jupyter==1.0.0
pip3 install --ignore-installed --upgrade matplotlib==3.1.1
pip3 install --ignore-installed --upgrade opencv-python==3.4.2.17
pip3 install --ignore-installed --upgrade pathlib==1.0.1
pip3 install --ignore-installed --upgrade numpy==1.16
```

3. Установить Protocol Buffer.

```
sudo add-apt-repository ppa:maarten-fonville/protobuf
sudo apt update
sudo apt protoc
```

4. Перейти в директорию `models/research/` и выполнить команду:

```
protoc object_detection/protos/*.proto --python_out=.
```

для компиляции файлов protobuf (они там уже есть скомпилированные, но возможно понядобится сделать это еще раз).

5. Установить пакет `object_detection`.

Перейти в директорию `models\research\object_detection` и выполнить команду `pip install .`

6. Добвить пакет `slim` в `PYTHONPATH`. В `~\.bashrc` добавить следующие строки, перезапустить сессию (сделать logout или закрыть эмулятор терминала).

```
export PYTHONPATH=$PYTHONPATH:<ПУТЬ_К_ЭТОМУ_РЕПОЗИТАРИЮ>/models/research/slim
```

7. Установить утилиту для разметки изображений LabelImg

```
pip install labelImg
```

# Подготовка и разметка датасета

1. Для каждого класса объектов необходимо сделать 100 или больше (желательно больше) фотографий не очень большого разрешения (1600х1200). Фотографии должны быть максимально разнообразны: общий план, крупный план, разная освещенность и т.д. Чем больше и разнообразнее будет датасет, тем лучше будет обучаться модель и тем лучше будет идти расплознавание.

2. Все фотографии положить в папку `workspace\training-demo\images`.

3. Запусить labelImg командой `python3 labelImg workspace\training_demo\images`. Обвести прямоугольниками все объеты, назначить им классы. В папке `images` для каждой фотографии будет создан отдельный xml-файл с описанием областей и классов.

4. Из папки `scripts` этого репозитария запустить утилиту разбиения датасета на тренировочный и тестовый:

```
python3 partition-dataset.py -x -i ../workspace/training_demo/images -r 0.1
```

В директории `workspace\training-demo\images` автоматически создадуться две поддиректории `train` и `test`. В первую будет помещено 90% фотографий с xml аннотациями, а во вторую -- 10%. Последний аргумент в команде задает это отношение.

5. Создаем файл с картой меток `label-map.pbtxt` и помещаем его в директорию `workspace\training-demo\annotations`.

```
item {
  id: 1
  name: 'one_finger'
}

item {
  id: 2
  name: 'two_fingers'
}
```

> **_ВАЖНО:_** Названия меток должны быть теми же, что и при разметке изображений в labelImg.






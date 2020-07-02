> **_ВАЖНО:_** Все компоненты очень чувствительны к версиям, поэтому необходимо использовать только указанные версии! Установка производилать под ОС Ubuntu 18.04.

# Установка Anaconda для Python 3.7

1. Скачать Anaconda Python 3.7 [отсюда](https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh).
2. Запусить исполняемый файл `*.sh` и выполнить инструкции устновщика. [Здесь](https://docs.anaconda.com/anaconda/install/linux/) есть подробная информация по установке.
3. В ответ на вопрос "Do you wish the installer to prepend the Anaconda<2 or 3> install location to PATH in your /home/<user>/.bashrc ?", ответить "yes". Если ввести "no", можно будет потом вручную добавить путь к Anaconda, иначе команда conda не будет работать.
4. Сделать `source .bashrc` или перезапустить терминал.

# Установка TensorFlow для CPU (это для работы уже с готовой моделью)

1. Cоздаем виртуальное окружение и активируем его

```
conda create -n tensorflow_cpu
conda activate tensorflow_cpu
```

2. Установить TensorFlow 1.13

```
conda install numpy=1.16 tensorflow=1.13
```

3. Проверяем работу.

```
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

> **_ВАЖНО:_** Возможно, будут выведены какие-то предупреждения, но если они с символом Info, то все нормально

4. Деактивировать виртуальное окружение

```
conda deactivate
```

# Установка TensorFlow для GPU (для обучения модели)

1. Установить CUDA Toolkit 10.0 (убедиться, что драйверы для nvidia установлены, для Ubuntu 18.04 -- версия 410).

Скачать и установить [отcюда.](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)

```
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo apt update
sudo apt install cuda
```

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

4. Создаем отдельное виртуальное окружение и устанавливаем TensorFlow GPU 1.13.

```
conda create -n tensorflow_gpu
conda activate tensorflow_gpu
conda install numpy=1.16 tensorflow-gpu=1.13
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
conda install pillow==6.2.1
conda install lxml==4.4.1
conda install jupyter==1.0.0
conda install matplotlib==3.1.1
conda install opencv==3.4.2
conda install pathlib==1.0.1
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

Скачать вот [отсюда](http://tzutalin.github.io/labelImg/) и запустить:

```
./labelImg
```

# Подготовка и разметка датасета

1. Для каждого класса объектов необходимо сделать 100 или больше (желательно больше) фотографий не очень большого разрешения (1600х1200). Фотографии должны быть максимально разнообразны: общий план, крупный план, разная освещенность и т.д. Чем больше и разнообразнее будет датасет, тем лучше будет обучаться модель и тем лучше будет идти расплознавание.

2. Все фотографии положить в папку `workspace\training-demo\images`.

3. Запусить labelImg командой `python3 labelImg workspace\training_demo\images`. Обвести прямоугольниками все объеты, назначить им классы. В папке `images` для каждой фотографии будет создан отдельный xml-файл с описанием областей и классов.

4. Из папки `scripts` этого репозитария запустить утилиту разбиения датасета на тренировочный и тестовый:

```
python partition-dataset.py -x -i ../workspace/training_demo/images -r 0.1
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

> **_ВАЖНО:_** Иденитификаторы обязательно должны начинаться с 1, а не с 0.


# Создание TensorFlow Records

1. Преобразовать файлы `xml` в два файла `csv`.

Из папки `scripts` этого репозитария запустить утилиту `xml2csv.py` для тестового и тренировочного датасетов:

```
python xml2csv.py -i ../workspace/training-demo/images/train -o ../workspace/training-demo/annotations/train-labels.csv
python xml2csv.py -i ../workspace/training-demo/images/test -o ../workspace/training-demo/annotations/test-labels.csv
```

2. Преобразовать файлы `csv` в `record`.

Из папки `scripts` этого репозитария запустить утилиту `generate-tfrecord.py`:

```
python generate-tfrecord.py --label0=one_finger --label1=two_fingers --csv_input=../workspace/training-demo/annotations/train-labels.csv --output_path=../workspace/training-demo/annotations/train.record --img_path=../workspace/training-demo/images/train
python generate-tfrecord.py --label0=one_finger --label1=two_fingers --csv_input=../workspace/training-demo/annotations/test-labels.csv --output_path=../workspace/training-demo/annotations/test.record --img_path=../workspace/training-demo/images/test
```

> **_ВАЖНО:_** Заменить `--label0=one_finger --label1=two_fingers` на свои метки.

Если будет больше двух классов меток, в файле `generate-tfrecord.py` изменить строки, начиная с 44:

```
# TO-DO replace this with label map
# for multiple labels add more else if statements.
def class_text_to_int(row_label):
    if row_label == FLAGS.label0:
        return 1
    elif row_label == FLAGS.label1:
        return 2
```

# Настроить Training Pipeline

1. Существует большое количество предаварительно натренированных моделей распознавания объектов. Свою модель мы будем строить на основании существующей. Единственная проблема: таких моделей очень много, отличаются они скоростью и точностью обучения. Посмотерть какие бывают модели и их параметры можно вот [здесь](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). В данном примере будет использоваться `ssd_inception_v2_coco`.

2. Скачиваем модель в формате `*.tar.gz` [отсюда](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) и распаковываем в директорию `workspace\training-demo\pre-trained-model`.

3. В директории `workspace\training-demo\training` уже лежит файл конфигурации для этой модели `ssd_inception_v2_coco.config`. Заготовка этого файла скачивается [отсюда](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_inception_v2_coco.config).

В файле важными явлюятся строки

* 9 -- num_classes: 2 -- Задает количество классов
* 77 -- type: 'ssd_inception_v2' -- Тип модели (именно так, без `coco` в конце)
* 136 -- batch_size: 12 -- Этот безразмерный параметр отвечает за использование ОЗУ, его можно увеличивать или уменьшать в зависимости от использованной памяти во время обучения. Значение 12 дало примерно 10 ГБ из 16 ГБ
* 151 -- fine_tune_checkpoint: "pre-trained-model/model.ckpt" -- Путь к файлам предварительно натренированной модели (которую скачали выше)
* 168+

```
train_input_reader: {
    tf_record_input_reader {
        input_path: "annotations/train.record"
    }
    label_map_path: "annotations/label-map.pbtxt"
}
```

Пути файл `record` для тренировачного датасета и файлу с классами объектов

* 187+

```
eval_input_reader: {
    tf_record_input_reader {
        input_path: "annotations/test.record"
    }
    label_map_path: "annotations/label-map.pbtxt"
    shuffle: false
    num_readers: 1
}
```

Тоже самое для тестового датасета

# Тренировка модели

1. Из директории `workspace/training-demo` запустить файл `model_main.py`.

> **_ВАЖНО:_** Он там уже есть, но вообще просто копируется из `models/research/object_detection`

```
python model_main.py --alsologtostderr --model_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config
```

Обучение началось!!! В консоле практически ничего интересного происходить не будет, только процесс начнет потреблять ресурсы. Процесс небыстрый, в конфигурации, в строке 157 стоит ограничение в 200000 шагов. На Core i5 8400 и GTX 1050 за 16 часов прошло только 50000 шагов на датасете из 45 изображений для двух классов. Хорошая новость: процесс всегда можно прервать по `CTRL+C`, а потом запусить с этого же места этой же командой.

2. Наблюдаем за процессом обучения.

```
tensorboard --logdir=training\
```

Открываем в браузере адрес `http://localhost:6006` и видим много графиков, а также распознанные изображения из тестового датасета. Самый вайжный параметр -- `total_loss`, он должен уменьшаться и стремиться к 1--2. Последнее означает, что обучение идет правильно, в противном случае, надо изменить датасет. Значения меньше 1 тоже не очень хороший показатель, они означают, что произошло переобучение и сеть будет находить объекты там, где их нет.

# Получение файла замороженной модели

1. В директории `training_demo/training` найти файл `model.ckpt-*` с максимальным номером.

2. Из директории `workspace/training-demo` запустить файл `export_inference_graph.py`

> **_ВАЖНО:_** Он там уже есть, но вообще просто копируется из `models/research/object_detection`

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-50802 --output_directory trained-inference-graphs/output_inference_graph_v1.pb
```

> **_ВАЖНО:_** Директория `trained-inference-graphs` создается автоматически и при повторном запуске этой команды, ее надо удалять

# Проверка работы модели с вебкамерой

Из корня этого репозитария запускаем файл `webcam.py`.

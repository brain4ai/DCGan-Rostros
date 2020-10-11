import tensorflow as tf
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time

#PATH
dataset = "./dataset"
#Decidimos si entrenamos desde cero o cargamos el ultimo checkpoint
load_ckpt = False
#Si no queremos entrenar, solo queremos generar imagenes, pondremos inference_mode = True
inference_mode = False


print("[INFO] Cargar modelo preentrenado = " + str(load_ckpt))
print("[INFO] Activar inference_mode = " + str(inference_mode))

#Funcion para cargar imagenes
def load_data():
    print("[INFO] Cargando imagenes de entrenamiento...")
    filelist = os.listdir(dataset)
    n_imgs = len(filelist)
    x_train = np.zeros((n_imgs, 128, 128, 3))

    for i, fname in enumerate(filelist):
        imagen = imread(os.path.join(dataset, fname))
        x_train[i, :] = (imagen - 127.5) / 127.5  # Normalizamos entre [-1, 1]
    print("[INFO] Terminado correctamente. Cargadas " + str(n_imgs) + " imagenes.")
    return x_train

'''Cargamos nuestro dataset de imagenes
En nuestro caso, a la vista de nuestro dataset, son 3754 imagenes con un tamaño todas de 128x128'''
train_images = load_data()

BUFFER_SIZE = train_images.shape[0] #3754 en el caso de nuestro dataset
BATCH_SIZE = 64

'''Con el siguiente reshape añadimos el BUFFER_SIZE a train_images
en nuestro caso (3754,128,128,3)'''
train_images = train_images.reshape(BUFFER_SIZE, 128, 128, 3).astype('float32')

'''Ahora si, construimos nuestro train_dataset añadiendo las imagenes cargadas, y asegurandonos de hacer un .shuffle
para cargar nuestras imagenes desordenadas'''
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

''' Definimos nuestro GENERADOR:
-El generador parte de un vector aleatorio de 100 componentes.
-El generador comienza con una capa Densa para capturar el vector aleatorio. Después, vamos aumentando varias veces el 
tamaño de imagen hasta alcanzar el tamaño de imagen deseando. En nuestro caso queremos 128x128x3'''
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 1024)))
    assert model.output_shape == (None, 8, 8, 1024) # Nota: None es nuestro batch size
    #8x8x1024

    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #8x8x512

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #16x16x256

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #32x32x128

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #64x64x64

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 3)
    #128x128x3

    return model

# Creamos el GENERADOR
generator = make_generator_model()
print("[INFO] Generador creado correctamente.")

''' Definimos nuestro DISCRIMINADOR:
-El discriminador es un clasificador basado en una convolutional neural network
-Recomiendo hacerlo bastante similar en cuanto a capas con el generador. Evitaremos mucha diferencia de aprendizaje
entre uno y otro que puede llevar a la red a no ser capaz de entrenarse'''
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Creamos el DISCRIMINADOR
discriminator = make_discriminator_model()
print("[INFO] Discriminador creado correctamente.")

#Definimos funcion de error y optimizadores para ambos modelos
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

'''Funcion de PERDIDAS o funcion de costes del DISCRIMINADOR.
-Este metodo evalua como de bueno es el discriminador distinguiendo imagenes reales
 de imagenes generadas (falsas)
-Cuando le estamos pasando una imagen real, se compara la prediccion del discriminador con un array de 1s.
-Cuando le estamos pasando una imagen falsa, se compara la prediccion del discriminador con una array de 0s.'''
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

'''Funcion de PERDIDAS  o funcion de costes del GENERADOR.
-Este metodo evalua como de bueno es el generador generando imagenes falsas que engañen lo maximo al discrimiandor.
-Si el generador funciona perfectamente, el discriminador debería clasificar como imagenes reales (1s) las
imagenes falsas que generamos'''
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

'''Como vemos, hay dos optimizadores diferentes, el optimizador para el generador y el optimizador para el 
discriminador.
Esto es porque vamos a entrenar dos redes diferentes por separado pero compitiendo entre si'''
generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.5)

'''A continuación definimos como guardaremos los checkpoint. En este caso es muy importante, porque el proceso
de entrenamiento es largo y podemos querer interrumpirlo y continuar entrenando por donde lo habiamos dejado'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

'''La variable load_ckpt controla si queremos cargar el ultimo checkpoint. Es decir, cargar la ultima version
preentrenada del generador y el discriminador.
-Esto puede usarse para hacer predicciones y generaciones de imagenes
-O, puede usarse para continuar entrenando por donde lo habiamos dejado'''
if load_ckpt :
    #Cargamos Generador y Discriminador
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    '''Comprobamos que la carga ha sido correcta generando una imagen con el generador preentrenado
        -Nota: hay que mantener training=False'''
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0])
    plt.show()

    '''Comprobamos que la carga ha sido correcta pasando la imagen por el discriminador preentrenado'''
    decision = discriminator(generated_image)
    print("[INFO] Decision del discriminador preentrenado:" + str(decision))


'''-Definimos numero de epocas a entrenar -> EPOCHS
   -Dimension del vector random a partir del cual generaremos las imagenes -> noise_dim
   -Numero de muestras a generar, esto es para ir visualizando si todo evoluciona correctamente
   -Definimos cada cuantas epocas guardaremos un checkpoint -> n_save_ckp
'''
EPOCHS = 1000
noise_dim = 100
num_examples_to_generate = 16
n_save_ckp = 100

'''Generamos un vector de numeros aleatorios y lo almacenamos.
Estos numeros aleatorios seran nuestras imagenes que generaremos.
De esta manera es mas facil visualizar el progreso del entrenamiento sobre una misma gama de imagenes.
'''
seed = tf.random.normal([num_examples_to_generate, noise_dim])

'''Training loop.
-Comienza con el generador recibiendo un vector aleatorio de longitud 100. Con esto genera una imagen falsa.
-Despues, usamos el discriminador para clasificar imagenes reales (extraidas de nuestro dataset) e imagenes
falsas (generadas por el generador).
-Se calculan las funciones de perdidas de ambos y se computan los gradientes para saber si van mejorando los
modelos o por el contrario van empeorando.
'''
# @tf.function es un decorador de funciones de python. En la funcion train es normal encontrarlo.
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Generamos algunas imagenes
    generate_and_save_images(generator, discriminator,
                             epoch + 1,
                             seed)

    # Salvamos los modelos cada n_save_ckp epocas
    if (epoch + 1) % n_save_ckp == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generamos algunas imagenes tras la ultima epoca
  generate_and_save_images(generator, discriminator,
                           epochs,
                           seed)

#Funcion para generar imagenes
def generate_and_save_images(modelG, modelD, epoch, test_input):
    # Mantenemos training = False
    # Esto es para que nuestro modelo funcione en modo inferencia
    predictions = modelG(test_input, training=False)
    # Si queremos ver las calificaciones del discriminador a las imagenes podemos descomentar lo siguiente
    #calificacion = modelD(predictions, training=False)
    #print(calificacion)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    plt.savefig('./image_at_epoch/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)

# Lanzamos a entrenamos nuestro modelo
if not inference_mode:
    print("[INFO] Comienza el entrenamiento...")
    train(train_dataset, EPOCHS)
    print("[INFO] Entrenamiento finalizado correctamente")


''' ----- Inference mode ----- '''
# Generamos n imagenes con el modelo preentrenado
def generate_n_im(modelG, modelD, n, test_input):
    predictions = modelG(test_input, training=False)
    for i in range(n):
        fig = plt.figure()
        plt.imshow(predictions[i])
        plt.axis('off')
        plt.savefig('image_example_{:04d}.png'.format(i))
        plt.close(fig)

if inference_mode:
    #Numero de imagenes a generar
    num_examples_to_generate = 10

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    generate_n_im(generator, discriminator, num_examples_to_generate, seed)
    print("[INFO] " + str(num_examples_to_generate) + " Imágenes generadas correctamente")
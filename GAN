import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Lambda, Concatenate, Dropout, BatchNormalization, Reshape, Conv2D, Input, LeakyReLU, Flatten, Dense, Activation, Conv2DTranspose
from keras.optimizers import Adam
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import keras.backend as K
import pickle
# %matplotlib qt #for better 3D Graph on Spyder

class GAN:
    def __init__(self, x1,x2,x3,x4,x5,x6,x7, GANlossHistory = [], DislossHistory = []):
        self.input_dim = x1
        self.generator_conv_t_filters = x2
        self.generator_conv_t_kernel_size = x3
        self.generator_conv_t_strides = x4
        self.discriminator_conv_t_filters = x5
        self.discriminator_conv_t_kernel_size = x6
        self.discriminator_conv_t_strides = x7
        
        self.n_layers_discriminator = len(self.discriminator_conv_t_filters)
        
        self.GANlossHistory = GANlossHistory
        self.DislossHistory = DislossHistory
    ####### Decoder definition #######
    def f_Generator(self):
        self.generator_input = Input(shape=(self.input_dim,), name='generator_input')

        x = Dense(7*7*128)(self.generator_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Reshape((7,7,128))(x)

        for i in range(len(self.generator_conv_t_filters)):
            conv_t_layer = Conv2DTranspose(
                filters = self.generator_conv_t_filters[i],
                kernel_size = self.generator_conv_t_kernel_size[i],
                strides = self.generator_conv_t_strides[i],
                padding = 'same',
                name = 'Generator_Conv2DTrans_' + str(i))
            x = conv_t_layer(x)
            
            
            if i < len(self.generator_conv_t_filters) - 1:
                x = BatchNormalization()(x)
                x = LeakyReLU(name='LeakyRelu_'+str(i))(x)

        self.generator_output = Activation('tanh')(x)
        self.generator = Model(self.generator_input, self.generator_output, name='Generator')
        
    ### Discriminateur 
    def f_Discriminator(self):
        self.discriminator_inputs = Input((28,28,1), name = "discriminator_input")
        x = self.discriminator_inputs

        for i in range(self.n_layers_discriminator):
            conv_layer = Conv2D(
                filters = self.discriminator_conv_t_filters[i]
                , kernel_size = self.discriminator_conv_t_kernel_size[i]
                , strides = self.discriminator_conv_t_strides[i]
                , padding = 'same'
                , name = 'discriminator_conv_' + str(i)
                )

            x = conv_layer(x)
            x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        self.discriminator_outputs = Dense(1, activation='sigmoid', name='discriminator_output')(x)
        self.discriminator = Model(self.discriminator_inputs, self.discriminator_outputs, name="discriminator")

    ####### Full Model definition #######
    def f_FullModel(self, learning_rate, dis_learning_rate=0.00002):

        self.discriminator.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = dis_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False
        
        model_input = self.generator_input
        reconstructed_image = self.generator(model_input)
        discriminator_output = self.discriminator(reconstructed_image)
        
        self.gan_model = Model(model_input, 
                                   discriminator_output,
                                   name='Gan')
        
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.gan_model.compile(optimizer=optimizer, loss = self.generator_adversarial_loss)

    def generator_adversarial_loss(self, y_true, y_pred):
        GA_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(y_pred), y_pred))
        # GA_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError(tf.ones_like(y_pred), y_pred))
        
        return GA_loss
    def Save_plot(self, epoch, seed=None):
        fig, axs = plt.subplots(3, 5)
        nb = 15
        
        rng = np.random.default_rng(seed=seed)
        Rand_vector = rng.normal(loc=0.0, scale=1.0, size=(nb, self.input_dim))
                 
        Image_generated = self.generator.predict(Rand_vector).reshape(nb, 28, 28, 1)
        Image_generated = (Image_generated+1)/2
        for i in range(nb):
            axs[i // 5, i % 5].imshow(Image_generated[i])
        if seed !=None:
            name = f"generated_random_images_epoch_{epoch+1}.png"
        else:
            name = f"generated_true_random_images_epoch_{epoch+1}.png"
        plt.savefig(name)
        plt.close()
        
    ####### Training #######
    def train_gan(self, ds_train, epochs, batch, n_dis):
        batch_per_epoch = len(ds_train)
        d_loss=[0, 0]
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}\n")
            step=0
            for real_images_batch in ds_train:
                step+=1
                real_images_batch = real_images_batch.numpy()
                
                rand_vect = np.random.normal(0, 1, size=(batch, self.input_dim))
                
                fake_images_batch = self.generator.predict(rand_vect, verbose=0)

                if step % n_dis == 0:
                    X_discriminator = np.concatenate([real_images_batch, fake_images_batch])
                    y_discriminator = np.concatenate([np.ones((len(real_images_batch), 1))*0.9, np.zeros((len(fake_images_batch), 1))])

                    self.discriminator.trainable = True
                    d_loss = self.discriminator.train_on_batch(X_discriminator, y_discriminator)
                    self.discriminator.trainable = False

                dummy_gan_target = np.ones((len(real_images_batch), 1))

                g_loss = self.gan_model.train_on_batch(rand_vect, dummy_gan_target)
                
                print(" " * 50, end='\r')
                print(f"  Step {step+1}/{batch_per_epoch} | Dis Loss: {d_loss[0]:.4e} | GAN Loss: Adv: {g_loss:.4e}", end='\r')

                self.GANlossHistory.append(g_loss)
                self.DislossHistory.append(d_loss)
            print(f"Epoch {epoch+1} finished. D Loss: {d_loss[0]:.4f} | GAN Total Loss: {g_loss:.4f}\n")
            if (epoch + 1) % 1 == 0:
                self.Save_plot(epoch)
                self.Save_plot(epoch, 42)
    
## Data loading
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image

Batch = 64

ds_train = ds_train.map(preprocess).shuffle(buffer_size=10000).batch(Batch).prefetch(tf.data.AUTOTUNE)

## Model Creation
new = 1
load = 0
if new:
    Model_GAN = GAN(
         256,
         [128,64, 1],
         [3,3,3],
         [2,2,1],
         [64, 128, 256],
         [3, 3, 3],
         [2, 2, 1]
         )
    
    ## Building the full model
    Model_GAN.f_Generator()
    Model_GAN.f_Discriminator()
    Model_GAN.f_FullModel(learning_rate = 5e-4, dis_learning_rate=5e-5)

elif load :
    with open("autoencoder_attributes_GAN.pkl", 'rb') as f:
        loaded_attributes = pickle.load(f)

    Model_GAN = GAN(
        x1=loaded_attributes['input_dim'],
        x2=loaded_attributes['generator_conv_t_filters'],
        x3=loaded_attributes['generator_conv_t_kernel_size'],
        x4=loaded_attributes['generator_conv_t_strides'],
        x5=loaded_attributes['discriminator_conv_t_filters'],
        x6=loaded_attributes['discriminator_conv_t_kernel_size'],
        x7=loaded_attributes['discriminator_conv_t_strides'],
        GANlossHistory=loaded_attributes['GANlossHistory'],
        DislossHistory=loaded_attributes['DislossHistory'],
    )

    # Model_GAN.f_FullModel(learning_rate = 0.0002)
    # Model_GAN.gan_model.load_weights('MyModel_GAN.keras')
    Model_GAN.generator = tf.keras.models.load_model('generator_full.keras')
    Model_GAN.discriminator = tf.keras.models.load_model('discriminator_full.keras')
    
    Model_GAN.generator_input = Input(shape=(Model_GAN.input_dim,), name='generator_input')
    Model_GAN.f_FullModel(learning_rate = 5e-4, dis_learning_rate=2e-5)
    
    Model_GAN.gan_model.summary()

Model_GAN.train_gan(ds_train,
                epochs = 200,
                batch = Batch,
                n_dis = 1
                )

## Plotting the loss
plt.plot(Model_GAN.GANlossHistory, label='GAN Loss')
plt.plot([loss[0] for loss in Model_GAN.DislossHistory], label='Discriminator Loss')
plt.plot([loss[1] for loss in Model_GAN.DislossHistory], label='Discriminator Accuracy')
plt.xlabel("Number of Batch")
plt.ylabel("Loss")
plt.legend()
plt.show()
    
## New Image generation
fig_3, axs_3 = plt.subplots(3, 5)
nb = 15
Rand_vector = np.array([np.random.normal(loc=0, scale=1, size=[1,Model_GAN.input_dim]) for i in range(nb)]).reshape(nb, Model_GAN.input_dim)
Image_generated = Model_GAN.generator.predict(Rand_vector).reshape(nb,28,28,1)
for i in range(nb):
    axs_3[i//5,i%5].imshow(Image_generated[i])

## Saving
save=1
if save:
    Model_GAN.generator.save('generator_full.keras')
    Model_GAN.discriminator.save('discriminator_full.keras') 
    Model_GAN.gan_model.save("MyModel_GAN.keras")
    attributes = {
        'input_dim':                         Model_GAN.input_dim,
        'generator_conv_t_filters':          Model_GAN.generator_conv_t_filters,
        'generator_conv_t_kernel_size':      Model_GAN.generator_conv_t_kernel_size,
        'generator_conv_t_strides':          Model_GAN.generator_conv_t_strides,
        'discriminator_conv_t_filters':      Model_GAN.discriminator_conv_t_filters,
        'discriminator_conv_t_kernel_size':  Model_GAN.discriminator_conv_t_kernel_size,
        'discriminator_conv_t_strides':      Model_GAN.discriminator_conv_t_strides,
        'GANlossHistory':                    Model_GAN.GANlossHistory,
        'DislossHistory':                    Model_GAN.DislossHistory,
    }
    with open('autoencoder_attributes_GAN.pkl', 'wb') as f:
        pickle.dump(attributes, f)

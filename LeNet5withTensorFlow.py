import tensorflow as tf

# loads and prepares MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizes data range from 0 to 1 by dividing by 255
x_train, x_test = x_train[..., tf.newaxis] / 255.0, x_test[..., tf.newaxis] / 255.0

# defines LeNet-5 architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# defines loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compiles model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# trains model
model.fit(x_train, y_train, epochs=10)

# evaluates model on test data
model.evaluate(x_test, y_test, verbose=2)
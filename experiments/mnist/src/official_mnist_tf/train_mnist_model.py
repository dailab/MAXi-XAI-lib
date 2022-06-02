import tensorflow as tf

def load_mnist():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return x_train, y_train, x_test, y_test

def init_model():
    return tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)])

def get_loss_fnc():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train_model(model, loss_fnc, x_train, y_train):
    model.compile(optimizer='adam',
        loss=loss_fnc,
        metrics=['accuracy']
    )   
    
    model.fit(x_train, y_train, epochs=5)
    
def evaluate_model(model, x_test, y_test):
    model.evaluate(x_test,  y_test, verbose=2)

def main():
    x_train, y_train, x_test, y_test = load_mnist()
    model, loss_fnc = init_model(), get_loss_fnc()
    
    train_model(model, loss_fnc, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    
    model.save_weights("/home/tuananhroman/dai/constrastive-explaination-prototype/experiments/mnist/models/tf")


if __name__ == '__main__':
    main()
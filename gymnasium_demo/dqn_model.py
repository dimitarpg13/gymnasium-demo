from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop


# neural network model for Deep Q Learning
def dqn_model(input_shape, action_space):
    x_input = Input(shape=input_shape)

    # 'Dense' is the basic form of a neural network layer
    # input layer of state size(4) and hidden layer with 512 nodes
    x = Dense(512, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform')(x_input)

    # hidden layer with 256 nodes
    x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)

    # hidden layer with 64 nodes
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)

    x = Dense(action_space, activation='linear', kernel_initializer='he_uniform')(x)

    model = Model(inputs=x_input, outputs=x, name="Cartpole DQN model")


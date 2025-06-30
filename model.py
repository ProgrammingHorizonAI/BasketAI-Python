import numpy as np
import json 

class Model:
    def __init__(self, game):
        self.ball = game.basketball
        self.player = game.player

        self.model = NeuralNetwork(10, 5, 2) # nbr d'input, nopbre de couche ça jsp, 2 sortie (velocié x et y)

    def update_input(self):
        state = np.array([
            45, 191,                                 # Coordonnées du panier pas précuse
            self.ball.pos[0], self.ball.pos[1],      # ball position
            self.ball.vel[0], self.ball.vel[1],      # ball velocity
            self.player.pos[0], self.player.pos[1],  # player position
            self.player.vel[0], self.player.vel[1],  # player velocity
        ])
        return state

    def save_model(self, filename="model.json"):
        params = self.model.get_parameters()
        params_json = {k: v.tolist() for k, v in params.items()}
        with open(filename, "w") as f:
            json.dump(params_json, f)


    def use_model(self, state):
        self.player.vel = np.array(self.model.forward(state), dtype=float)

    


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def get_parameters(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2 

    def backward(self, target, output, lr):
        error = output - target  # shape (2,)
        dW2 = np.outer(self.a1, error)  # (5, 2)
        db2 = error  # (2,)

        da1 = error @ self.W2.T  # (5,)
        dz1 = da1 * relu_deriv(self.z1)  # (5,)

        dW1 = np.outer(self.x, dz1)  # (10, 5)
        db1 = dz1  # (5,)

        # Gradient descent
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
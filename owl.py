import random as r
from neural_net import Neural_Net
import numpy as np
import copy as cp


class Owl:
    def __init__(self, net, test):
        self.piece_dict = {'p': 0, 'n': 8, 'b': 10, 'r': 12, 'q': 14, 'k': 15, 'P': 16, 'N': 24, 'B': 26, 'R': 28,
                           'Q': 30, 'K': 31}
        self.piece_max = {'p': 7, 'n': 9, 'b': 11, 'r': 13, 'q': 14, 'k': 15, 'P': 23, 'N': 25, 'B': 27, 'R': 29,
                          'Q': 30, 'K': 31}
        self.layers = [96, 45, 22, 1]
        if net == "no net":
            self.brain = Neural_Net(self.layers)
        else:
            self.brain = cp.deepcopy(net)
        print("setting up NN")
        self.l_rate = 0.1
        self.lam = 0.8
        self.test = test
        self.mini_batch_size = 100
        self.mini_batch = []

    def min_max(self, board, colour, prob, make_random, first, min_prob):
        if make_random:
            return [-1, np.random.choice(list(board.legal_moves))]
        if prob < min_prob: return [self.move_rating(board, "") * colour, '']
        if board.is_game_over():
            if board.result() == "1/2-1/2":
                return [0, '']
            if board.result() == "0-1":
                return [-colour, '']
            if board.result() == "1-0":
                return [colour, '']
        best_rating = -9999
        best_move = 0        
        moves = list(board.legal_moves)
        num = len(moves)
        for m in moves:
            board.push(m)
            temp_prob = prob / num
            score = -self.min_max(board, -colour, temp_prob, False, False, min_prob)[0]
            if score > best_rating:
                best_rating = score
                best_move = m
            board.pop()
        my_input =  ""        
        if first:            
            my_input = self.move_rating(board, True)
        return [best_rating, best_move, my_input]

    def move_rating(self, board, just_input):
        self.piece_dict = {'p': 0, 'n': 8, 'b': 10, 'r': 12, 'q': 14, 'k': 15, 'P': 16, 'N': 24, 'B': 26, 'R': 28,
                           'Q': 30, 'K': 31}
        pieces = board.piece_map()
        inputs = np.zeros((self.layers[0], 1))
        for p in pieces:
            piece_index = str(pieces[p])
            if self.piece_dict[piece_index] > self.piece_max[piece_index]: continue
            inputs[self.piece_dict[piece_index] * 3] = 1
            inputs[(self.piece_dict[piece_index] * 3) + 1] = (p // 8) / 10
            inputs[(self.piece_dict[piece_index] * 3) + 2] = (p % 8) / 10
            self.piece_dict[piece_index] += 1
        if just_input:
            return inputs
        return float(self.brain.feed_forward(inputs))

    def grad_descent(self, ratings):
        num = len(ratings)
        to_return = 0
        count = 0
        for n in range(num - 1):
            to_return += self.change_calc(ratings, num - n)
            ratings.pop(0)
        return (to_return / (num - 1))

    def change_calc(self, ratings, num):
        total_change = 0
        total_grad = 0
        for r in range(1, num):
            total_change += (ratings[r][0] - ratings[r - 1][0]) * (self.lam ** (r - 1))
        guess = ratings[0][0]
        desired_output = np.array([guess + total_change])
        inputs = ratings[0][1]
        self.mini_batch.append([inputs, desired_output])
        if len(self.mini_batch) >= self.mini_batch_size:
            self.brain.update(self.mini_batch, self.l_rate)
            self.mini_batch = []
        return total_change

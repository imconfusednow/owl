import chess
import pickle
import chess.pgn
import numpy as np
from owl import Owl
import chess.uci
import csv
import copy as cp
import datetime
import glob

search_prob = 0.0001




def train(num, number_of_moves, pid):
    reward = []
    with open('H:\\Documents\\scores.csv', mode='w') as csv_file:
        csv_file = csv.writer(csv_file)
        csv_file.writerow(["Score"])
    with open('H:\\Documents\\test.csv', mode='w') as csv_file:
        csv_file = csv.writer(csv_file)
        csv_file.writerow(["random", "player"])
    if (input("Load player?(y/n):") == "y"):
        loaded_net = load(pid)
        player = Owl(loaded_net, False)
    else:
        player = Owl("no net", False)
        pid += 1
    for j in range(num):
        file_number = np.random.randint(0, 984000)
        game_file = open("H:\\Documents\\Coding_projects\\owl\\pgns\\pgn_" + str(file_number) + ".pgn")
        position = chess.pgn.read_game(game_file)
        board = position.board()
        # print(board.unicode())
        r_moves = 0
        game_moves = list(position.mainline_moves())
        for i in range(np.random.randint(len(game_moves)) + 1):
            r_moves = i
            try:
                board.push(game_moves[i])
            except:
                print("board move failed in setup")
                print(i)
                print(chess.pgn.Game.from_board(board))
            # print("Random move ", i)
            # print(board.unicode())
        ratings = epoch([player, player], board, number_of_moves, search_prob)[0]
        if len(ratings) > 0:
            reward.append(player.grad_descent(ratings))
        else:
            print("== Ignoring game with no moves ==")
        print(j, " ", datetime.datetime.now().time().strftime("%H:%M:%S"))

        if ( j % 100 == 0 and j != 0):
            print("////////////// Testing //////////////")
            for i in range(20):                
                test_result = test(20, player, "", False)
                with open('H:\\Documents\\test.csv', mode='a', newline='') as csv_file:
                    csv_file = csv.writer(csv_file, delimiter=',')
                    csv_file.writerow([test_result[0], test_result[1]])
        if (j % 10 == 0 and j != 0):
            save(player,pid)
            with open('H:\\Documents\\scores.csv', mode='a', newline='') as csv_file:
                csv_file = csv.writer(csv_file, delimiter=',')
                csv_file.writerow(["", np.mean(reward)])
            reward = []
    save(player,pid)


def epoch(this_players, board, number_of_moves, min_prob):
    ratings = []
    for i in range(number_of_moves):
        if board.is_game_over():
            # print("game over")
            break
        ratings.append(new_turn(this_players[0], board, True, min_prob))
        # print("game move ", i, " white ")
        # print( board.unicode())
        if board.is_game_over():
            # print("game over")
            break
        ratings.append(new_turn(this_players[1], board, False, min_prob))
        # print("game move ", i, " black ")
        # print(board.unicode())
    # print("ratings ", ratings)
    return [ratings, board.result()]


def new_turn(player, board, white, min_prob):
    make_random = False
    '''if board.can_claim_threefold_repetition():
        make_random = True'''
    white = 1 if white else -1
    # print("white ", white)
    result = player.min_max(board, white, 1, make_random, True, min_prob)
    # print("min max results ", result)
    this_move = result[1]
    this_rating = result[0]
    this_input = result[2]
    board.push(this_move)
    return [this_rating * white, this_input]


def display_board(this_board):
    print(this_board.unicode())


def save(player,pid):
    save_file = open('H:\\Documents\\top_player-' + str(pid) + '-.pickle', mode='wb')
    pickle.dump(player.brain, save_file)
    print("player saved to ", save_file)
    save_file.close()

def load(this_id):
    path = 'H:\\Documents\\'
    if not (this_id):
        this_id = get_top_id()
    pickle_in = open("H:\\Documents\\top_player-" + str(this_id) + "-.pickle", "rb")
    return(pickle.load(pickle_in))

def get_top_id():
    print("Getting top id...")
    path = 'H:\\Documents\\'
    files = [f for f in glob.glob(path + "*.pickle", recursive=True)]
    top_id = 0    
    for file in files:         
        idt = file.split("-")
        if len(idt) > 1 and int(idt[1]) > top_id:
            top_id = int(idt[1])
    print("Top id: " + str(top_id))
    return int(top_id)


def inspect(board, opponent, opponent2, pid, white):
    opponent = opponent if opponent else Owl("no net", True)
    opponent2 = opponent2 if opponent2 else Owl(load(pid), True)
    for i in range(1):
        inputs = [1, 6, 6, 1, 6, 3, 1, 6, 1, 1, 5, 0, 1, 4, 5, 1, 4, 2,
                  1, 3, 4, 0, 0, 0, 1, 5, 7, 1, 4, 1, 1, 7, 2, 1, 3, 3,
                  1, 7, 0, 1, 6, 7, 1, 7, 7, 1, 6, 4, 1, 2, 4, 1, 1, 7,
                  1, 1, 6, 1, 1, 5, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1, 0,
                  1, 0, 1, 0, 0, 0, 1, 0, 5, 1, 0, 2, 1, 0, 7, 1, 0, 0,
                  1, 0, 3, 1, 0, 4]
        # one = opponent.brain.feed_forward(inputs)
        # two = opponent2.brain.feed_forward(inputs)
        one = opponent.min_max(board, white, 1, False, True, search_prob)[0]
        two = opponent2.min_max(board, white, 1, False, True, search_prob)[0]
        print("rand", one)
        print(two)
        handler = chess.uci.InfoHandler()
        engine = chess.uci.popen_engine("H:\\Documents\\Coding_projects\\stockfish-10-win\\Windows\\stockfish_10_x64.exe")
        engine.info_handlers.append(handler)
        engine.position(board)
        evaluation = engine.go(movetime=5000)
        print('evaluation value: ', handler.info["score"][1].cp/1000)
        # print(opponent2.brain.weights[2])
    board.push(np.random.choice(list(board.legal_moves)))
    inspect(board, opponent, opponent2, pid, -white)


def game(pid):
    board = chess.Board()
    opponent = load(pid)
    print("playing ", pid)
    while True:
        human_move = input("Select move")

        board.push_san(human_move)

        new_turn(opponent, board, False)

        display_board(board)


'''def test(reps, opponent, opponent2, do_print):
    test_players = []
    player_score = 0
    random_score = 0
    file_number = np.random.randint(0, 984000)
    game_file = open("H:\\Documents\\Coding_projects\\owl\\pgns\\pgn_" + str(file_number) + ".pgn")
    position = chess.pgn.read_game(game_file)
    board = position.board()
    r_moves = 0
    game_moves = list(position.mainline_moves())
    if opponent2:
        for i in range(np.random.randint(len(game_moves))):
            r_moves = i
            try:
                board.push(game_moves[i])
            except:
                print(i)
                print(chess.pgn.Game.from_board(board))
    other = opponent2 if opponent2 else Owl("no net", True)
    test_players.append(other)
    test_players.append(opponent)
    if do_print: print("Saved player playing random for ", reps, " reps")
    for i in range(reps):
        board = chess.Board()
        other = opponent2 if opponent2 else Owl("no net", True)
        test_players = [other, opponent]
        np.random.shuffle(test_players)
        result = epoch(test_players, board, 100, search_prob * 10)[1]
        if result == "1-0":
            if test_players[0].test:
                random_score += 1
            else:
                player_score += 1                
        if result == "0-1":
            if test_players[1].test:
                random_score += 1
            else:
                player_score += 1                
        print("p score ", player_score, "r score ", random_score)
    if do_print:
        first_player = "Saved player"
        second_player = "Random"
        if test_players[0].test:
            first_player = "Random"
            second_player = "Saved player"
        print("Last game: " + first_player + " VS " + second_player + " after ")
        print(chess.pgn.Game.from_board(board))
    return [random_score, player_score]'''

def test(reps, opponent, opponent2, do_print):
    test_players = []
    player_score = 0
    random_score = 0
    file_number = np.random.randint(0, 984000)
    game_file = open("H:\\Documents\\Coding_projects\\owl\\pgns\\pgn_" + str(file_number) + ".pgn")
    position = chess.pgn.read_game(game_file)
    board = position.board()
    handler = chess.uci.InfoHandler()
    engine = chess.uci.popen_engine("H:\\Documents\\Coding_projects\\stockfish-10-win\\Windows\\stockfish_10_x64.exe")
    engine.info_handlers.append(handler)
    r_moves = 0
    game_moves = list(position.mainline_moves())    
    for i in range(np.random.randint(len(game_moves))):
        r_moves = i
        try:
            board.push(game_moves[i])
        except:
            print(i)
            print(chess.pgn.Game.from_board(board))
    player_score = opponent.min_max(board, 1, 1, False, True, search_prob)[0]
    engine.position(board)
    evaluation = engine.go(movetime=5000)
    if handler.info["score"][1].cp == None: return [0,0]
    print(handler.info["score"][1].cp/1000, player_score)
    return [handler.info["score"][1].cp/1000, player_score]


pid = get_top_id()

mode = input("Mode? Training (t) or Game (g) or Test (te)")
if mode == "t":
    reps = int(input("Number of epoches: "))
    moves_per_rep = int(input("Moves per epoche: "))
    train(reps, moves_per_rep, pid)
elif mode == "te":
    temp_id = input("Player id to load: ")
    temp_id = temp_id if temp_id else pid
    opponent = load(temp_id)
    reps = int(input("Number of epoches: "))
    test(reps, opponent, "", True)
elif mode == "g":
    temp_id = input("Player id to load: ")
    temp_id = temp_id if temp_id else pid
    game(temp_id)
else:
    print("Inspect mode")
    temp_id = input("Player id to load: ")
    temp_id = temp_id if temp_id else pid
    inspect(chess.Board(), False, False, temp_id, 1)

import kerasfrom keras.models import Sequentialfrom keras.layers import Dense, Flatten, Activationfrom keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropoutfrom keras.optimizers import Adamfrom keras.models import load_modelfrom keras.callbacks import ModelCheckpoint, EarlyStoppingimport randomimport numpy as npimport warningswarnings.filterwarnings("ignore")class FindAWay:    def __init__(self):        self.LEFT = "LEFT"  # 0        self.RIGHT = "RIGHT"  # 1        self.UP = "UP"  # 2        self.DOWN = "DOWN"  # 3        self.MOVES = [self.LEFT, self.RIGHT, self.UP, self.DOWN]        self.MOVE = {self.LEFT: (0, -1), self.RIGHT: (0, 1), self.UP: (-1, 0), self.DOWN: (1, 0)}        self.CURRENT_CELL = 1        self.BLOCKED_CELL = -1        self.EMPTY_CELL = 0        self.GRID_SIZE = 5        self.N_CLASSES = 4        self.LEARNING_RATE = 1e-3        self.VALID_MOVES_INFO_STORE = {}        self.STATE_RESPONSES_INFO_STORE = {}        self.GAMES_STORE = {}        self.MODEL_NAME = "find-a-way.hdf5"        self.EPOCHS = 50        self.BATCH_SIZE = 32    def get_features(self, grid):        return np.int32([grid.flatten()])    def search_grid(self, grid, val):        loc = np.where(grid == val)        return (loc[0][0], loc[1][0])    def initialize_grid(self, grid_size=None, current_cell=(1, 0), blocked_cells=None):        if grid_size is None: grid_size = self.GRID_SIZE        if blocked_cells is None: blocked_cells = [(0, 0),(0, self.GRID_SIZE - 1)]        grid = np.full((grid_size, grid_size), self.EMPTY_CELL, dtype=np.int32)        for blocked_cell in blocked_cells:            grid[blocked_cell[0]][blocked_cell[1]] = self.BLOCKED_CELL        grid[current_cell[0]][current_cell[1]] = self.CURRENT_CELL        return grid    def validate_move(self, grid, current_loc, move):        grid = grid.copy()        next_loc = (current_loc[0] + move[0], current_loc[1] + move[1])        if next_loc[0] < 0 or next_loc[1] < 0 or next_loc[0] == len(grid) or next_loc[1] == len(grid):            return False, current_loc, grid        if grid[next_loc[0]][next_loc[1]] != self.EMPTY_CELL:            return False, current_loc, grid        grid[current_loc[0]][current_loc[1]] = self.BLOCKED_CELL        grid[next_loc[0]][next_loc[1]] = self.CURRENT_CELL        return True, next_loc, grid    def is_game_over(self, grid):        if self.is_game_won(grid):            return True        valid_move = False        for possible_movement in self.MOVES:            possible_move = self.MOVE[possible_movement]            move_validity, _, _ = self.validate_move(grid=grid,                                                current_loc=self.search_grid(grid=grid,                                                                        val=self.CURRENT_CELL), move=possible_move)            valid_move = valid_move or move_validity        return not valid_move    def fetch_n_empty_cells(self, grid):        return (grid == self.EMPTY_CELL).sum()    def fetch_n_non_empty_cells(self, grid):        n_empty_cells = self.fetch_n_empty_cells(grid)        return np.size(grid) - n_empty_cells    def fetch_n_blocked_cells(self, grid):        return (grid == self.BLOCKED_CELL).sum()    def is_game_won(self, grid):        if not np.any(grid == self.EMPTY_CELL):            return True    def is_game_valid(self, grid):        if self.is_game_over(grid):            if self.is_game_won(grid):                return True            return False        f_str = str(self.get_features(grid))        validity_result = {}        result = False        #if f_str in self.VALID_MOVES_INFO_STORE: validity_result = self.VALID_MOVES_INFO_STORE[f_str]        for move in self.MOVES:            if move not in validity_result:                valid_move, next_loc, next_grid = self.validate_move(grid=grid,                                                                current_loc=self.search_grid(grid=grid,                                                                                        val=self.CURRENT_CELL),                                                                move=self.MOVE[move])                validity_result[move] = valid_move and self.is_game_valid(next_grid)            result = result or validity_result[move]        #self.VALID_MOVES_INFO_STORE[f_str] = validity_result        return result    def create_all_game_grids(self, grid_size, n_blocked_cells, games_count=10):        games = []        n_tries = 0        while len(games) < games_count and n_tries < (10 * games_count):            grid_arr = np.full((grid_size * grid_size), self.EMPTY_CELL, dtype=np.int32)            rem_indxs = [i for i in range(grid_size * grid_size)]            chosen_indx = random.sample(rem_indxs, n_blocked_cells)            for indx in chosen_indx: grid_arr[indx] = self.BLOCKED_CELL            for indx in range(grid_size * grid_size):                grid_arr_copy = grid_arr.copy()                if grid_arr_copy[indx] == self.EMPTY_CELL:                    grid_arr_copy[indx] = self.CURRENT_CELL                    grid_arr_copy = grid_arr_copy.reshape((grid_size, grid_size))                    f_str = str(self.get_features(grid_arr_copy))                    if f_str in self.GAMES_STORE: continue                    if self.is_game_valid(grid_arr_copy):                        games.append(grid_arr_copy)                        self.GAMES_STORE[f_str] = True                        print("Selected Game: ", f_str)                        break                    else: self.GAMES_STORE[f_str] = False        return games    def find_valid_game_grids(self, grid_size, games_count=10):        games = []        n_blocked_cells_min = 2        n_blocked_cells_max = np.int32((grid_size * grid_size) / 2)        n_game_search_tries = 0        while len(games) < games_count:            n_blocked_cells = np.random.randint(n_blocked_cells_min, n_blocked_cells_max + 1)            sub_games = self.create_all_game_grids(grid_size, n_blocked_cells, games_count=games_count)            if n_blocked_cells % 2 == 1: games = games + sub_games            else: games = sub_games + games            n_game_search_tries += 1            if n_game_search_tries > (10 * games_count): break        return games    def simulate(self, grid, features, labels):        if self.is_game_won(grid=grid): return 1        f_str = str(self.get_features(grid))        if f_str in self.STATE_RESPONSES_INFO_STORE:            responses = self.STATE_RESPONSES_INFO_STORE[f_str]["responses"]            if np.sum(responses) == 0: return 0            features.append(grid)            labels.append(responses)            return 1        responses = np.array([0, 0, 0, 0])        for move_indx in range(len(self.MOVES)):            possible_movement = self.MOVES[move_indx]            possible_move = self.MOVE[possible_movement]            move_validity, _, next_grid = self.validate_move(grid=grid,                                                current_loc=self.search_grid(grid=grid,                                                                        val=self.CURRENT_CELL), move=possible_move)            if move_validity:                responses[move_indx] = self.simulate(grid=next_grid, features=features, labels=labels)        state_responses_obj = {}        state_responses_obj["state"] = grid        state_responses_obj["responses"] = responses        self.STATE_RESPONSES_INFO_STORE[f_str] = state_responses_obj        if np.sum(responses) == 0: return 0        features.append(grid)        labels.append(responses)        return 1    def load_model(self, show_summary=False):        try:            model = load_model(self.MODEL_NAME)        except:            model = None        if model is None:            inputShape = (self.GRID_SIZE, self.GRID_SIZE, 1)            model = Sequential()            model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(Dropout(0.25))            model.add(Conv2D(128, (3, 3), padding="same"))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(Conv2D(128, (3, 3), padding="same"))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(MaxPooling2D(pool_size=(2, 2)))            model.add(Dropout(0.25))            model.add(Conv2D(256, (3, 3), padding="same"))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(Conv2D(256, (3, 3), padding="same"))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(MaxPooling2D(pool_size=(2, 2)))            model.add(Dropout(0.25))            model.add(Conv2D(128, (3, 3), padding="same"))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(Conv2D(128, (3, 3), padding="same"))            model.add(Activation("relu"))            model.add(BatchNormalization(axis=-1))            model.add(Dropout(0.25))            model.add(Flatten())            model.add(Dense(1024))            model.add(Activation("relu"))            model.add(BatchNormalization())            model.add(Dropout(0.5))            model.add(Dense(256))            model.add(Activation("relu"))            model.add(BatchNormalization())            model.add(Dropout(0.5))            model.add(Dense(self.N_CLASSES, activation='sigmoid'))        else:            print(self.MODEL_NAME, " is restored.")        if show_summary:            model.summary()        adam = Adam(lr=self.LEARNING_RATE, decay=self.LEARNING_RATE / self.EPOCHS)        model.compile(loss='binary_crossentropy',                      optimizer=adam,                      metrics=['accuracy'])        return self.MODEL_NAME, model    def train_model(self, model_name, model, features, labels, verbose=1):        callbacks = [ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)]        features = np.reshape(features, (-1, self.GRID_SIZE, self.GRID_SIZE, 1))        labels = np.reshape(labels, (-1, self.N_CLASSES))        _ = model.fit(features, labels,                            batch_size=self.BATCH_SIZE,                            epochs=self.EPOCHS,                            verbose=verbose,                            validation_data=(features, labels),                            callbacks=callbacks                            )        return model_name, model    def predict(self, model_name, model, features):        features = np.reshape(features, (-1, self.GRID_SIZE, self.GRID_SIZE, 1))        return model.predict(features)    # returns loss and accuracy    def evaluate(self, model_name, model, features, labels):        features = np.reshape(features, (-1, self.GRID_SIZE, self.GRID_SIZE, 1))        labels = np.reshape(labels, (-1, self.N_CLASSES))        score = model.evaluate(features, labels, verbose=1)        return score[0], score[1]
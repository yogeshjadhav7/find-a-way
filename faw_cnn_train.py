
# coding: utf-8

# In[1]:


import numpy as np
from faw import FindAWay


# In[2]:


faw = FindAWay()


# In[3]:


N_EPISODES = 100

n_blocked_cells_min = 2

for n_episode in range(N_EPISODES):
    print("\n\nPlaying episode #" + str(n_episode))
    features = []
    labels = []
    games_counter = 0
    games_count = 25
    
    while games_counter < games_count: 
        seed_game = faw.random_initialize_grid()
        games_counter += faw.create_all_game_grids(seed_grid=seed_game, grid_size=faw.GRID_SIZE, features=features, labels=labels, games_count=games_count)
        print("games_counter", games_counter)
    
    model_name, model = faw.load_model()
    model_name, model = faw.train_model(model_name=model_name, model=model, features=features, labels=labels, verbose=0)
    loss, accuracy = faw.evaluate(model_name=model_name, model=model, features=features, labels=labels)
    print("Loss", loss)
    print("Accuracy", accuracy)


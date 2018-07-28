
# coding: utf-8

# In[1]:


import numpy as np
from faw import FindAWay


# In[ ]:


faw = FindAWay()


# In[ ]:


N_EPISODES = 100
for n_episode in range(N_EPISODES):
    print("\nPlaying episode #" + str(n_episode))
    features = []
    labels = []
    games = faw.find_valid_game_grids(grid_size=faw.GRID_SIZE, games_count=10)
    for game_indx in range(len(games)): 
        game = games[game_indx]
        faw.simulate(grid=game, features=features, labels=labels)
        #if game_indx % 5 == 0: faw.STATE_RESPONSES_INFO_STORE = {}

    #faw.STATE_RESPONSES_INFO_STORE = {}
    #faw.VALID_MOVES_INFO_STORE = {}
    #faw.GAMES_STORE = {}
    
    model_name, model = faw.load_model()
    model_name, model = faw.train_model(model_name=model_name, model=model, features=features, labels=labels, verbose=0)
    loss, accuracy = faw.evaluate(model_name=model_name, model=model, features=features, labels=labels)
    print("Loss", loss)
    print("Accuracy", accuracy)


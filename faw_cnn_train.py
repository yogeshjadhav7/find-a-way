
# coding: utf-8

# In[1]:


import numpy as np
from faw import FindAWay


# In[2]:


faw = FindAWay()


# In[ ]:


N_EPISODES = 100
N_GAMES_PER_EPISODE = 100
verbose = 0

for n_episode in range(N_EPISODES):
    print("\n\n\nN_EPISODE: ", n_episode)
    model_name, model = faw.load_model()
    features = []
    labels = []
    lost = 0
    won = 0
    for n_game in range(N_GAMES_PER_EPISODE):
        grid = faw.random_initialize_grid()
        trackrecord = []
        faw.simulate(grid=grid, model=model, trackrecord=trackrecord, verbose=verbose)
        (final_grid, _, _)  = trackrecord[len(trackrecord) - 1]
        
        if faw.is_game_won(grid=final_grid): won += 1
        else: lost += 1
        
        if verbose == 1: print(final_grid)
        features_game, labels_game = faw.extract_features_labels(final_grid=final_grid, trackrecord=trackrecord)
        if len(features_game) == 0: continue
                
        if len(features) == 0:
            features = features_game
            labels = labels_game
        else:
            features = np.concatenate((features, features_game), axis=0)
            labels = np.concatenate((labels, labels_game), axis=0)
            
    print("won: ", won, " lost: ", lost)
    print("Training model on " + str(len(features)) + " instances...")
    _, model = faw.train_model(model=model, features=features, labels=labels, verbose=verbose)
    loss, accuracy = faw.evaluate(model=model, features=features, labels=labels, verbose=verbose)
    print("loss: ", loss)
    print("accuracy: ", accuracy)
    


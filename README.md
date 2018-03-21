# find-a-way
Simulator for playing popular android game "Find-A-Way" using deep learning (MLP) and Bellman's equation.


## Jupyter Notebook's cell wise explanation of the approach:

### Cell 1:
    contains the hyperparamenters of the Q-learning technique as well as game play configurations
    
### Cell 2:
    contains the hypermeters for deep MLP network. 
    
### Cell 3:
    contains the utility methods which helps in carrying out the game play efficiently.
    
### Cell 4:
    contains the methods which optimally search and initiate valid game grid using dynamic programming.
    
### Cell 5 & 6:
    sets up the model object and game play by getting the list of game grids to simulate and train the model. Here, 40 unique valid games are loaded into the list for simulation serially.
        
### Cell 7:
    Out of the 40 games, 30 are used for training and 10 are used for validating after the training process. Q_EXPLORATION_PROBABILITY helps to introduce randomness while training which is important to make the model explore other paths explicityly. Q_EXPLORATION_PROBABILITY is replaced by 1 in validation mode so that model strictly uses its own brains of what it has learned so far.
    
    Each game is attempted till the game gets 100% gain (finds the optimal way) from the play or the tries count reaches 15. In validation mode, this trial enforcement is switched off to organically track how well the model has been trained.
    
### Cell 8:
    This shows the performance of the trained model on the validation games. The count of invalid moves drastically decrease towards the end of the validation process, and also the gain of 100% starts to become frequent without trial enforcement. 
    
## So the model who was completely unaware how to play this game / rules of the game, starts to become pretty good as it trains on the game play ONLY by reinforcing rewards depending upon its performance in each subsequent game play simulation :)
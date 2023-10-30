python -m train.jepa.proteins device 0 # Default training on proteins on gpu 0

# Note that if you wish to use these objectives you have to modify the model code
# In the current version full automatic parametrization is not supported
# More details can be found in the GraphJepa model inside 'core/model.py'

python -m train.mutag jepa.dist 1 device 1 # Use euclidean objective on mutag on gpu 1
python -m train.jepa.zinc jepa.dist 2 model.hidden_size 32 device 0 # use poincaré embedding distance with smaller latent dimension
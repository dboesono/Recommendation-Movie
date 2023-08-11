import numpy as np
import rbm
import projectLib as lib

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

allUsersRatings = lib.getAllUsersRatings(trStats["u_users"], training)

# SET PARAMETERS HERE!!!
# number of hidden units
# F
# epochs
# gradientLearningRate
K = 5
F = 20
epochs = 100
learning_rate = 0.02
momentum_coeff = 0.2
reg_coeff = 0.1
es = True
dlr = True

# Initialise the best RMSE and corresponding parameters
lowest_rmse = float('inf')
opt_params = None

epochs_list = []
training_losses = []
validation_losses = []
# Unpack the parameters
params = F, epochs, learning_rate, momentum_coeff, reg_coeff 
# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
grad = np.zeros(W.shape)
b_v = rbm.getInitialBiasVisible(trStats["n_movies"], K)
b_h = rbm.getInitialBiasHidden(F)
grad_bh = np.zeros(b_h.shape)
grad_bv = np.zeros(b_v.shape)
momentum_W = np.zeros([trStats["n_movies"], F, K])
momentum_val_bh = np.zeros(b_h.shape)
momentum_val_bv = np.zeros(b_v.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)
# mini-batch size
batch_size= 1

for epoch in range(1, epochs+1):
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder)
    batch_count = 0 # used for mini-batch
    for user in visitingOrder:
        batch_count += 1
        # get the ratings of that user
        ratingsForUser = allUsersRatings[user]
        
        # build the visible input
        v = rbm.getV(ratingsForUser)

        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :]
        bvForUser = rbm.getInitialBiasVisible(trStats["n_movies"], K)[ratingsForUser[:, 0], :]

        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, b_h)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser, b_v)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, b_h)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        grad[ratingsForUser[:, 0], :, :] = learning_rate * (posprods[ratingsForUser[:, 0], :, :] - negprods[ratingsForUser[:, 0], :, :])

        # gradient descent algorithm for the biases
        grad_bv[ratingsForUser[:, 0], :] = v - negData
        grad_bh = posHiddenProb - negHiddenProb

        # moementum
        if momentum_coeff != 0:
            momentum_W[ratingsForUser[:, 0], :, :] = momentum_coeff * momentum_W[ratingsForUser[:, 0], :, :] + (1- momentum_coeff) * grad[ratingsForUser[:, 0], :, :]
            grad[ratingsForUser[:, 0], :, :] = momentum_W[ratingsForUser[:, 0], :, :]
            momentum_val_bh = momentum_coeff * momentum_val_bh + (1-momentum_coeff) * grad_bh
            grad_bh = momentum_val_bh
            momentum_val_bv[ratingsForUser[:, 0], :] = momentum_coeff * momentum_val_bv[ratingsForUser[:, 0], :] + (1 - momentum_coeff) * grad_bv[ratingsForUser[:, 0], :]
            grad_bv[ratingsForUser[:, 0], :] = momentum_val_bv[ratingsForUser[:, 0], :]

        # regularization
        if reg_coeff != 0:
            grad[ratingsForUser[:, 0], :, :] += - learning_rate * reg_coeff/2 * W[ratingsForUser[:, 0], :, :]
        if batch_count == batch_size:
            W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :] / batch_size
            b_v[ratingsForUser[:, 0], :] += learning_rate * grad_bv[ratingsForUser[:, 0], :] / batch_size
            b_h += learning_rate * grad_bh / batch_size
            batch_count = 0
            
    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, allUsersRatings, b_h, b_v)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, allUsersRatings, b_h, b_v)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    # Print the RMSE and parameters for this iteration
    print("### Parameters: hidden_units = %d, epoch = %d, learning_rate = %f" % (F, epoch, learning_rate))
    print("Validation loss = %f" % vlRMSE)
    print('Training Loss = %f' % trRMSE)
    epochs_list.append(epoch)
    training_losses.append(trRMSE)
    validation_losses.append(vlRMSE)
    # Update the best RMSE and corresponding parameters if necessary
    if vlRMSE < lowest_rmse:
        lowest_rmse = vlRMSE
        opt_params = params
    # decrease lr
    if dlr:
        learning_rate = learning_rate * lib.decrease_lr(validation_losses, 6, 2, 0.03, 0.05)
    # early stop 
    if es:
        if lib.early_stopping(validation_losses, 10, 3, 0.005):
            break
            
# Print the best parameters and corresponding RMSE
print("Best validation loss = %f" % lowest_rmse)
lib.plot_loss(epochs_list,training_losses,validation_losses)
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
predictedRatings = np.array([rbm.predictForUser(user, W, allUsersRatings, b_h, b_v) for user in trStats["u_users"]])
rdm = lib.generate_raw_data(1000,1667,allUsersRatings)
predictedRatings = lib.neighbourhood(rdm,predictedRatings, 1000, 1667, 5)
np.savetxt("predictedRatings.txt", predictedRatings)
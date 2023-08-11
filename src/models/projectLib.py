import numpy as np
import matplotlib.pyplot as plt

def getTrainingData():
    DATA_PATH = "../data/interim/training.csv"
    return np.genfromtxt(DATA_PATH, delimiter=",", dtype=np.int)

def getValidationData():
    DATA_PATH = "../data/interim/validation.csv"
    return np.genfromtxt(DATA_PATH, delimiter=",", dtype=np.int)

def getUsefulStats(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()

    users = [x[1] for x in training]
    u_users = np.unique(users).tolist()

    return {
        "movies": movies, # movie IDs
        "u_movies": u_movies, # unique movie IDs
        "n_movies": len(u_movies), # number of unique movies

        "users": users, # user IDs
        "u_users": u_users, # unique user IDs
        "n_users": len(u_users), # number of unique users

        "ratings": [x[2] for x in training], # ratings
        "n_ratings": len(training) # number of ratings
    }


def getAllUsersRatings(users, training):
    return [getRatingsForUser(user, training) for user in np.sort(users)]


def getRatingsForUser(user, training):
    # user is a user ID
    # training is the training set
    # ret is a matrix, each row is [m, r] where
    #   m is the movie ID
    #   r is the rating, 1, 2, 3, 4, or 5
    ret = []
    for x in training:
        if x[1] == user:
            ret.append([x[0], x[2]])
    return np.array(ret)

# RMSE function to tune your algorithm
def rmse(r, r_hat):
    r = np.array(r)
    r_hat = np.array(r_hat)
    return np.linalg.norm(r - r_hat) / np.sqrt(len(r))

def early_stopping(ref_list, ref_size, test_size, improve_threshold):
    # ref_size: length of all numbers used.
    ref_list_len = len(ref_list)
    if ref_list_len < ref_size:
        return False
    ref_array = np.array(ref_list)
    tested_array = ref_array[-test_size:]
    pure_ref_array = ref_array[-ref_size:]
    tested_avg = np.mean(tested_array)
    pure_ref_avg = np.mean(pure_ref_array)
    if pure_ref_avg - tested_avg <= improve_threshold:
        return True
    else:
        return False

def decrease_lr(ref_list, ref_size, test_size, improve_threshold, decrease_rate):
    ref_list_len = len(ref_list)
    if ref_list_len < ref_size:
        return 1
    ref_array = np.array(ref_list)
    tested_array = ref_array[-test_size:]
    pure_ref_array = ref_array[-ref_size:]
    tested_avg = np.mean(tested_array)
    pure_ref_avg = np.mean(pure_ref_array)
    if pure_ref_avg - tested_avg <= improve_threshold:
        return decrease_rate
    else:
        return 1

def generate_raw_data(num_users,num_movies,allUsersRatings):
    # Get raw data matrix from training data
    M = [[0] * num_movies for _ in range(num_users)]
    for i, user in enumerate(allUsersRatings):
        for m, r in user:
            M[i][m] = r
    R = np.array(M)
    return R

def neighbourhood(R,R_baseline,num_users,num_movies,k=5):
    # Neighbourhood method
    R_tilde = (R - R_baseline) * (R > 0)
    D = np.zeros((num_movies, num_movies))
    epsilon = 1e-9 
    for i in range(num_movies):
        for j in range(num_movies):
            idx = np.logical_and(R[:, i] > 0, R[:, j] > 0)
            vi = R_tilde[idx, i]
            vj = R_tilde[idx, j]
            numerator = np.sum(vi * vj)
            denominator = np.sqrt(np.sum(vi * vi) * np.sum(vj * vj)) + epsilon
            D[i, j] = numerator / denominator

    R_neighborhood = np.zeros((num_users, num_movies))
    k = k
    for j in range(num_movies):
        val_sort = np.argsort(np.abs(D[j, :]))[::-1]
        idx = val_sort[0:k+1] 
        idx = idx[idx != j]
        for i in range(num_users):
            idx2 = idx.copy()
            idx2 = idx2[R[i, idx2] != 0]
            if len(idx2) > 0:
                R_neighborhood[i, j] = np.sum(D[j, idx2] * R_tilde[i, idx2]) / np.sum(np.abs(D[j, idx2]))
                
    R_neighborhood = R_baseline + R_neighborhood
    R_neighborhood[R_neighborhood > 5] = 5
    R_neighborhood[R_neighborhood < 1] = 1
    return R_neighborhood

def plot_loss(epoch_list,training_loss_list,validation_loss_list):
    plt.plot(epoch_list, training_loss_list, label='Training Loss')
    plt.plot(epoch_list, validation_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title('Training and Validation Loss by Epoch')
    plt.legend()
    # plt.savefig('loss_plot.png',bbox_inches='tight')
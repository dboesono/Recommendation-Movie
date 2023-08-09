import numpy as np

def getTrainingData():
    DATA_PATH = "../data/interim/training.csv"
    return np.genfromtxt(DATA_PATH, delimiter=",", dtype=np.int)[1:]

def getValidationData():
    DATA_PATH = "../data/interim/validation.csv"
    return np.genfromtxt(DATA_PATH, delimiter=",", dtype=np.int)[1:]


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
    #   r is the rating, 1, 2, 3, 4 or 5
    return np.array([[x[0], x[2]] for x in training if x[1] == user])

# RMSE function to tune your algorithm
def rmse(r, r_hat):
    r = np.array(r)
    r_hat = np.array(r_hat)
    return np.linalg.norm(r - r_hat) / np.sqrt(len(r))
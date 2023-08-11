import numpy as np
import src.models.projectLib as lib
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        # Initialize the mean rating, b parameter, and training statistics
        self.rBar = None
        self.b = None
        self.trStats = None
        # Create a linspace for regularization parameter and initialize RMSE lists
        self.x = np.linspace(0, 20, 20)
        self.y1 = []
        self.y2 = []

    def fit(self, training, validation, regularized=True):
        # Extract useful statistics from training and validation data
        self.trStats = lib.getUsefulStats(training)
        vlStats = lib.getUsefulStats(validation)
        # Calculate the mean rating
        self.rBar = np.mean(self.trStats["ratings"])
        # Get matrix A and vector c
        A = self._getA(self.trStats)
        c = self._getc(self.rBar, self.trStats["ratings"])

        # Loop through regularization parameters and fit the model
        for l in self.x:
            if regularized:
                self.b = self._param_reg(A, c, l)
            else:
                self.b = self._param(A, c)
            print("Linear regression, l = %f" % l)
            # Calculate RMSE for training and validation
            rmse_1 = lib.rmse(self.predict(self.trStats["movies"], self.trStats["users"]), self.trStats["ratings"])
            self.y1.append(rmse_1)
            print("RMSE for training %f" % rmse_1)
            rmse_2 = lib.rmse(self.predict(vlStats["movies"], vlStats["users"]), vlStats["ratings"])
            print("RMSE for validation %f" % rmse_2)
            self.y2.append(rmse_2)

    def predict(self, movies, users):
        # Predict ratings for given movies and users
        n_predict = len(users)
        p = np.zeros(n_predict)
        for i in range(0, n_predict):
            # Calculate rating using the learned parameters
            rating = self.rBar + self.b[movies[i]] + self.b[self.trStats["n_movies"] + users[i]]
            if rating > 5: rating = 5.0
            if rating < 1: rating = 1.0
            p[i] = rating
        return p

    # Plotting function to visualize the RMSE vs regularization parameter
    def plot(self):
        # Find the index of the minimum RMSE in the validation RMSE list
        min_rmse_index = np.argmin(self.y2)

        # Find the corresponding regularization parameter (lambda) using the index
        best_lambda = self.x[min_rmse_index]

        # Output the best parameter
        print("Best λ value: ", best_lambda)
        print("Lowest RMSE for validation: ", self.y2[min_rmse_index])

        # Plot validation and training loss
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.plot(self.x, self.y2, label='Validation', linewidth=2, marker='o', markersize=5, linestyle=':', color='red')
        plt.plot(self.x, self.y1, label='Training', linewidth=2, marker='X', markersize=5, linestyle=':', color='blue')
        plt.legend(fontsize=12)  # Set legend font size
        plt.xlabel('Regularization Parameter (λ)', fontsize=14)  # Set x-axis label
        plt.ylabel('RMSE', fontsize=14)  # Set y-axis label
        plt.title('RMSE vs Regularization Parameter', fontsize=16)  # Set plot title
        plt.xticks(fontsize=12)  # Set x-axis ticks font size
        plt.yticks(fontsize=12)  # Set y-axis ticks font size
        plt.grid(True)  # Enable grid
        # plt.savefig('lr_plot.png')
        plt.show()

    # Private methods for internal calculations
    def _getA(self, trStats):
        A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
        temp_movie = trStats['movies']
        temp_user = trStats['users']
        for i in range(trStats["n_ratings"]):
            A[i][temp_movie[i]] = 1
            A[i][temp_user[i] + trStats["n_movies"]] = 1
        return A

    def _getc(self, rBar, ratings):
        ratings_array = np.array(ratings)
        c = ratings_array - rBar
        return c

    def _param(self, A, c):
        A_T = A.transpose()
        b = np.linalg.pinv(A_T @ A) @ c
        return b

    def _param_reg(self, A, c, l):
        A_T = A.transpose()
        I = np.eye((A_T @ A).shape[0])
        b = np.linalg.inv(A_T @ A + l * I) @ (A_T @ c)
        return b

# # Example usage
# def main():
#     training = lib.getTrainingData()
#     validation = lib.getValidationData()
#     model = BaselinePredictor()
#     model.fit(training, validation)
#     model.plot()
    

# if __name__ == "__main__":
#     main()
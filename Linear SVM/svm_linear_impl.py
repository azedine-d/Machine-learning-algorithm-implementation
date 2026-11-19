import numpy as np

class MySVMLinear:
    def __init__(self,x,y,lambda_=0.01,learning_rate=0.01):
        self.x = x
        self.y = y
        self.w = np.zeros(self.x.shape[1])
        self.b = 0
        self.losses = np.array([])
        self.instance_number = len(self.x)
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

        if len(self.x)!= len(self.y):
            raise ValueError("X and y must have the same length")

        if  not np.isin(self.y,[-1,1]).all() :
            raise ValueError("Y's value must be 1 or -1")

        if np.isnan(self.x).any()  or np.isnan(self.y).any()  :
            raise ValueError("X and Y must not contain NaN")

        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("X and y must contain only finite values")

    def decision_function(self,i):
        return np.dot(self.w,self.x[i]) + self.b

    def hinge_loss_function(self,i):
        loss = max (0 , 1 - self.y[i]*self.decision_function(i))
        return loss


    def average_loss(self):
        self.losses = np.array([])
        for i in range(self.instance_number):
            self.losses = np.append(self.losses,self.hinge_loss_function(i))
        average = np.mean(self.losses)
        return average


    def regularization(self):
        return (self.lambda_/2 ) * np.dot(self.w,self.w)

    def objective_function(self):
        return self.regularization() + self.average_loss()


    def gradient_w(self):
        gradient_sum = 0
        for i in range(self.instance_number):

            x_i = self.x[i]
            y_i = self.y[i]

            margin = y_i * (np.dot(self.w,x_i)+self.b)
            if margin < 1 :
                gradient_sum = gradient_sum - x_i * y_i

        return self.lambda_ * self.w + 1/self.instance_number * gradient_sum


    def gradient_b(self):
        gradient_sum = 0
        for i in range(self.instance_number):
            x_i = self.x[i]
            y_i = self.y[i]
            margin = y_i * (np.dot(self.w, x_i) + self.b)
            if margin < 1 :
                gradient_sum = gradient_sum - y_i
        return (1/self.instance_number ) * gradient_sum


    def weights_update(self):
        self.w = self.w - self.learning_rate * self.gradient_w()
        self.b = self.b - self.learning_rate * self.gradient_b()


    def fit(self,epochs = 1000 ,tolerance = 0.001 ):
        j_old = self.objective_function()
        for i in range(epochs):
            self.weights_update()
            j_new = self.objective_function()
            if abs(j_old - j_new) < tolerance :
                break
            j_old = j_new


    def predict_decision_function(self, x_new):
        return np.dot(x_new, self.w) + self.b

    def predict(self, x_new):
       scores = self.predict_decision_function(x_new)
       return np.where(scores >= 0, 1, -1)

























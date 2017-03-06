#Ridge Regression with Gradient Descent & using Closed form solution
#Lp-Norm Regularized Regression using Gradient Descent Algorithm

import numpy as np
from numpy import *

                       
def main():
    train_path="/home/bodhisattwa/Desktop/ML_Data/train.csv"
    test_path="/home/bodhisattwa/Desktop/ML_Data/test.csv"
    train_data = genfromtxt(train_path,delimiter=',')
    test_data = genfromtxt(test_path,delimiter=',')

    #Populating the basis vector with features
    phi_train = train_data[1:,1:14]
    phi_test = test_data[1:,1:14]

    y_train = train_data[1:,[14]]
    

    #Normalizing the overall train+test data
    phi_augmented = np.concatenate((phi_train,phi_test),axis=0)
    phi_normalized = normalize_data(phi_augmented)
    train_rows =np.shape(phi_train)
    phi_train = phi_normalized[0:(train_rows[0]),:]
    phi_test = phi_normalized[(train_rows[0]):,:]


    #plt.plot(phi_train[:,[12]],y_train,"o")
    #plt.show()
    
 

    #Generating randomized training sets
    folds = 5
    augmented = np.concatenate((phi_train,y_train),axis =1)
    augmented_set = random_subset_gen(augmented,folds)

    #hyperparameters initialization
    iterations = 10000
    step_length = 0.0001
    lamb = [10,5,2,8,4]
    index_train_part = np.random.choice(range(0,folds),size = folds,replace=False)
    
    w_star=[]
    mse =[]
    

    #Linear Regression
    for i in range(0,folds):
        train,test =train_validation_split(augmented_set,index_train_part[i])
        w,m = ridge_regression(test,iterations,step_length,lamb[i])
        w_star.append(w)
        mse.append(m)


    
    #Finding the lambda with lowest error
    min_lamb_index = mse.index(min(mse))

    #Final prediction using Gradient Descent
    print("Shrinkage Parameter for Gradient Descent",lamb[min_lamb_index])
    phi_test_final = np.ones((np.shape(phi_test)[0],np.shape(phi_train)[1]))
    phi_test_final = phi_test
    y_predicted = predictor(phi_test_final,w_star[min_lamb_index])
    print("Prediction Results for Gradient Descent")
    print(y_predicted)

    #Saving output of Penalized Ridge Regression using Grad. Descent in CSV file
    heading_1 = np.array(["ID"])
    a = np.arange(0,np.shape(y_predicted)[0]).reshape(np.shape(y_predicted)[0],1)
    heading_2 = np.array(["MEDV"])
    b = np.vstack((heading_1,a))
    c = np.vstack((heading_2,y_predicted))
    out = np.hstack((b,c))
    np.savetxt("output.csv",out,fmt='%s,%s',delimiter=',')

    
    
    #computing for Closed form solution for Penalized Ridge Regression
    print("Shrinkage Parameter for Closed Form",lamb[min_lamb_index])
    w_closed_form = np.zeros((np.shape(phi_train)[1],1))
    phi_transpose = np.transpose(phi_train)
    lamb_identity = lamb[min_lamb_index]*np.identity(len(phi_transpose))
    the_inverse = np.linalg.inv(phi_transpose.dot(phi_train) + lamb_identity)
    w_closed_form = (the_inverse.dot(phi_transpose)).dot(y_train)
    y_predicted_closed_form = predictor(phi_test_final,w_closed_form)
    print("Prediction Result for Closed Form")
    print(y_predicted_closed_form)
      


    #Regression with Lp-norm Regularization
    p =[1.25,1.5,1.75]
    index=1
    
    for j in p:
        w_star_lp = []
        mse_lp = []
        for i in range(0,folds):
            train_lp,test_lp =train_validation_split(augmented_set,index_train_part[i])
            w,m = penalized_lp_norm_regression(test_lp,iterations,step_length,lamb[i],j)
            w_star_lp.append(w)
            mse_lp.append(m)
        min_lamb_index_lp = mse_lp.index(min(mse_lp))
        print("Value of p: ",j)
        print("Shrinkage Parameter for Gradient Descent",lamb[min_lamb_index_lp])
        y_predicted_lp = predictor(phi_test_final,w_star_lp[min_lamb_index_lp])
        print(y_predicted_lp)
        #Outputting values to CSV file for different values of p
        heading_1_lp = np.array(["ID"])
        a_lp = np.arange(0,np.shape(y_predicted_lp)[0]).reshape(np.shape(y_predicted_lp)[0],1)
        heading_2_lp = np.array(["MEDV"])
        b_lp = np.vstack((heading_1_lp,a_lp))
        c_lp = np.vstack((heading_2_lp,y_predicted_lp))
        out_lp = np.hstack((b_lp,c_lp))
        np.savetxt('output_p'+str(index)+'.csv',out_lp,fmt='%s,%s',delimiter=',')
        index = index + 1


#Normalizing data                
def normalize_data(data):
    mean = np.mean(data)
    stdev = np.std(data)
    norm = np.divide(np.subtract(data,mean),stdev)
    return norm


#Splitting basis matrix(phi) and predicted values(y) in training data
def split_phi_y(augmented_data):
    rows_augmented = np.shape(augmented_data)[0]
    cols_augmented = np.shape(augmented_data)[1]
    phi = augmented_data[:,0:(cols_augmented-1)]
    y = augmented_data[:,cols_augmented -1].reshape(rows_augmented,1)
    return phi,y



#Code for penalized Ridge Regression
def ridge_regression(data,iterations,step_length,lamb):
    phi,y = split_phi_y(data)
    w=np.zeros((np.shape(phi)[1],1))
    err = np.zeros((iterations,1))

    for i in range(iterations):
        loss_func = phi.dot(w)-y
        gradient = 2*phi.T.dot(loss_func) + 2*lamb*w
        err[i]=np.linalg.norm(loss_func)
        w=np.subtract(w,np.multiply(step_length,gradient))

    mse=np.linalg.norm(phi.dot(w)-y)
    return w,mse



#Code for penalized Lp norm Regression using Gradient Descent
def penalized_lp_norm_regression(data,iterations,step_length,lamb,p):
    phi,y = split_phi_y(data)
    w=np.zeros((np.shape(phi)[1],1))

    for i in range(iterations):
        loss_func = phi.dot(w)-y
        gradient_lp = 2*phi.T.dot(loss_func)+ lamb*p*np.power(np.abs(w),p-1)
        w = np.subtract(w,np.multiply(step_length,gradient_lp))
   
    mse=np.linalg.norm(phi.dot(w)-y)
    
    return w,mse


#Splitting Train and Test data after cross-validation
def train_validation_split(data,test_part):
    size = len(data)
    train = []
    for i in range(0,size):
        if(i!=test_part):
            train.append(data[i])
    train = np.vstack(train)
    test  = data[test_part]

    return train,test


#Populating partitions(formed by K-Fold Cross-Validation) by random training samples.
def random_subset_gen(data,folds):
    rows_data = np.shape(data)[0]
    cols_data = np.shape(data)[1]
    set_size = int(rows_data/folds)
    randm = np.random.choice(range(0,rows_data),size = rows_data,replace=False)
    final_val_set = []
    for i in range(0,folds):
        final_val_set.append(np.zeros((int(rows_data/folds),cols_data)))

    for i in range(0,rows_data):
        final_val_set[int(i/set_size)][int(i%set_size),:]=data[randm[i],:]

    return final_val_set


#Final Predictor function
def predictor(phi,w):
    return phi.dot(w)


                           

if __name__ =='__main__':
    main()
    
    
    
                       
    
    





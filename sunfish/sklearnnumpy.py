import csv as csv;
import numpy as np;
from sklearn.svm import SVC
from sklearn import cross_validation
import matplotlib.pyplot as plt

clf = SVC(kernel='linear', C=1);
csv_file_object = csv.reader(open('data.o'));
data = [];
for row in csv_file_object:
	temp = [];
	for i in row:
		temp.append(int(i));
	data.append(temp);
data = np.array(data);


features = data[:,range(0,15)];
x_normed = features / features.max(axis=0)
classes = data[:,16];
data_train,data_test,target_train,target_test = cross_validation.train_test_split(features, classes,test_size = 0.20, random_state = 42)

def drawLearningCurve(model):
    sizes = np.linspace(2, 200, 50).astype(int)
    train_error = np.zeros(sizes.shape)
    crossval_error = np.zeros(sizes.shape)
     
    for i,size in enumerate(sizes):
         
        #getting the predicted results of the GaussianNB
        model.fit(data_train[:size,:],target_train[:size])
        predicted = model.predict(data_train)
         
        #compute the validation error
        crossval_error[i] = compute_error(data_test,target_test,model)
         
        #compute the training error
        train_error[i] = compute_error(data_train[:size,:],target_train[:size],model)
        
    #draw the plot
    fig,ax = plt.subplots()
    ax.plot(sizes,crossval_error,lw = 2, label='cross validation error')
    ax.plot(sizes,train_error, lw = 2, label='training error')
    ax.set_xlabel('cross val error')
    ax.set_ylabel('generalization error')
     
    ax.legend(loc = 0)
    ax.set_xlim(0,99)
    ax.set_title('Learning Curve' )
    plt.show();

def compute_error(x, y, model):
    yfit = model.predict(x)
    count = 0;
    for i in range(0,len(y)):
    	if yfit[i] != y[i]:
    		count = count + 1;
    return (float(count)/float(len(y)));
  






if __name__ == '__main__':
	drawLearningCurve(clf);


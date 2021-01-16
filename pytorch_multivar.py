#Author:Harindranath Ambalampitiya, PhD (Theoretical physics)
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import numpy as np
from numpy import pi,sin,cos,sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio


#Data creation for non-linear multi-variate regression
#Simulating the classical path search between two points in 2D
#point 1 is the origin (0,0) and point 2 is arbitrary: (x,y)
#shooting happens from (0,0)
#Given (x,y), find the initial ejection angle w.r.t. x axis and travel time
#Gravity is acting downward along the y-axis.
#model parameters, i.e.,number of independent and
#dependent variables and hidden layers
input_size=2
output_size=2
n_hidden1=800
n_hidden2=400
#Prepare the physics based data
#Input energy of the projectile (in Joules)
E_in=200.
#Mass of the projectile (in Kg)
mass=0.1
#Initial velocity (in m/s)
vel=sqrt(2.*E_in/mass)
#gravity (in m/s/s)
g=9.81
#min-max of roots
tmin,tmax=0.,10.
thetmin,thetmax=0.,pi/2.
#number of samples for  training and test
nsamp=1000
x_data=torch.zeros((nsamp,2))
y_data=torch.zeros((nsamp,2))

ntry_max=10000000
icount=0
for i in range(0,ntry_max):
    thet=thetmin+(thetmax-thetmin)*np.random.ranf()
    time=tmin+(tmax-tmin)*np.random.ranf()
    xt=vel*cos(thet)*time
    yt=vel*sin(thet)*time-.5*g*time**2
    if(yt>=0):
        icount=icount+1
        x_data[icount-1][0]=xt
        x_data[icount-1][1]=yt
    
        y_data[icount-1][0]=thet
        y_data[icount-1][1]=time
        
        if(icount==nsamp-1):
            break
    
#Data normalization
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

x_data=normalize(x_data)
y_data=normalize(y_data)

#Do a train-test split on data
ts=10./nsamp
X_train,X_test,y_train,y_test=train_test_split(x_data.numpy(),y_data.numpy(),test_size=ts)
#convert back to tensors
X_train,X_test=torch.from_numpy(X_train),torch.from_numpy(X_test)
y_train,y_test=torch.from_numpy(y_train),torch.from_numpy(y_test)


class NonLinearRegression(nn.Module):
    def __init__(self,input_size,n_hidden1,n_hidden2,output_size):
        super(NonLinearRegression,self).__init__()
        self.hidden1=nn.Linear(input_size, n_hidden1)
        self.hidden2=nn.Linear(n_hidden1, n_hidden2)
        self.predict=nn.Linear(n_hidden2,output_size)
    
    def forward(self,x):
    
        y_out=Fun.relu(self.hidden1(x))
        y_out=Fun.relu(self.hidden2(y_out))
        return self.predict(y_out)
    
#Model description
model=NonLinearRegression(input_size,n_hidden1,n_hidden2,output_size)
#Mean squared error (MSE) loss function
criterion=nn.MSELoss()
#stochastic gradient descent (SGD) optimization
#lr is the learning-rate
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#
#Visualization of the learning-process
#create the image-array
image_seq=[]
fig,ax=plt.subplots(figsize=(8,8))
#
#
for epoch in range(10000):
    
    #initial prediction with a forward-pass
    y_predict=model(X_train)
    
    #compute the error functin
    loss=criterion(y_predict,y_train)
    #minimize error with gradients
    optimizer.zero_grad()
    #update the weights
    loss.backward()
    optimizer.step()
#    
    #only for illustration purpose
    y_new=model(X_test).detach()
    if(epoch+1)%200==0:
        
        plt.cla()
        ax.scatter(y_test[:,0],y_test[:,1],label='Test Data',c='r')
        ax.scatter(y_new[:,0],y_new[:,1],label='Prediction',c='b')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel('t')
        ax.legend()
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0., 1.25)
    
        ax.text(0.1, 1.2, 'epoch = %d' % epoch)
        ax.text(0.1, 1.1, 'Loss = %.4f' % loss.item())
    
        #Store the images in array
        fig.canvas.draw()       
        image=np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image=image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image_seq.append(image)
#            

# check if the model is successful
#y_new=model(X_test).detach()
#
#
#plt.scatter(y_test[:,0],y_test[:,1],label='Data',c='r')
#plt.scatter(y_new[:,0],y_new[:,1],label='Data',c='b')
#plt.plot(x_test_data,y_new,label='Fit',c='black')
#plt.xlabel(r'$\theta$')
#plt.ylabel('Intensity')
#plt.legend()
#plt.savefig('pytorch_example2.jpeg',dpi=1200)

imageio.mimsave('pytorch_multivar.gif', image_seq, fps=10)

        




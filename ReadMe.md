
An experienced basketball player knows from his or her guts the throw angle and the initial speed of the ball to make a 3-point goal from the player's position. In physics, this problem relates to the search for classical paths between two fixed points. For a moving object under gravity, if (0,0) and (x,y) are the initial and final points, we would like to find the initial elevation angle with respect to the x-axis and the travel time of a projectile which connects (0,0) and (x,y). If the energy E of the projectile is a constant, there can exist 2 classical paths between (0,0) and (x,y). Formally, these paths can be obtained by minimizing,

<img src="http://latex.codecogs.com/svg.latex?\Delta=|x-x_t(E,\theta,t)|^2&plus;|y-y_t(E,\theta,t)|^2&space;" title="http://latex.codecogs.com/svg.latex?\Delta=|x-x_t(E,\theta,t)|^2+|y-y_t(E,\theta,t)|^2 " />

with respect to <img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" /> and t. <img src="http://latex.codecogs.com/svg.latex?x_t(E,\theta,t)" title="http://latex.codecogs.com/svg.latex?x_t(E,\theta,t)" /> and <img src="http://latex.codecogs.com/svg.latex?y_t(E,\theta,t)" title="http://latex.codecogs.com/svg.latex?y_t(E,\theta,t)" /> are the predicted coordinates obtained from Newton's equation. For the minimization process to be efficient, the initial guess for <img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" /> and t must be close to the actual values.

Finding classical paths in such a way becomes cumbersome if the motion happens in more than 2 dimensions and the force is much more complicated than the gravity. We can try to resolve this by using a Machine learning technique. First we generate the following sets of data.

<img src="http://latex.codecogs.com/svg.latex?\begin{bmatrix}&space;\theta_1&t_1&space;&space;\\\theta_2&space;&t_2&space;&space;\\&space;&&space;&space;\\&space;&&space;&space;\\&space;\theta_n&t_n&space;&space;\\\end{bmatrix},\begin{bmatrix}x_t(E,\theta_1,t_1)&space;&&space;y_t(E,\theta_1,t_1)&space;\\&space;x_t(E,\theta_2,t_2)&&space;y_t(E,\theta_2,t_2)&space;\\&space;&&space;&space;\\&space;&&space;&space;\\&space;x_t(E,\theta_n,t_n)&&space;y_t(E,\theta_n,t_n)&space;\\\end{bmatrix}" title="http://latex.codecogs.com/svg.latex?\begin{bmatrix} \theta_1&t_1 \\\theta_2 &t_2 \\ & \\ & \\ \theta_n&t_n \\\end{bmatrix},\begin{bmatrix}x_t(E,\theta_1,t_1) & y_t(E,\theta_1,t_1) \\ x_t(E,\theta_2,t_2)& y_t(E,\theta_2,t_2) \\ & \\ & \\ x_t(E,\theta_n,t_n)& y_t(E,\theta_n,t_n) \\\end{bmatrix}" />


In the Machine learning process, we use (x,y) as the input and (<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />,t) as the output. After the training process, we can use the model to predict (<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />,t) for any given (x,y). These predcited data can be "good guesses" for a minimization algorithm.

Note that this is a toy model and can be generalized to more complicated motion.

In the following figure, I show the exact and predicted (<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />,t) data for some (x,y) coordinates. 

![pytorch_multivar](pytorch_multivar.gif)

Overall, the predicted data approach the exact values over the iterations. But still we can see there are some outliers. 

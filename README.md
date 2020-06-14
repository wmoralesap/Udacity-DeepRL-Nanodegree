# Udacity-DeepRL-Nanodegree

[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

In this repository you can find several reinforcement learning projects. This projects are developed to complete [Udacity's Deep Reinforcement Learning Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Projects

* [Project 1: Navigation](https://github.com/wmoralesap/Udacity-DeepRL-Nanodegree/p1_navigation)
* [Project 2: Continous Control](https://github.com/wmoralesap/Udacity-DeepRL-Nanodegree/p2_continuous-control)
* [Project 2: Continous Control](https://github.com/wmoralesap/Udacity-DeepRL-Nanodegree/p2_collab-compet)

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.
	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
2. Install pytorch.
	- __Linux__ or __Mac__: 
	```bash
	conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
	```
	- __Windows__: 
	```bash
	conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
	```

3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/wmoralesap/Udacity-DeepRL-Nanodegree
cd Udacity-DeepRL-Nanodegree/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]
# EVA-8 Assignment3 (Creating a simple neural network in Excel sheet

## PART-1

In this assignment, we are asked to create a simple neural network using the backpropagation rule in an excel sheet. The network for which, we need to build the formula is shown below:
![Getting Started](model.png)

In the first row, we start by adding random weights w1, w2, w3, w4, w5, w6, w7, w8 
Then we fill other values using the formula based on the model and using backpropogation. The formulas look as follows:
![Getting Started](full.png)


In the second row, we update all the weights using this formula (w1 = w1-η*∂E/∂w1). The remaining values are filled with the new weights in second row


We repeat the above process for more than 100 rows (representing 100 epochs). Following image shows, how things look after several epochs:
![Getting Started](sheet.png)



We tried different values of learning rate (η)
η=0.1
![Getting Started](0.1.png)


η=0.2
![Getting Started](0.2.png)

η=0.5
![Getting Started](0.5.png)


η=0.8
![Getting Started](0.8.png)


η=1.0
![Getting Started](1.0.png)


η=2.0
![Getting Started](2.0.png)



From, above image we can see the loss converges faster when we increase the learning rate. 


## PART-2 

In this part, we are asked to achieve 99.4% validation accuracy in 20k and ess than 20 Epochs. We are allowed to do things like batch normalization, dropouts, Global average pooling. 

My network has 16,196 parameters. I am using dropout in each colvolution layer with rate of 0.05 and batch normalization in each layer except the last layer. I achieved validation accuracy of 99.48%. PART-2.ipynb contains the code and I am also attached a pdf version of the code file. 
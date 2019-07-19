# fortran-ann
Simple neural network module for f90

### Functions

Below is an example whereby a network with 2 inputs, 1 output and two hidden layers each with 3 nodes is trained to output the XOR of the two inputs:

```Fortran
use neural

type(network) :: testNet

call neural_init_network(testNet, (/ 2, 3, 3, 1 /))
call neural_train_network(testNet, "xor.dat", 10000)

! call neural_save_network(testNet, "network.dat")
! call neural_load_network(testNet, "network.dat")

print *, "1 and 1 = ", neural_use_network(testNet, (/ 1.0, 1.0 /))
print *, "0 and 1 = ", neural_use_network(testNet, (/ 0.0, 1.0 /))

```

### Training data file format

The training files should start with the number of inputs, followed by the number of outputs, each on a new lines. The rest of the data should then follow, with inputs before their outputs. For example, if there are two inputs and one output, the following file suggests that 1 and 0 should result in 1:

```
2
1
1
0
1
```

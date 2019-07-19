module neural

  ! Explicit typing only
  implicit none

  ! Everything is private unless specified
  private

  ! The neural network type, stores info about lauer sizes, weights and biases
  type network
      integer, dimension(:), allocatable    :: ls            ! Layer sizes
      integer                               :: nl            ! Number of layers
      real, dimension(:, :, :), allocatable :: weights
      real, dimension(:, :), allocatable    :: biases
      integer                               :: max_size
  end type

  public :: network, neural_train_network, neural_use_network, neural_init_network
  public :: neural_save_network, neural_load_network, neural_train_network_direct

contains

  subroutine neural_train_network(net, file_name, repeat, learn_rate)

    !=========================================================================!
    !                                                                         !
    !  Train the network using data from a given file. This acts              !
    !  as a wrapper around neural_train_network_direct, loading the file      !
    !  and passing it as arrays.                                              !
    !                                                                         !
    !  The first two lines should contain the number of inputs and outputs,   !
    !  with the rest of the lines containing inputs and then outputs,         !
    !  with every value on a different line.                                  !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !       net,     intent(inout)       the network to train                 !
    !       filename,   intent(in)       the file to train on                 !
    !       repeat,     intent(in)       how many times to repeat the data    !
    !       learn_rate, intent(in)       how fast the network should learn    !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    type(network), intent(inout) :: net
    character(*), intent(in) :: file_name
    integer, optional, intent(in) :: repeat
    real, optional, intent(in) :: learn_rate

    real, dimension(:, :), allocatable :: input, output

    integer, parameter :: file_num = 18
    integer :: stat, input_size, output_size, num_lines, i, j, num_data

    open(unit=file_num, file=file_name, iostat=stat)
    if (stat /= 0) stop "Error opening file"

    num_lines = 0
    do

      read(file_num, *, iostat=stat)
      if (stat /= 0) exit
      num_lines = num_lines + 1

    end do

    rewind(file_num)
    read(file_num, *, iostat=stat) input_size
    read(file_num, *, iostat=stat) output_size

    num_data = int((num_lines - 2) / (input_size + output_size))

    allocate(input(num_data, input_size))
    allocate(output(num_data, output_size))

    do i = 1, num_data

      do j = 1, input_size
        read(file_num, *, iostat=stat) input(i, j)
      end do

      do j = 1, output_size
        read(file_num, *, iostat=stat) output(i, j)
      end do

    end do

    close(file_num)

    call neural_train_network_direct(net, input, output, repeat, learn_rate)

  end subroutine neural_train_network

  subroutine neural_train_network_direct(net, input, output, repeat, learn_rate)

    !=========================================================================!
    !                                                                         !
    !  Train the neural network on sets of data, given as arrays, which       !
    !  is generally less useful that the file-based wrapper                   !
    !  neural_train_network.                                                  !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !       net,     intent(inout)       the network to train                 !
    !       input,      intent(in)       the input data to train on           !
    !       output,     intent(in)       the output data to train on          !
    !       repeat,     intent(in)       how many times to repeat the data    !
    !       learn_rate, intent(in)       how fast the network should learn    !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    type(network), intent(inout) :: net
    real, dimension(:, :), intent(in) :: input
    real, dimension(:, :), intent(in) :: output
    integer, optional, intent(in) :: repeat
    real, optional, intent(in) :: learn_rate

    integer :: i,j,k,l, max

    real, dimension(net%nl, net%max_size, net%max_size) :: delta_weights
    real, dimension(net%nl-1, net%max_size) :: delta_biases

    real, dimension(net%nl, size(input, 1), net%max_size) :: layer_activations, layer_outputs
    real, dimension(net%nl, size(input, 1), net%max_size) :: layer_errors, d_layer

    real :: lr

    ! If given a learn_rate, use that instead of the default
    if (present(learn_rate)) then
      lr = learn_rate
    else
      lr = 1.0
    end if

    ! If given a repeat number, use that instead of the default
    if (present(repeat)) then
      max = repeat
    else
      max = 1
    end if

    ! Repeat as many times as requested
    do l=1, max

      ! Reset things just in case
      layer_activations = 0
      layer_outputs = 0
      layer_errors = 0
      d_layer = 0

      !----------------------------!
      ! Begin forward propagation  !
      !----------------------------!

      ! Treat the input like the output of the first layer
      layer_outputs(1, :, 1:net%ls(1)) = input

      ! Loop over the layers
      do j=2, net%nl

        ! Multiply the previous layer results by the weights
        layer_activations(j, :, 1:net%ls(j)) = matmul(layer_outputs(j-1,:,1:net%ls(j-1)), &
        & net%weights(j-1,1:net%ls(j-1),1:net%ls(j)))

        ! Add the biases
        do i=1, size(input, 1)
          layer_activations(j, i, 1:net%ls(j)) = layer_activations(j, i, 1:net%ls(j)) + net%biases(j-1, 1:net%ls(j))
        end do

        ! Sigmoid the activations to get the outputs
        layer_outputs(j, :, 1:net%ls(j)) = neural_sigmoid_array(layer_activations(j, :, 1:net%ls(j)))

      end do

      !------------------------------!
      ! Begin backwards propagation  !
      !------------------------------!

      ! Do output layer seperate since slightly different
      layer_errors(net%nl, :, 1:net%ls(net%nl)) = output - layer_outputs(net%nl, :, 1:net%ls(net%nl))
      d_layer(net%nl, :, 1:net%ls(net%nl)) = layer_errors(net%nl, :, 1:net%ls(net%nl)) * &
      & neural_sigmoid_prime_array(layer_outputs(net%nl, :, 1:net%ls(net%nl)))

      ! Backpropagate the rest of the layers
      do j=net%nl-1, 2, -1

        layer_errors(j, :, 1:net%ls(j)) = matmul(d_layer(j+1, :, 1:net%ls(j+1)), &
        & transpose(net%weights(j, 1:net%ls(j), 1:net%ls(j+1))))
        d_layer(j, :, 1:net%ls(j)) = layer_errors(j, :, 1:net%ls(j)) * neural_sigmoid_prime_array(layer_outputs(j, :, 1:net%ls(j)))

      end do

      ! Determine the change in weights and biases
      do j=net%nl-1, 1, -1

        delta_weights(j,1:net%ls(j),1:net%ls(j+1)) = matmul(transpose(layer_outputs(j, :, 1:net%ls(j))), &
        & d_layer(j+1, :, 1:net%ls(j+1))) * lr
        delta_biases(j,1:net%ls(j+1)) = sum(d_layer(j+1, :, net%ls(j+1)), 1) * lr

      end do

      ! Update the network weights
      do i = 1, net%nl-1
        do j = 1, net%max_size
          do k = 1, net%max_size
             net%weights(i,j,k) = net%weights(i,j,k) + delta_weights(i,j,k)
          end do
        end do
      end do

      ! Update the network biases
      do i = 1, net%nl-1
        do j = 1, net%max_size
            net%biases(i,j) = net%biases(i,j) + delta_biases(i,j)
        end do
      end do

    end do

  end subroutine neural_train_network_direct

  function neural_use_network(net, input)

    !=========================================================================!
    !                                                                         !
    !  Use the neural network on a set of inputs to produce a set of outputs  !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !       net,     intent(in)       the network to train                    !
    !       input,   intent(in)       the inputs to use                       !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    type(network), intent(in) :: net
    real, dimension(net%ls(1)), intent(in) :: input
    real, dimension(net%ls(net%nl)) :: neural_use_network
    real, dimension(net%nl, 1, net%max_size) :: layer_activations, layer_outputs
    integer :: j

    ! Treat the input like the output of the first layer
    layer_outputs(1, 1, 1:net%ls(1)) = input

    ! Loop over the layers
    do j=2, net%nl

      ! Multiply the previous layer results by the weights
      layer_activations(j, 1, 1:net%ls(j)) = matmul(layer_outputs(j-1,1,1:net%ls(j-1)), net%weights(j-1,1:net%ls(j-1),1:net%ls(j)))

      ! Add the biases
      layer_activations(j, 1, 1:net%ls(j)) = layer_activations(j, 1, 1:net%ls(j)) + net%biases(j-1, 1:net%ls(j))

      ! Sigmoid the activations to get the outputs
      layer_outputs(j, 1, 1:net%ls(j)) = neural_sigmoid_vector(layer_activations(j, 1, 1:net%ls(j)))

    end do

    ! The results are the last layer outputs
    neural_use_network = layer_outputs(net%nl, 1, 1:net%ls(net%nl))

  end function neural_use_network

  subroutine neural_init_network(net, sizes)

    !=========================================================================!
    !                                                                         !
    !  Initialise the network to certain sizes, then using a normal           !
    !  distribution to guess the starting weights and biases                  !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !       net,     intent(inout)       the network to initialise            !
    !       sizes,      intent(in)       an array of layer sizes              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    type(network), intent(inout) :: net
    integer, dimension(:), intent(in) :: sizes
    integer :: i, j, k

    net%nl = size(sizes(:))
    allocate(net%ls(net%nl))
    net%ls = sizes
    net%max_size = maxval(sizes(:))
    allocate(net%weights(net%nl-1, net%max_size, net%max_size))
    allocate(net%biases(net%nl-1, net%max_size))

    call RANDOM_seed()

    net%weights = 0
    do i=1, net%nl-1
      do j=1, net%ls(i)
        do k=1, net%ls(i+1)
          net%weights(i,j,k) = neural_normal()
        end do
      end do
    end do

    net%biases = 0
    do i=1, net%nl-1
      do j=1, net%max_size
          net%biases(i,j) = neural_normal()
      end do
    end do

  end subroutine neural_init_network

  subroutine neural_load_network(net, file)

    !=========================================================================!
    !                                                                         !
    !  Loads the network from a file, loading sizes first                     !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    type(network), intent(inout) :: net
    character(*), intent(in) :: file
    integer, parameter :: file_num = 16
    integer :: i, j, k, stat

    ! Deallocate if already allocated
    if (allocated(net%ls)) deallocate(net%ls)
    if (allocated(net%weights)) deallocate(net%weights)
    if (allocated(net%biases)) deallocate(net%biases)

    ! Open the file containing network info
    open(unit=file_num, file=file, iostat=stat)
    if (stat /= 0) stop "Error opening file"

    ! Sizes first
    read(file_num, *) net%nl
    allocate(net%ls(net%nl))
    do i = 1, net%nl
      read(file_num, *) net%ls(i)
    end do

    ! Recalculate the max size
    net%max_size = maxval(net%ls(:))

    ! Allocate the arrays
    allocate(net%weights(net%nl-1, net%max_size, net%max_size))
    allocate(net%biases(net%nl-1, net%max_size))

    ! Then weights
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        do k = 1, net%ls(i+1)
          read(file_num, *) net%weights(i,j,k)
        end do
      end do
    end do

    ! Then biases
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        read(file_num, *) net%biases(i,j)
      end do
    end do

    close(file_num)

  end subroutine neural_load_network

  subroutine neural_save_network(net, file)

    !=========================================================================!
    !                                                                         !
    !  Saves the network to a file, with size info first                      !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    type(network), intent(in) :: net
    character(*), intent(in) :: file
    integer, parameter :: file_num = 16
    integer :: i, j, k, stat

    open(unit=file_num, file=file, iostat=stat)
    if (stat /= 0) stop "Error opening file"

    ! Sizes first
    write(file_num, *) net%nl
    do i = 1, net%nl
      write(file_num, *) net%ls(i)
    end do

    ! Then weights
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        do k = 1, net%ls(i+1)
          write(file_num, *) net%weights(i,j,k)
        end do
      end do
    end do

    ! Then biases
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        write(file_num, *) net%biases(i,j)
      end do
    end do

    close(file_num)

  end subroutine neural_save_network

  function neural_sigmoid_vector(x)

    !=========================================================================!
    !                                                                         !
    !  Applies the sigmoid function to a 1D vector                            !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    real, dimension(:), intent(in) :: x
    real, dimension(size(x)) :: neural_sigmoid_vector
    integer :: i

    do i=1, size(x)
      neural_sigmoid_vector(i) = 1.0 / (1.0 + exp(-x(i)))
    end do

  end function neural_sigmoid_vector

  function neural_sigmoid_array(x)

    !=========================================================================!
    !                                                                         !
    !  Applies the sigmoid function to an array             !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    real, dimension(:,:), intent(in) :: x
    real, dimension(size(x,1), size(x,2)) :: neural_sigmoid_array
    integer :: i, j

    do i=1, size(x,1)
      do j=1, size(x,2)
        neural_sigmoid_array(i, j) = 1.0 / (1.0 + exp(-x(i, j)))
      end do
    end do

  end function neural_sigmoid_array

  function neural_sigmoid_prime_array(x)

    !=========================================================================!
    !                                                                         !
    !  Applies the derivative of the sigmoid function to an array             !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    real, dimension(:,:), intent(in) :: x
    real, dimension(size(x, 1), size(x, 2)) :: neural_sigmoid_prime_array

    neural_sigmoid_prime_array(:, :) = x(:, :)*(1.0 - x(:, :))

  end function neural_sigmoid_prime_array

  function neural_normal()

    !=========================================================================!
    !                                                                         !
    !  Approximates a normal distribution for the initial weights and biases  !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    real :: neural_normal
    real :: temp_rand1, temp_rand2, temp_rand3

    call random_number(temp_rand1)
    call random_number(temp_rand2)
    call random_number(temp_rand3)

    neural_normal = (temp_rand1 + temp_rand2 + temp_rand3) / 3.0

  end function neural_normal

end module neural

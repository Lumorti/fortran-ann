!=============================================================================!
!                             N E U R A L                                     !
!=============================================================================!
!                                                                             !
! This module allows operations with artifical neural networks, including     !
! intialisation, training, usage, saving and loading.                         !
!                                                                             !
!-----------------------------------------------------------------------------!
! Written by Luke Mortimer, July 2019                                         !
!=============================================================================!

module neural

  ! Explicit typing only
  implicit none

  ! Everything is private unless specified
  private

  !---------------------------------------------------------------------------!
  !                       P u b l i c   R o u t i n e s                       !
  !---------------------------------------------------------------------------!
  public :: neural_init_network
  public :: neural_use_network
  public :: neural_train_network
  public :: neural_train_network_direct
  public :: neural_load_network
  public :: neural_save_network

  !---------------------------------------------------------------------------!
  !                        P u b l i c   T y p e s                            !
  !---------------------------------------------------------------------------!
  public :: network

  ! The neural network type, stores info about lauer sizes, weights and biases
  type network
      integer, dimension(:), allocatable    :: ls            ! Layer sizes
      integer                               :: nl            ! Number of layers
      real, dimension(:, :, :), allocatable :: weights
      real, dimension(:, :), allocatable    :: biases
      integer                               :: max_size
  end type

  !---------------------------------------------------------------------------!
  !                      P r i v a t e   R o u t i n e s                      !
  !---------------------------------------------------------------------------!

  !---------------------------------------------------------------------------!
  !                      P r i v a t e   V a r i a b l e s                    !
  !---------------------------------------------------------------------------!

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
    ! trace     for tracing entry and exit                                    !
    ! io        for error messages                                            !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    use trace, only: trace_entry, trace_exit
    use io, only: io_allocate_abort

    type(network), intent(inout) :: net
    character(*), intent(in) :: file_name
    integer, optional, intent(in) :: repeat
    real, optional, intent(in) :: learn_rate

    real, dimension(:, :), allocatable :: input, output

    integer, parameter :: file_num = 18
    integer :: ierr, input_size, output_size, num_lines, i, j, num_data

    call trace_entry('neural_train_network', ierr)

    ! Open the file to train from
    open(unit=file_num, file=file_name, iostat=ierr)
    if (ierr /= 0) stop "Error opening file"

    ! Count the number of lines in the file
    num_lines = 0
    do

      read(file_num, *, iostat=ierr)
      if (ierr /= 0) exit
      num_lines = num_lines + 1

    end do

    ! Restart
    rewind(file_num)

    ! Get the size of the input and output arrays
    read(file_num, *, iostat=ierr) input_size
    read(file_num, *, iostat=ierr) output_size

    ! Determine how many data sets there are
    num_data = int((num_lines - 2) / (input_size + output_size))

    ! Allocate the arrays for storing the input/output data
    allocate(input(num_data, input_size), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('input', 'neural_train_network')
    allocate(output(num_data, output_size), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('output', 'neural_train_network')

    ! Load the data into the input/output arrays
    do i = 1, num_data

      do j = 1, input_size
        read(file_num, *, iostat=ierr) input(i, j)
      end do
      do j = 1, output_size
        read(file_num, *, iostat=ierr) output(i, j)
      end do

    end do

    close(file_num)

    ! Train the network using this data now in memory
    call neural_train_network_direct(net, input, output, repeat, learn_rate)

    call trace_exit('neural_train_network', ierr)

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
    ! trace     for tracing entry and exit                                    !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    use trace, only: trace_entry, trace_exit

    type(network), intent(inout) :: net
    real, dimension(:, :), intent(in) :: input
    real, dimension(:, :), intent(in) :: output
    integer, optional, intent(in) :: repeat
    real, optional, intent(in) :: learn_rate

    integer :: i,j,k,l, max, ierr
    real :: lr

    real, dimension(net%nl, net%max_size, net%max_size) :: delta_weights
    real, dimension(net%nl-1, net%max_size) :: delta_biases
    real, dimension(net%nl, size(input, 1), net%max_size) :: layer_activations, layer_outputs
    real, dimension(net%nl, size(input, 1), net%max_size) :: layer_errors, d_layer

    call trace_entry('neural_train_network_direct', ierr)

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

    call trace_exit('neural_train_network_direct', ierr)

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
    ! trace     for tracing entry and exit                                    !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    use trace, only: trace_entry, trace_exit

    type(network), intent(in) :: net
    real, dimension(net%ls(1)), intent(in) :: input
    real, dimension(net%ls(net%nl)) :: neural_use_network
    real, dimension(net%nl, 1, net%max_size) :: layer_activations, layer_outputs
    integer :: j, ierr

    call trace_entry('neural_use_network', ierr)

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

    call trace_exit('neural_use_network', ierr)

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
    !       trace     for tracing entry and exit                              !
    !       io        for error messages                                      !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    use trace, only: trace_entry, trace_exit
    use io, only: io_allocate_abort

    type(network), intent(inout) :: net
    integer, dimension(:), intent(in) :: sizes
    integer :: i, j, k, ierr

    call trace_entry('neural_init_network', ierr)

    net%nl = size(sizes(:))

    ! Allocate the layer sizes list
    allocate(net%ls(net%nl), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('net%nl', 'neural_init_network')

    net%ls = sizes
    net%max_size = maxval(sizes(:))

    ! Allocate the weights and biases arrays
    allocate(net%weights(net%nl-1, net%max_size, net%max_size), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('net%weights', 'neural_init_network')
    allocate(net%biases(net%nl-1, net%max_size), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('net%biases', 'neural_init_network')

    ! Init the random generator
    call RANDOM_seed()

    ! Randomise the initial weights
    net%weights = 0
    do i=1, net%nl-1
      do j=1, net%ls(i)
        do k=1, net%ls(i+1)
          net%weights(i,j,k) = neural_normal()
        end do
      end do
    end do

    ! Randomise the initial biases
    net%biases = 0
    do i=1, net%nl-1
      do j=1, net%max_size
          net%biases(i,j) = neural_normal()
      end do
    end do

    call trace_exit('neural_init_network', ierr)

  end subroutine neural_init_network

  subroutine neural_load_network(net, file)

    !=========================================================================!
    !                                                                         !
    !  Loads the network from a file, loading sizes first                     !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !       net,     intent(inout)       the network to load into             !
    !       file,       intent(in)       the filename to load fro             !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    ! trace     for tracing entry and exit                                    !
    ! io        for error messages                                            !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    use trace, only: trace_entry, trace_exit
    use io, only: io_abort, io_allocate_abort

    type(network), intent(inout) :: net
    character(*), intent(in) :: file
    integer, parameter :: file_num = 16
    integer :: i, j, k, ierr

    call trace_entry('neural_load_network', ierr)

    ! Deallocate if already allocated
    if (allocated(net%ls)) then
      deallocate(net%ls, stat=ierr)
      if (ierr /= 0) call io_abort('Error in deallocating net%ls in neural_load_network')
    end if
    if (allocated(net%weights)) then
      deallocate(net%weights, stat=ierr)
      if (ierr /= 0) call io_abort('Error in deallocating net%weights in neural_load_network')
    end if
    if (allocated(net%biases)) then
      deallocate(net%biases, stat=ierr)
      if (ierr /= 0) call io_abort('Error in deallocating net%biases in neural_load_network')
    end if

    ! Open the file containing network info
    open(unit=file_num, file=file, iostat=ierr)
    if (ierr /= 0) stop "Error opening file"

    ! Sizes first
    read(file_num, *) net%nl
    allocate(net%ls(net%nl), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('net%ls', 'neural_load_network')

    do i = 1, net%nl
      read(file_num, *) net%ls(i)
    end do

    ! Recalculate the max size
    net%max_size = maxval(net%ls(:))

    ! Allocate the arrays
    allocate(net%weights(net%nl-1, net%max_size, net%max_size), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('net%weights', 'neural_load_network')
    allocate(net%biases(net%nl-1, net%max_size), stat=ierr)
    if (ierr /= 0) call io_allocate_abort('net%biases', 'neural_load_network')

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

    call trace_exit('neural_load_network', ierr)

  end subroutine neural_load_network

  subroutine neural_save_network(net, file)

    !=========================================================================!
    !                                                                         !
    !  Saves the network to a file, with size info first                      !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    !       net,        intent(in)       the network to save                  !
    !       file,       intent(in)       the filename to save to              !
    !-------------------------------------------------------------------------!
    ! Parent module variables used:                                           !
    !-------------------------------------------------------------------------!
    ! Modules used:                                                           !
    ! trace     for tracing entry and exit                                    !
    ! io        for error messages                                            !
    !-------------------------------------------------------------------------!
    ! Key Internal Variables:                                                 !
    !-------------------------------------------------------------------------!
    ! Necessary conditions:                                                   !
    !-------------------------------------------------------------------------!
    ! Written by Luke Mortimer, July 2019                                     !
    !=========================================================================!

    use trace, only: trace_entry, trace_exit
    use io, only: io_abort

    type(network), intent(in) :: net
    character(*), intent(in) :: file
    integer, parameter :: file_num = 16
    integer :: i, j, k, ierr

    call trace_entry('neural_save_network', ierr)

    open(unit=file_num, file=file, iostat=ierr)
    if (ierr /= 0) then
      call io_abort('Error opening file in neural_save_network')
      return
    end if

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

    call trace_exit('neural_save_network', ierr)

  end subroutine neural_save_network

  function neural_sigmoid_vector(x)

    !=========================================================================!
    !                                                                         !
    !  Applies the sigmoid function to a 1D vector                            !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    ! x,     intent(in)      the vector to apply the sigmoid function to      !
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
    !  Applies the sigmoid function to an array                               !
    !                                                                         !
    !-------------------------------------------------------------------------!
    ! Arguments:                                                              !
    ! x,     intent(in)      the array to apply the sigmoid function to       !
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
    ! x,     intent(in)     the array to apply the sigmoid prime function to  !
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

    ! Simple approximation, tends to stick around 0.5 as desired
    neural_normal = (temp_rand1 + temp_rand2 + temp_rand3) / 3.0

  end function neural_normal

end module neural

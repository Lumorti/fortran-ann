
module neural

  ! Explicit typing only
  implicit none

  ! Everything is private unless specified
  private

  public :: neural_init_network
  public :: neural_use_network
  public :: neural_train_network_file
  public :: neural_train_network_direct
  public :: neural_load_network
  public :: neural_load_network_from_string
  public :: neural_save_network
  public :: neural_save_network_to_string
  public :: neural_convert_input
  public :: neural_convert_output
  public :: neural_sigmoid

  public :: network

  ! The neural network type, stores info about lauer sizes, weights and biases
  type network
      integer                                         :: nl            ! Number of layers
      integer                                         :: max_size
      integer, dimension(:), allocatable              :: ls            ! Layer sizes
      double precision, dimension(:, :, :), allocatable :: weights
      double precision, dimension(:, :), allocatable    :: biases
  end type

  interface neural_sigmoid
    module procedure neural_sigmoid_vector
    module procedure neural_sigmoid_array
  end interface

contains

  ! Train a certain network from a file, repeating a number of times
  subroutine neural_train_network_file(net, file_name, repeat, learn_rate)

    ! The network to train
    type(network), intent(inout) :: net

    ! The file to train on
    character(*), intent(in) :: file_name

    ! How many times to repeat the training
    integer, optional, intent(in) :: repeat

    ! How fast should it learn? [0,1]
    double precision, optional, intent(in) :: learn_rate

    ! Internal vars
    double precision, dimension(:, :), allocatable :: input, output
    integer :: file_num
    integer :: ierr, input_size, output_size, num_lines, i, j, num_data

    open(newunit=file_num, file=file_name, status="old", iostat=ierr)
    if (ierr /= 0) stop "Couldn't load file for training"

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

    ! Check if these are right for the network and exit otherwise
    if (input_size /= net%ls(1) .or. output_size /= net%ls(net%nl)) then
      print *, "Error training from file, wrong number of inputs/outputs"
      return
    end if

    ! Determine how many data sets there are
    num_data = int((num_lines - 2) / (input_size + output_size))

    ! Allocate the arrays for storing the input/output data
    allocate(input(num_data, input_size), stat=ierr)
    if (ierr /= 0) stop "Error allocating input array in neural_train_network_file"
    allocate(output(num_data, output_size), stat=ierr)
    if (ierr /= 0) stop "Error allocating output array in neural_train_network_file"

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

  end subroutine neural_train_network_file

  ! Train a network directly on some input/output data
  subroutine neural_train_network_direct(net, input, output, repeat, learn_rate)


    type(network), intent(inout) :: net
    double precision, dimension(:, :), intent(in) :: input
    double precision, dimension(:, :), intent(in) :: output
    integer, optional, intent(in) :: repeat
    double precision, optional, intent(in) :: learn_rate

    integer :: i,j,k,l, max, ierr
    double precision :: lr

    double precision, dimension(net%nl, net%max_size, net%max_size) :: delta_weights
    double precision, dimension(net%nl-1, net%max_size) :: delta_biases
    double precision, dimension(net%nl, size(input, 1), net%max_size) :: layer_activations, layer_outputs
    double precision, dimension(net%nl, size(input, 1), net%max_size) :: layer_errors, d_layer

    ! Check if these sizes are right for the network and exit otherwise
    if (size(input, 2) /= net%ls(1) .or. size(output, 2) /= net%ls(net%nl)) then
      stop 'Error in neural_train_network_direct: wrong number of inputs/outputs'
      return
    end if

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

  ! Forward propogate some inputs through the network
  function neural_use_network(net, input)


    type(network), intent(in) :: net
    double precision, dimension(net%ls(1)), intent(in) :: input
    double precision, dimension(net%ls(net%nl)) :: neural_use_network
    double precision, dimension(net%nl, 1, net%max_size) :: layer_activations, layer_outputs
    integer :: j, ierr

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

  ! Initialise the network with random weights
  subroutine neural_init_network(net, sizes)


    type(network), intent(inout) :: net
    integer, dimension(:), intent(in) :: sizes
    integer :: i, j, k, ierr

    ! Dellocate if already initialized
    if (allocated(net%ls)) then
      deallocate(net%ls, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%ls in neural_init_network'
    end if
    if (allocated(net%weights)) then
      deallocate(net%weights, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%weights in neural_init_network'
    end if
    if (allocated(net%biases)) then
      deallocate(net%biases, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%biases in neural_init_network'
    end if

    net%nl = size(sizes(:))

    ! Allocate the layer sizes list
    allocate(net%ls(net%nl), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"

    net%ls = sizes
    net%max_size = maxval(sizes(:))

    ! Allocate the weights and biases arrays
    allocate(net%weights(net%nl-1, net%max_size, net%max_size), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"
    allocate(net%biases(net%nl-1, net%max_size), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"

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

  end subroutine neural_init_network

  ! Load the network from a file
  subroutine neural_load_network(net, file)

    type(network), intent(inout) :: net
    character(*), intent(in) :: file
    integer :: file_num
    integer :: i, j, k, ierr

    ! Deallocate if already allocated
    if (allocated(net%ls)) then
      deallocate(net%ls, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%ls in neural_load_network'
    end if
    if (allocated(net%weights)) then
      deallocate(net%weights, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%weights in neural_load_network'
    end if
    if (allocated(net%biases)) then
      deallocate(net%biases, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%biases in neural_load_network'
    end if

    ! Open the file containing network info
    open(unit=file_num, file=file, status="old")
    if (ierr /= 0) stop 'Failed to open file'

    ! Sizes first
    read(file_num, *) net%nl
    allocate(net%ls(net%nl), stat=ierr)
    if (ierr /= 0) stop "Failed to allocate array"

    do i = 1, net%nl
      read(file_num, *) net%ls(i)
    end do

    ! Recalculate the max size
    net%max_size = maxval(net%ls(:))

    ! Allocate the arrays
    allocate(net%weights(net%nl-1, net%max_size, net%max_size), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"
    allocate(net%biases(net%nl-1, net%max_size), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"

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

  ! Save the network to a file
  subroutine neural_save_network(net, file)


    type(network), intent(in) :: net
    character(*), intent(in) :: file
    integer :: file_num
    integer :: i, j, k, ierr

    ! Open the file to save the network to
    open(newunit=file_num, file=file, status="new")
    if (ierr /= 0) stop "Failed to open file"

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

  ! Save the network to a string and return it
  function neural_save_network_to_string(net)


    type(network), intent(in) :: net
    character(300) :: temp_result
    character(:), allocatable :: neural_save_network_to_string
    character(10) :: temp
    integer :: i, j, k, ierr

    ! Start with a blank string
    temp_result = ""
    temp = ""

    ! Sizes first
    write(temp, "(I3)") net%nl
    temp_result = trim(temp_result) // trim(temp) // ","
    do i = 1, net%nl
      write(temp, "(I3)") net%ls(i)
      temp_result = trim(temp_result) // trim(temp) // ","
    end do

    ! Then weights
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        do k = 1, net%ls(i+1)
          write(temp, "(f10.5)") net%weights(i,j,k)
          temp_result = trim(temp_result) // trim(temp) // ","
        end do
      end do
    end do

    ! Then biases
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        write(temp, "(f10.5)") net%biases(i,j)
        temp_result = trim(temp_result) // trim(temp) // ","
      end do
    end do

    ! Compress the result by removing whitespace
    neural_save_network_to_string = ""
    do i=1, len_trim(temp_result)
      if (temp_result(i:i) /= " ") then
        neural_save_network_to_string = trim(neural_save_network_to_string) // temp_result(i:i)
      end if
    end do

  end function neural_save_network_to_string

  ! Load the network from a string
  subroutine neural_load_network_from_string(net, string)

    type(network), intent(inout) :: net
    character(*), intent(in) :: string
    integer :: i, j, k, ierr, l, prev_l

    ! Deallocate if already allocated
    if (allocated(net%ls)) then
      deallocate(net%ls, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%ls in neural_load_network'
    end if
    if (allocated(net%weights)) then
      deallocate(net%weights, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%weights in neural_load_network'
    end if
    if (allocated(net%biases)) then
      deallocate(net%biases, stat=ierr)
      if (ierr /= 0) stop 'Error in deallocating net%biases in neural_load_network'
    end if

    ! Read the number of layers
    prev_l = 0
    l = index(string(:), ",")
    read(string(prev_l+1:l-1), *) net%nl
    allocate(net%ls(net%nl), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"

    ! Read the size of each layer
    do i = 1, net%nl
      prev_l = l
      l = l+index(string(l+1:), ",")
      read(string(prev_l+1:l-1), *) net%ls(i)
    end do

    ! Recalculate the max size
    net%max_size = maxval(net%ls(:))

    ! Allocate the arrays
    allocate(net%weights(net%nl-1, net%max_size, net%max_size), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"
    allocate(net%biases(net%nl-1, net%max_size), stat=ierr)
    if (ierr /= 0) stop "failed to allocate array"

    ! Then weights
    net%weights = 0
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        do k = 1, net%ls(i+1)
          prev_l = l
          l = l+index(string(l+1:), ",")
          read(string(prev_l+1:l-1), *) net%weights(i,j,k)
        end do
      end do
    end do

    ! Then biases
    net%biases = 0
    do i = 1, net%nl-1
      do j = 1, net%ls(i)
        prev_l = l
        l = l+index(string(l+1:), ",")
        read(string(prev_l+1:l-1), *) net%biases(i,j)
      end do
    end do

  end subroutine neural_load_network_from_string

  ! Apply the sigmoid to a 1D array
  function neural_sigmoid_vector(x)

    double precision, dimension(:), intent(in) :: x
    double precision, dimension(size(x)) :: neural_sigmoid_vector
    integer :: i

    do i=1, size(x)
      neural_sigmoid_vector(i) = 1.0 / (1.0 + exp(-x(i)))
    end do

  end function neural_sigmoid_vector

  ! Apply the sigmoid to a 2D array
  function neural_sigmoid_array(x)

    double precision, dimension(:,:), intent(in) :: x
    double precision, dimension(size(x,1), size(x,2)) :: neural_sigmoid_array
    integer :: i, j

    do i=1, size(x,1)
      do j=1, size(x,2)
        neural_sigmoid_array(i, j) = 1.0 / (1.0 + exp(-x(i, j)))
      end do
    end do

  end function neural_sigmoid_array

  ! Apply the inverse of the sigmoid to a 1D array
  function neural_sigmoid_inv_vector(x)

    double precision, dimension(:), intent(in) :: x
    double precision, dimension(size(x)) :: neural_sigmoid_inv_vector
    integer :: i

    do i=1, size(x)
      neural_sigmoid_inv_vector(i) = -log((1.0 / x(i)) - 1.0)
    end do

  end function neural_sigmoid_inv_vector

  ! Apply the derivative of the sigmoid to a 2D array
  function neural_sigmoid_prime_array(x)

    double precision, dimension(:,:), intent(in) :: x
    double precision, dimension(size(x, 1), size(x, 2)) :: neural_sigmoid_prime_array

    neural_sigmoid_prime_array(:, :) = x(:, :)*(1.0 - x(:, :))

  end function neural_sigmoid_prime_array

  ! Return a number in a normal (ish) distribution
  function neural_normal()

    double precision :: neural_normal
    double precision :: temp_rand1, temp_rand2, temp_rand3

    call random_number(temp_rand1)
    call random_number(temp_rand2)
    call random_number(temp_rand3)

    ! Simple approximation, tends to stick around 0.5 as desired
    neural_normal = (temp_rand1 + temp_rand2 + temp_rand3) / 3.0

  end function neural_normal

end module neural

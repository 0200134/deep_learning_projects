module matrix_mul_mod
  implicit none
contains
  subroutine matrix_mul(A, B, C, n) bind(C, name="matrix_mul")
    implicit none
    integer, intent(in) :: n
    real, intent(in) :: A(n, n), B(n, n)
    real, intent(out) :: C(n, n)
    integer :: i, j, k

    do i = 1, n
      do j = 1, n
        C(i,j) = 0.0
        do k = 1, n
          C(i,j) = C(i,j) + A(i,k) * B(k,j)
        end do
      end do
    end do
  end subroutine matrix_mul
end module matrix_mul_mod

#include <random>
#include <iostream>
#include <ctime>
#include "omp.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
using namespace tbb;

using namespace std;

void generate_matrix(double * A, int N)
{
#pragma omp parallel for
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A[i*N + j] = rand() % 10;
}

class matrix_multiplication
{
	const double *A, *B;
	double *const C;
	int const N, q;

public:
	matrix_multiplication(double *tA, double *tB, double *tC, int tq, int tN) : A(tA),B(tB), C(tC), N(tN), q(tq){}

	void operator()(const blocked_range<int>& r) const
	{
		int block_size = N / q;
		int begin = r.begin(), end = r.end();
		for (int ijb = begin; ijb != end; ijb++)
		{
			int glob_cord, glob_cord_A, glob_cord_B;
			int ib = ijb / q;
			int jb = ijb % q;
			int loc_b_cord_i = ib;
			int j_new;
			int ib_bs = ib * block_size;
			int jb_bs = jb * block_size;
			for (int i = 0; i < block_size; i++)
				for (int j = 0; j < block_size; j++)
				{
					glob_cord = (ib_bs + i)*N + (jb_bs+ j);
					C[glob_cord] = 0;
				}
			for (int m = 0; m < q; m++)
			{
				j_new = (ib + m) % q;
				for (int i = 0; i < block_size; i++)
					for (int j = 0; j < block_size; j++)
					{
						glob_cord = (ib_bs + i)*N + (jb_bs+ j);
						for (int k = 0; k < block_size; k++)
						{
							glob_cord_A = (ib_bs + i)*N + (j_new*block_size + k);
							glob_cord_B = (loc_b_cord_i*block_size + k)*N + (jb_bs+ j);
#pragma simd
							C[glob_cord] += A[glob_cord_A] * B[glob_cord_B];
						}
					}
				loc_b_cord_i = (loc_b_cord_i + 1) % q;
			}
		}
	}
};
void matrix_multiplication_tbb(double * A, double * B, double * C, int q, int N, int NT = 4)
{
	task_scheduler_init init(NT);
	int grainSize = 1;
	parallel_for(blocked_range<int>(0, q*q, grainSize), matrix_multiplication(A, B, C, q, N));
}

void matrix_multiplication_omp(double * A, double * B, double * C, int q, int N, int NT = 4)
{	
	const int block_size = N / q;

	omp_set_num_threads(NT);
#pragma omp parallel
	{
#pragma omp for
		for (int ijb = 0; ijb < q*q; ijb++)
		{
			int glob_cord, glob_cord_A, glob_cord_B;
			int ib = ijb / q;
			int jb = ijb % q;
			int loc_b_cord_i = ib;
			int ib_bs = ib * block_size;
			int jb_bs = jb * block_size;
			int loc_b_bs;
			int jn_bs;
			for (int i = 0; i < block_size; i++)
				for (int j = 0; j < block_size; j++)
				{
					glob_cord = (ib_bs + i)*N + (jb_bs+ j);
					C[glob_cord] = 0;
				}
			for (int m = 0; m < q; m++)
			{
				jn_bs = ((ib + m) % q) * block_size;
				loc_b_bs = loc_b_cord_i * block_size;
				for (int i = 0; i < block_size; i++)
					for (int j = 0; j < block_size; j++)
					{
						glob_cord = (ib_bs + i)*N + (jb_bs+ j);
#pragma vector always
#pragma ivdep
						for (int k = 0; k < block_size; k++)
						{
							C[glob_cord] = C[glob_cord] + A[(ib_bs + i)*N + (jn_bs + k)] * B[(loc_b_bs + k)*N + (jb_bs + j)];
						}
					}
				loc_b_cord_i = (loc_b_cord_i + 1) % q;
			}
		}
	}
}

int main()
{

	int q = 20;
	const int N = 2000;
	const int NT = 4;
	//const int block_size = 2;
	double* A = new double[N*N];
	double* B = new double[N*N];
	double* C = new double[N*N];
	double* Res = new double[N*N];
	generate_matrix(A, N);
	generate_matrix(B, N);
	int ijb = 10;
	int block_i = ijb / q;
	int block_j = ijb % q;
	//cout << block_i << block_j;
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		cout << A[i*N + j] << " ";
	//	cout << "\n";
	//}
	//cout << "\n";
	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		cout << B[i*N + j] << " ";
	//	cout << "\n";
	//}
	//cout << "\n";

	double start = omp_get_wtime();
	matrix_multiplication_omp(A, B, C, q, N, NT);
	double finish = omp_get_wtime();

	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j < N; j++)
	//		cout << C[i*N+j]<< " ";
	//	cout << "\n";
	//}
	//cout << "\n";
	cout << finish - start;
	//cin >> q;
	return 0;
}

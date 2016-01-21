/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#ifndef __ITALGOS_H
#define __ITALGOS_H

#ifndef NUM_INTERNAL
// #warning "Use of private interfaces"
#endif


struct vec_iter_s;

/**
 * @brief Generic operator function type
 **/
typedef void (*op_f)(void* data, float* dst, const float* src);


/**
 * @brief Proximal operator function type
 */
typedef void (*prox_f)(void* data, float lambda, float* dst, const float* src);

/**
 * @brief Objective function type
 */
typedef float (*obj_f)(const void*, const float*);

/**
 * @brief Line search criterion function type
 */
typedef _Bool (*ls_f)(const void *fdata, float alpha, const float* x_new, const float* gradfx, const float* x );

/**
 * @brief Inverse operator function type
 *
 * dst = inv(A + alpha I) * src
 */
typedef void (*inv_f)(void* data, float alpha, float* dst, const float* src);

/**
 * @brief Inverse operator 2 function type
 *
 */
typedef void (*inv2_f)(void* data, float alpha, float* res, float* dst, const float* src);


/**
 * @brief Store italg history
 */
struct iter_history_s {
	unsigned int numiter;   ///< Number of iterations
	double* objective;	///< Objective values over iterations
	double* relMSE;		///< Relative mean square errors over iterations
	double* resid;		///< Residual over iterations
};


struct pocs_proj_op {

	prox_f proj_fun;
	void* data;
};

/**
 * @brief Conjugate gradient
 *
 * Solve (A + lambda I) x = b
 * where A is Hermitian, ie A = A^*
 */
float conjgrad(unsigned int iter,		///< Number of iterations
	       float l2lambda,			///< L2 regularization, ie, || x ||_2^2
	       float tol,			///< Termination tolerance
	       long N,				///< Length of variable x
	       void* data,			///< Data structure for linear operator A
	       const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	       op_f linop,			///< Hermitian linear operator A
	       float* x,			///< Optimization variable, ex, image
	       const float* b,			///< Observed data, ex, zero-filled image
	       const float* x_truth,            ///< Ground truth (optional)    
	       void* obj_data,			///< Objective data structure (optional)
	       obj_f obj);			///< Objective function (optional)


/**
 * @brief Conjugate gradient with stored iteration parameters
 *
 * Solve (A + lambda I) x = b
 * where A is Hermitian, ie A = A^*
 */
float conjgrad_hist(struct iter_history_s* iter_history,///< Data structure for iteration history
		    unsigned int iter,			///< Number of iterations
		    float l2lambda,			///< L2 regularization, ie, || x ||_2^2
		    float tol,				///< Termination tolerance
		    long N,				///< Length of variable x
		    void* data,				///< Data structure for linear operator A
		    const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
		    op_f linop,			        ///< Hermitian linear operator A
		    float* x,				///< Optimization variable, ex, image
		    const float* b,			///< Observed data, ex, zero-filled image
		    const float* x_truth,               ///< Ground truth (optional)
		    void* obj_data,			///< Objective data structure (optional)
		    obj_f obj);				///< Objective function (optional)


extern const struct cg_data_s* cg_data_init(long N, ///< Length of variable x
					    const struct vec_iter_s* vops); ///< Vector arithmetic operator (CPU/GPU)

extern void cg_data_free(const struct cg_data_s* cgdata, const struct vec_iter_s* vops);

float conjgrad_hist_prealloc(struct iter_history_s* iter_history, 
			     unsigned int iter,			///< Number of iterations 
			     float l2lambda,			///< L2 regularization, ie, || x ||_2^2
			     float tol,				///< Termination tolerance 
			     long N,				///< Length of variable x
			     void* data,			///< Data structure for linear operator A
			     struct cg_data_s* cgdata,
			     const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
			     op_f linop,			///< Linear operator A (Hermitian)
			     float* x,				///< Optimization variable, ex, image
			     const float* b,			///< Observed data, ex, zero-filled image
			     const float* x_truth,		///< Ground truth (optional)
			     void* obj_data,			///< Objective data structure (optional)
			     obj_f obj);			///< Objective function (optional)


/**
 * @brief Landweber iteration
 *
 * Solves \min_x 1/2 || A x - b ||_2^2
 */
void landweber(unsigned int iter,		///< Number of iterations 
	       float tol,			///< Termination tolerance
	       float stepsize,			///< Step size
	       long N,				///< Length of variable x
	       long M,				///< Length of Ax
	       void* data,			///< Data structure for linear operator A
	       const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	       op_f op,				///< Linear operator A
	       op_f adj,			///< Linear operator adjoint A^T 
	       float* x,			///< Optimization variable, ex, image
	       const float* b,			///< Observed data, ex, kspace
	       obj_f obj);			///< Objective function (optional)

/**
 * @brief Symmetric landweber iteration
 *
 * Solves A x = b or equivalently
 * \min_x x^T A x - x^T b
 * A must be conjugate symmetric
 */
void landweber_sym(unsigned int iter,			///< Number of iterations
		   float tol,				///< Termination tolerance
		   float stepsize,			///< Step size	
		   long N,				///< Length of variable x
		   void* data,				///< Data structure for linear operator A
		   const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
		   op_f op,				///< Linear operator A  (Hermitian)
		   float* x,				///< Optimization variable, ex, image
		   const float* b);			///< Observed data, ex, zero-filled image


/**
 * @brief Iterative soft-thresholding
 *
 * Solves \min_x x^T A x - x^T b + \lambda prox
 * A must be conjugate symmetric
 */
void ist(unsigned int iter,		///< Number of iterations
	 float tol,			///< Termination tolerance
	 float stepsize,		///< Step size 
	 float continuation,		///< TODO
	 _Bool hogwild,			///< TODO
	 long N,			///< Length of variable x
	 void* data,			///< Data structure for linear operator A
	 const struct vec_iter_s* vops, ///< Vector arithmetic operator (CPU/GPU)
	 op_f op,			///< Linear operator A (Hermitian)
	 prox_f prox,			///< Proximal operator, ex, soft-threshold
	 void* pdata,			///< Proximal operator data structure
	 float* x,			///< Optimization variable, ex, image
	 const float* b,		///< Observed data, ex, zero-filled image
	 const float* x_truth,		///< Ground truth (optional)
	 void* obj_data,		///< Objective data structure (optional)
	 obj_f obj);			///< Objective function (optional)


/**
 * @brief Fast iterative soft-thresholding
 *
 * Solves \min_x x^T A x - x^T b + \lambda prox
 * A must be conjugate symmetric
 */
void fista(unsigned int iter,			///< Number of iterations
	   float tol,				///< Termination tolerance
	   float stepsize,			///< Step size 
	   float continuation,			///< TODO
	   _Bool hogwild,			///< TODO
	   long N,				///< Length of variable x
	   void* data,				///< Data structure for linear operator A
	   const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	   op_f op,				///< Linear operator A (Hermitian)
	   prox_f prox,				///< Proximal operator, ex, soft-threshold
	   void* pdata,				///< Proximal operator data structure
	   float* x,				///< Optimization variable, ex, image
	   const float* b,			///< Observed data, ex, zero-filled image
	   const float* x_truth,		///< Ground truth (optional)
	   void* obj_data,			///< Objective data structure (optional)
	   obj_f obj);				///< Objective function (optional)


/**
 * @brief Proximal gradient descent
 *
 * Solves \min_x f(x) + g(x)
 * where f is smooth (can be non-convex)
 * and g is non-smooth but convex
 *
 * Requires gradient of f and proximal of g
 * 
 */
void pgd( unsigned int iter,			///< Number of iterations
	  float tol,				///< Termination tolerance 
	  float alpha,			        ///< Step size
	  long N,				///< Length of variable x
	  const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	  void* fdata,			        ///< Linear operator data structure
	  op_f gradf,				///< Linear operator A
	  void* gdata,			        ///< Proximal operator data structure
	  prox_f proxg,				///< Proximal operator, ex, soft-threshold
	  float* x,				///< Optimization variable, ex, image
	  const float* x_truth,			///< Ground truth (optional)
	  void* odata,			        ///< Objective data structure (optional)
	  obj_f obj, 				///< Objective function (optional)
	  ls_f ls);                             ///< Line search criterion (optional)

/**
 * @brief Iterative regularized Gauss-Newton method
 *
 * TODO
 */
void irgnm(unsigned int iter,			///< Number of iterations
	   float stepsize,			///< Step size
	   float redu,
	   void* data,				///< Data structure for linear operator A 
	   long N,				///< Length of variable x
	   long M,				///< Length of Ax
	   const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	   op_f op,				///< Linear operator A 
	   op_f adj,				///< Linear operator adjoint A^T 
	   inv_f inv,				///< Inverse linear operator (A + alphaI)^-1
	   float* x,				///< Optimization variable, ex, image
	    const float* x0,			///< Initial estimate of x
	    const float* y);			///< Observed data, ex, zero-filled image


/**
 * @brief Iterative regularized Gauss-Newton method
 *
 * TODO
 */
void irgnm2(unsigned int iter,			///< Number of iterations
	    float stepsize,			///< Step size
	    float redu,				///< TODO
	    void* data,				///< Data structure for linear operator A 
	    long N,				///< Length of variable x
	    long M,				///< Length of Ax
	    const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	    op_f op,				///< Linear operator A 
	    op_f adj,				///< Linear operator adjoint A^T 
	    inv2_f inv2,			///< TODO 
	    float* x,				///< Optimization variable, ex, image
	    const float* x0,			///< Initial estimate of x
	    const float* y);			///< Observed data, ex, zero-filled image

/**
 * @brief Split?
 *
 * TODO
 */
void split(unsigned int iter,			///< Number of iterations
	   float tol,				///< Termination tolerance
	   float mu,				///< TODO
	   float lambda,			///< TODO
	   long N,				///< Length of variable x
	   void* data,				///< Data structure for linear operator A
	   const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	   op_f op,				///< Linear operator A (Hermitian)
	   prox_f prox,				///< Proximal operator, ex, soft-threshold
	   float* x,				///< Optimization variable, ex, image
	   const float* b);			///< Observed data, ex, zero-filled image


/**
 * @brief Split bregman
 *
 * TODO
 */
void splitbreg(unsigned int iter,		 ///< Number of iterations
	       float tol,			 ///< Termination tolerance
	       float mu,                         ///< TODO
	       float lambda,                     ///< TODO
	       long N,				 ///< Length of variable x
	       void* data,			 ///< Data structure for linear operator A
	       const struct vec_iter_s* vops,	 ///< Vector arithmetic operator (CPU/GPU)
	       op_f op,				 ///< Linear operator A (Hermitian)
	       prox_f prox,			 ///< Proximal operator, ex, soft-threshold
	       float* x,			 ///< Optimization variable, ex, image
	       const float* b,			 ///< Observed data, ex, zero-filled image
	       obj_f obj);			 ///< Objective function (optional)


/**
 * @brief Iterative regularized Gauss-Newton method
 *
 * TODO
 */
void irgnm_t(unsigned int iter,			 ///< Number of iterations
	     float stepsize,			 ///< Step size
	     float lambda,                       ///< TODO
	     float redu,                         ///< TODO
	     void* data,			 ///< Data structure for linear operator A
	     long N,				 ///< Length of variable x
	     long M,				 ///< Length of Ax
	     const struct vec_iter_s* vops,	 ///< Vector arithmetic operator (CPU/GPU)
	     op_f op,				 ///< Linear operator A 
	     op_f adj,				 ///< Linear operator adjoint A^T
	     inv_f inv,				 ///< Inverse linear operator (A + alphaI)^-1
	     prox_f prox,			 ///< Proximal operator, ex, soft-threshold
	     float* x,				 ///< Optimization variable, ex, image
	     const float* x0,			 ///< Initial estimate of x
	     const float* y);			 ///< Observed data, ex, zero-filled image


/**
 * @brief Projection over convex sets
 *
 * Find x such that x \in C_i
 * where C_i's are convex sets
 */
void pocs(unsigned int iter,			 ///< Number of iterations
	  unsigned int D,                        ///< Number of projection operators
	  const struct pocs_proj_op* proj_ops,   ///< Projection operators
	  const struct vec_iter_s* vops,	 ///< Vector arithmetic operator (CPU/GPU)
	  long N,				 ///< Length of variable x
	  float* x,				 ///< Optimization variable, ex, image
	  const float* x_truth,			 ///< Ground truth (optional)
	  void* obj_data,			 ///< Objective data structure (optional)
	  obj_f obj);				 ///< Objective function (optional)

/**
 * @brief Power iteration
 *
 * @return \max_x x^T A x
 */
double power(unsigned int iter,			///< Number of iterations
	     long N,				///< Length of variable x
	     void* data,			///< Data structure for linear operator A
	     const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU/GPU)
	     op_f op,				///< Linear operator A 
	     float* x);				///< Optimization variable
	   

#endif // __ITALGOS_H



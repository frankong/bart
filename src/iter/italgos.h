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

#ifndef __PROX_FUN_T
#define __PROX_FUN_T
typedef void (*prox_fun_t)(void* prox_data, float rho, float* z, const float* x_plus_u);
#endif


typedef void (*op_f)(void* data, float* dst, const float* src);
typedef void (*prox_f)(void* data, float lambda, float* dst, const float* src);
typedef float (*obj_f)(const void*, const float*);
typedef void (*inv_f)(void* data, float alpha, float* dst, const float* src);
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

	prox_fun_t proj_fun;
	void* data;
};

/**
 * @brief Conjugate gradient
 *
 * Solve min_x || A x - b ||_2^2 + lambda || x ||_2^2
 * where A is Hermitian, ie A = A^*
 */
float conjgrad(unsigned int iter,		///< Number of iterations
	       float l2lambda,			///< L2 regularization, ie, || x ||_2^2
	       float tol,			///< Terminating tolerance
	       long N,				///< Length of variable x
	       void* data,			///< Data structure for linear operator A
	       const struct vec_iter_s* vops,	///< Vector arithmetic operator, (CPU / GPU)
	       op_f linop,			///< Hermitian linear operator A
	       float* x,			///< Optimization variable, ie, image
	       const float* b,			///< Observed data, ie, kspace
	       const float* x_truth,		///< Ground truth (optional)
	       void* obj_data,			///< Objective function data struction (optional)
	       obj_f obj);			///< Objective function (optional)


float conjgrad_hist(struct iter_history_s* iter_history,///< Data structure for iteration history
		    unsigned int iter,			///< Number of iterations
		    float l2lambda,			///< L2 regularization, ie, || x ||_2^2
		    float tol,				///< Terminating tolerance
		    long N,				///< Length of variable x
		    void* data,				///< Data structure for linear operator A
		    const struct vec_iter_s* vops,	///< Vector arithmetic operator (CPU / GPU)
		    op_f linop,			        ///< Hermitian linear operator A
		    float* x,				///< Optimization variable, ie, image
		    const float* b,			///< Observed data, ie, kspace
		    const float* x_truth,		///< Ground truth (optional)
		    void* obj_data,			///< Objective function data struction (optional)
		    obj_f obj);				///< Objective function (optional)


extern const struct cg_data_s* cg_data_init(long N, const struct vec_iter_s* vops);

extern void cg_data_free(const struct cg_data_s* cgdata, const struct vec_iter_s* vops);

float conjgrad_hist_prealloc(struct iter_history_s* iter_history, 
			     unsigned int iter,
			     float l2lambda,
			     float tol, 
			     long N,
			     void* data,
			     struct cg_data_s* cgdata,
			     const struct vec_iter_s* vops,
			     op_f linop, 
			     float* x,
			     const float* b,
			     const float* x_truth,
			     void* obj_data,
			     obj_f obj);


void landweber(unsigned int iter,
	       float tol,
	       float stepsize,
	       long N,
	       long M,
	       void* data,
	       const struct vec_iter_s* vops,
	       op_f op, 
	       op_f adj, 
	       float* x,
	       const float* b,
	       obj_f obj);

void landweber_sym(unsigned int iter,
		   float tol,
		   float stepsize,	
		   long N, void* data,
		   const struct vec_iter_s* vops,
		   op_f op, 
		   float* x,
		   const float* b);

void ist(unsigned int iter,
	 float tol,
	 float stepsize, 
	 float continuation,
	 _Bool hogwild, 
	 long N,
	 void* data,
	 const struct vec_iter_s* vops,
	 op_f op, 
	 prox_f thresh,
	 void* tdata,
	 float* x,
	 const float* b,
	 const float* x_truth,
	 void* obj_data,
	 obj_f obj);

void fista(unsigned int iter,
	   float tol,
	   float stepsize, 
	   float continuation,
	   _Bool hogwild, 
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   op_f op, 
	   prox_f thresh,
	   void* tdata,
	   float* x,
	   const float* b,
	   const float* x_truth,
	   void* obj_data,
	   obj_f obj);

void pgd( unsigned int iter,	///< Number of iterations
	  float tol,		///< Terminating tolerance
	  float stepsize,	///< Step size
	  float continuation, 
	  _Bool hogwild,
	  long N,
	  const struct vec_iter_s* vops,
	  void* op_data,
	  op_f op,
	  void* prox_data,
	  prox_f prox,
	  float* x,
	  const float* b,
	  const float* x_truth,
	  void* obj_data,
	  obj_f obj);


void irgnm(unsigned int iter,
	   float stepsize,
	   float redu,
	   void* data, 
	   long N,
	   long M,
	   const struct vec_iter_s* vops,
	   op_f op, 
	   op_f adj, 
	   inv_f inv, 
	   float* x,
	   const float* x0,
	   const float* y);

void irgnm2(unsigned int iter,
	    float stepsize,
	    float redu,
	    void* data, 
	    long N,
	    long M,
	    const struct vec_iter_s* vops,
	    op_f op, 
	    op_f adj, 
	    inv2_f inv2,
	    float* x,
	    const float* x0,
	    const float* y);

void split(unsigned int iter,
	   float tol,
	   float mu,
	   float lambda, 
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   op_f op, 
	   prox_f thresh,
	   float* x,
	   const float* b);

void splitbreg(unsigned int iter,
	       float tol,
	       float mu,
	       float lambda,
	       long N,
	       void* data,
	       const struct vec_iter_s* vops,
	       op_f op, 
	       prox_f thresh,
	       float* x,
	       const float* b,
	       obj_f obj);

void irgnm_t(unsigned int iter,
	     float stepsize,
	     float lambda,
	     float redu,
	     void* data,
	     long N,
	     long M,
	     const struct vec_iter_s* vops,
	     op_f op, 
	     op_f adj,
	     inv_f inv,
	     prox_f thresh,
	     float* x,
	     const float* x0,
	     const float* y);


void pocs(unsigned int iter,
	  unsigned int D,
	  const struct pocs_proj_op* proj_ops, 
	  const struct vec_iter_s* vops,
	  long N,
	  float* x,
	  const float* x_truth,
	  void* obj_data,
	  obj_f obj);

double power(unsigned int iter,
	     long N,
	     void* data,
	     const struct vec_iter_s* vops,
	     op_f op, 
	     float* u);
	   

#endif // __ITALGOS_H



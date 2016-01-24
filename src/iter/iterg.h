/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

/**
 * Generic iterative function iterface
 *
 * This file contains algorithms that does not fit iter or iter2 format
 */


#ifndef __ITERG_H
#define __ITERG_H

struct operator_s;
struct operator_p_s;

typedef _Bool (*ls_f)(const void *fdata, float alpha, const float* x_new, const float* gradfx, const float* x );


struct iterg_irgnm_conf {

	int iter;
	float alpha;
	float redu;
};

void iter3_irgnm (void* _conf,
		  void (*frw)(void* _data, float* dst, const float* src),
		  void (*der)(void* _data, float* dst, const float* src),
		  void (*adj)(void* _data, float* dst, const float* src),
		  void* data2,
		  long N, float* dst, long M, const float* src);



struct iterg_landweber_conf {

	int iter;
	float alpha;
	float epsilon;
};

void iterg_landweber(void* _conf,
		     void (*frw)(void* _data, float* dst, const float* src),
		     void (*adj)(void* _data, float* dst, const float* src),
		     void* data2,
		     long N, float* dst, long M, const float* src);



struct iterg_pgd_conf {

	int iter; ///< Number of iteration
	float alpha; ///< Step size
	float tol; ///< Termination tolerance
};

/**
 * @brief Projected gradient descent
 * 
 */
void iterg_pgd( void* _conf,				///< Configuration structure
		const struct operator_s* gradf,		///< Gradient of f
		const struct operator_p_s* proxg,	///< Proximal operator of g
		long size,				///< Length of image
 		float* image,				///< Image
		ls_f linesearch );			///< Linesearch function (optional)



#endif


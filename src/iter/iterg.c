/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2016       Frank Ong <frnakong@berkeley.edu>
 */

#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"

#include "iter/italgos.h"
#include "iter/vec.h"

#include "iterg.h"



struct irgnm_s {

	void (*frw)(void* _data, float* dst, const float* src);
	void (*der)(void* _data, float* dst, const float* src);
	void (*adj)(void* _data, float* dst, const float* src);

	void* data;
	float* tmp;

	long size;
};

static void normal(void* _data, float* dst, const float* src)
{
	struct irgnm_s* data = _data;

	data->der(data->data, data->tmp, src);
	data->adj(data->data, dst, data->tmp);
}

static void inverse(void* _data, float alpha, float* dst, const float* src)
{
        struct irgnm_s* data = _data;

	md_clear(1, MD_DIMS(data->size), dst, FL_SIZE);

        float eps = md_norm(1, MD_DIMS(data->size), src);
        conjgrad(100, alpha, 0.1f * eps, data->size, (void*)data, select_vecops(src), normal, dst, src, NULL, NULL, NULL);
}

static void forward(void* _data, float* dst, const float* src)
{
	struct irgnm_s* data = _data;

	data->frw(data->data, dst, src);
}

static void adjoint(void* _data, float* dst, const float* src)
{
	struct irgnm_s* data = _data;

	data->adj(data->data, dst, src);
}




void iterg_irgnm(void* _conf,
		void (*frw)(void* _data, float* dst, const float* src),
		void (*der)(void* _data, float* dst, const float* src),
		void (*adj)(void* _data, float* dst, const float* src),
		void* data2,
		long N, float* dst, long M, const float* src)
{
	struct iterg_irgnm_conf* conf = _conf;

	float* tmp = md_alloc_sameplace(1, MD_DIMS(M), FL_SIZE, src);
	struct irgnm_s data = { frw, der, adj, data2, tmp, N };

	float* x0 = md_alloc_sameplace(1, MD_DIMS(N), FL_SIZE, src);
	md_copy(1, MD_DIMS(N), x0, dst, FL_SIZE);

	irgnm(conf->iter, conf->alpha, conf->redu, &data, N, M, select_vecops(src),
		forward, adjoint, inverse, dst, x0, src);

	md_free(x0);
	md_free(tmp);
}



void iterg_landweber(void* _conf,
		void (*frw)(void* _data, float* dst, const float* src),
		void (*adj)(void* _data, float* dst, const float* src),
		void* data2,
		long N, float* dst, long M, const float* src)
{
	struct iterg_landweber_conf* conf = _conf;

	float* tmp = md_alloc_sameplace(1, MD_DIMS(N), FL_SIZE, src);

	landweber(conf->iter, conf->epsilon, conf->alpha, N, M,
		data2, select_vecops(src), frw, adj, dst, src, NULL);

	md_free(tmp);
}




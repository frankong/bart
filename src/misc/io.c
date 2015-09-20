/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/version.h"
#include "misc/misc.h"

#include "io.h"


struct history_s {

	const char* tag;
	const char* cmd;
	struct history_s* prev;
};

static struct history_s* history = NULL;


int write_cfl_header(int fd, int n, const long dimensions[n])
{
	char header[4096];
	memset(header, 0, 4096);

	int pos = 0;

	pos += snprintf(header + pos, 4096 - pos, "# Dimensions\n");

	for (int i = 0; i < n; i++)
		pos += snprintf(header + pos, 4096 - pos, "%ld ", dimensions[i]);

	pos += snprintf(header + pos, 4096 - pos, "\n");

	if (NULL != command_line) {

		pos += snprintf(header + pos, 4096 - pos, "# Command\n");
		pos += snprintf(header + pos, 4096 - pos, "%s\n", command_line);
	}

	pos += snprintf(header + pos, 4096 - pos, "# Creator\n");
	pos += snprintf(header + pos, 4096 - pos, "BART %s\n", bart_version);

	if (history) {

		struct history_s* h = history;

		pos += snprintf(header + pos, 4096 - pos, "# History\n");

		while (h) {

			pos += snprintf(header + pos, 4096 - pos, "%s: %s\n", h->tag, h->cmd);
			h = h->prev;
		}
	}

	if (pos != write(fd, header, pos))
		return -1;

	return 0;
}



int read_cfl_header(const char* tag, int fd, int n, long dimensions[n])
{
	char header[4097];
	memset(header, 0, 4097);

	int max;
	if (0 > (max = read(fd, header, 4096)))
		return -1;

	int pos = 0;
	int delta;
	bool ok = false;

	char* keyword;

	while (true) {

		// skip lines not starting with '#'

		while ('#' != header[pos]) {

			if ('\0' == header[pos])
				goto out;

			if (0 != sscanf(header + pos, "%*[^\n]\n%n", &delta))
				return -1;

			pos += delta;
		}

		if (1 == sscanf(header + pos, "# %ms\n%n", &keyword, &delta)) {

			pos += delta;

			if (0 == strcmp(keyword, "Dimensions")) {

				for (int i = 0; i < n; i++)
					dimensions[i] = 1;

				long val;
				int i = 0;

				while (1 == sscanf(header + pos, "%ld%n", &val, &delta)) {

					pos += delta;

					if (i < n)
						dimensions[i] = val;
					else
					if (1 != val)
						return -1;

					i++;
				}

				if (0 != sscanf(header + pos, "\n%n", &delta))
					return -1;

				pos += delta;

				if (ok)
					return -1;

				ok = true;

			} else
			if (   (0 == strcmp(keyword, "History"))
			    || (0 == strcmp(keyword, "Command"))) {

				int c = 0;

				while ('#' != header[pos]) {

					if ('\0' == header[pos])
						goto out;

					char* cmd;

					if (1 != sscanf(header + pos, "%m[^\n]\n%n", &cmd, &delta))
						return -1;

					if (c++ < 5) {

						struct history_s* h = xmalloc(sizeof(struct history_s));
						h->tag = strdup(tag);
						h->cmd = cmd;
						h->prev = history;
						history = h;

					} else {

						free(cmd);
					}

					pos += delta;
				}
			}

			free(keyword);
		}
	}

out:
	return ok ? 0 : -1;
}




int write_coo(int fd, int n, const long dimensions[n])
{
	char header[4096];
	memset(header, 0, 4096);

	int pos = 0;

	pos += snprintf(header, 4096, "Type: float\nDimensions: %d\n", n);

	long start = 0;
	long stride = 1;

	for (int i = 0; i < n; i++) {

		long size = dimensions[i];

		pos += snprintf(header + pos, 4096 - pos, "[%ld\t%ld\t%ld\t%ld]\n", start, stride * size, size, stride);
		stride *= size;
	}

	if (4096 != write(fd, header, 4096))
		return -1;

	return 0;
}


int read_coo(int fd, int n, long dimensions[n])
{
	char header[4096];

	if (4096 != read(fd, header, 4096))
		return -1;

	int pos = 0;
	int delta;

	if (0 != sscanf(header + pos, "Type: float\n%n", &delta))
		return -1;

	pos += delta;

	int dim;
	
	if (1 != sscanf(header + pos, "Dimensions: %d\n%n", &dim, &delta))
		return -1;

	pos += delta;

//	if (n != dim)
//		return -1;
	
	for (int i = 0; i < n; i++)
		dimensions[i] = 1;

	for (int i = 0; i < dim; i++) {

		long val;
		
		if (1 != sscanf(header + pos, "[%*d %*d %ld %*d]\n%n", &val, &delta))
			return -1;

		pos += delta;
		
		if (i < n)
			dimensions[i] = val;
		else
		if (1 != val)	// fail if we have to many dimensions not equal 1
			return -1;
	}

	return 0;
}






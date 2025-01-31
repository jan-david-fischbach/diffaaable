
# TODO
def reduce(self, tol=1e-3, max_iters=None):

        svs = self.sampling_values
        samples = self.samples

        # add complex conjugate samples
        if self.conjugate:
            for i in range(len(svs[0])):
                s = svs[0][i]
                if np.conj(s) not in svs[0]:
                    svs[0] = np.append(svs[0], np.conj(s))
                    samples = np.vstack((samples, np.conj(samples[i, None])))

        num_vars = len(svs)
        max_samples = np.max(np.abs(samples))
        rel_tol = tol * max_samples

        # Transform samples for MIMO case
        if len(samples.shape) != len(svs):
            assert len(samples.shape) == len(svs) + 2
            dim_input = samples.shape[-2]
            dim_output = samples.shape[-1]
            samples_T = np.empty(samples.shape[:-2], dtype=samples.dtype)
            w = np.random.uniform(size=(1, dim_input))
            v = np.random.uniform(size=(dim_output, 1))
            w = w / np.linalg.norm(w)
            v = v / np.linalg.norm(v)
            for li in list(itertools.product(*(range(s) for s in samples.shape[:-2]))):
                samples_T[li] = w @ samples[li] @ v
            samples_orig = samples
            samples = samples_T
        else:
            dim_input = 1
            dim_output = 1

        # initilize data partitions, error, max iterations
        err = np.inf
        itpl_part = [*([] for _ in range(num_vars))]
        if max_iters is None:
            max_iters = [*(len(s)-1 for s in svs)]

        assert len(max_iters) == len(svs)

        # start iteration with constant function
        bary_func = np.vectorize(lambda *args: np.mean(samples))

        # iteration counter
        j = 0

        while np.any([*(len(i) < mi for (i, mi) in zip(itpl_part, max_iters))]):

            # compute approximation error over entire sampled data set
            grid = np.meshgrid(*(sv for sv in svs), indexing='ij')
            err_mat = np.abs(bary_func(*(g for g in grid))-samples)

            # set errors to zero such that new interpolation points are consistent with max_iters
            zero_idx = []
            for i in range(num_vars):
                if len(itpl_part[i]) >= max_iters[i]:
                    zero_idx.append(list(range(samples.shape[i])))
                else:
                    zero_idx.append(itpl_part[i])
            err_mat[np.ix_(*(zi for zi in zero_idx))] = 0
            err = np.max(err_mat)

            j += 1
            logger.info(f'Relative error at step {j}: {err/max_samples:.5e}, number of interpolation points {[*(len(ip) for ip in itpl_part)]}')

            # stopping criterion based on relative approximation error
            if err <= rel_tol:
                break

            greedy_idx = np.unravel_index(err_mat.argmax(), err_mat.shape)
            for i in range(num_vars):
                if greedy_idx[i] not in itpl_part[i] and len(itpl_part[i]) < max_iters[i]:
                    itpl_part[i].append(greedy_idx[i])

                    # perform double interpolation step to enforce real state-space representation
                    if i == 0 and self.conjugate and np.imag(svs[i][greedy_idx[i]]) != 0:
                        conj_sample = np.conj(svs[i][greedy_idx[i]])
                        conj_idx = np.where(svs[0] == conj_sample)[0]
                        itpl_part[i].append(conj_idx[0])

            # solve LS problem
            L = full_nd_loewner(samples, svs, itpl_part)

            _, S, V = np.linalg.svd(L)
            VH = np.conj(V.T)
            coefs = VH[:, -1:]

            # post-processing for non-minimal interpolants
            d_nsp = np.sum(S/S[0] < self.nsp_tol)
            if d_nsp > 1:
                if self.post_process:
                    logger.info('Non-minimal order interpolant computed. Starting post-processing.')
                    pp_coefs, pp_itpl_part = _post_processing(samples, svs, itpl_part, d_nsp, self.L_rk_tol)
                    if pp_coefs is not None:
                        coefs, itpl_part = pp_coefs, pp_itpl_part
                    else:
                        logger.warning('Post-processing failed. Consider reducing "L_rk_tol".')
                else:
                    logger.warning('Non-minimal order interpolant computed.')

            # update barycentric form
            itpl_samples = samples[np.ix_(*(ip for ip in itpl_part))]
            itpl_samples = np.reshape(itpl_samples, -1)
            itpl_nodes = [*(sv[lp] for sv, lp in zip(svs, itpl_part))]
            bary_func = np.vectorize(make_bary_func(itpl_nodes, itpl_samples, coefs))

            if self.nsp_conv and d_nsp >= 1:
                logger.info('Converged due to non-trivial null space of Loewner matrix.')
                break

        # in MIMO case construct barycentric form based on matrix/vector samples
        if dim_input != 1 or dim_output != 1:
            itpl_samples = samples_orig[np.ix_(*(ip for ip in itpl_part))]
            itpl_samples = np.reshape(itpl_samples, (-1, dim_input, dim_output))
            bary_func = make_bary_func(itpl_nodes, itpl_samples, coefs, dim_input, dim_output)

        return TransferFunction(dim_input, dim_output, lambda s, mu: bary_func(s, *(mu[p] for p in self.parameters)), parameters=self.parameters)

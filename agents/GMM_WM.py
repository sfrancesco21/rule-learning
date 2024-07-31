import numpy as np
from scipy.stats import multivariate_normal, chi2
import random
import copy
from scipy.special import logsumexp
from scipy.spatial.distance import pdist, squareform
from random import shuffle, sample


class GMMWM_Agent:
    def __init__(self, config):
        # Initialise time
        self.t = 0
        # Set tot resources
        self.tot_memory = config['Total memory']
        # Initialise max number of 
        self.min_clusters = 2
        self.max_clusters = max(self.min_clusters, np.round(self.tot_memory*config['Cluster strategy weight']))
        # Flexible strategy?
        self.switch = config['Fluid strategy']
        # threshold for new cluster formation
        self.threshold = config['Threshold']
        # initial uncertainty
        self.start_p = config['Initial uncertainty']
        # initial variance of clusters
        self.sigma0 = config['Sigma0']
        # Initialise working memory
        self.working_memory = []
        # Initialise alphas
        self.alpha = []
        # Initialise means
        self.mu = []
        # Initialise covariance matrices
        self.Sigma = []
        # Initialise cumulative variable
        self.v = []
        # set time for which a cluster is safe
        self.ceasefire = config["Minimum cluster lifetime"]
        # set minimum cluster weight
        self.min_w = config['Minimum cluster weight']
        # Initialise cumulative posterior
        self.sp = []
        # Initialise cluster label
        self.label = []
        # Initialise stimuli labels
        self.stim_labels = []
        # Initialise rewards
        self.reward_train = []
        self.reward_test = []
        # Learning rate
        self.lr = config['Strategy learning rate']
        # Initialise cluster responsibilities
        self.r = []
        # Initialise working memory strategy probabilities
        self.wm_p = []
        # Initialise cluster strategy probability
        self.cl_p = []
        # Initialise relative weight of clustering model
        self.phi = config['Cluster strategy weight']
        self.phis = [config['Cluster strategy weight']]
        # Set strategy for pruning clusters:
        self.pruning = config['Pruning strategy']
        
    def infer_cluster(self, stimulus):
        logjoint = []
        for alpha, mu, sigma in zip(self.alpha, self.mu, self.Sigma):
            lj = np.log(alpha/np.sum(self.alpha)) + multivariate_normal.logpdf(stimulus, mu, sigma)
            logjoint.append(lj)
        r = np.exp(logjoint)/np.sum(np.exp(logjoint))
        return r
        
    def infer_wm(self, stimulus):
        # Euclidean distance
        unnorm = [self.start_p, self.start_p]
        for x in self.working_memory:
            d = np.linalg.norm(x['position'] - stimulus)
            unnorm[x['label']] += 1/(d+1e-6)
        p = unnorm/np.sum(unnorm)
        return np.array(p)
    
    def decision(self, stimulus):
        if len(self.label) == 0:
            ap = [0.5, 0.5]
            self.r.append(np.array([1]))
            self.cl_p.append([-1, -1])
            self.wm_p.append([-1, -1])
        else:
            r = self.infer_cluster(stimulus)
            wm_p = self.infer_wm(stimulus)
            cl_p = np.array([np.sum((1-np.array(self.label))*r), np.sum(np.array(self.label)*r)])
            ap = self.phi*cl_p + (1-self.phi)*wm_p
            self.r.append(r)
            self.wm_p.append(wm_p)
            self.cl_p.append(cl_p)
        return ap
    
    def generate_feedback(self, true_label, p, feedback):
        self.stim_labels.append(true_label)
        if np.argmax(p) == true_label:
            reward = 1
        else:
            reward = 0
        if feedback == True:
            self.reward_train.append(reward)
        else:
            self.reward_test.append(reward)
    
    def add_cluster(self, stimulus, label):
        self.mu.append(stimulus)
        self.Sigma.append(self.sigma0*np.eye(2))
        self.v.append(1)
        self.sp.append(0)
        self.alpha.append(1)
        self.label.append(label)

    def kl_divergence(self, mu_p, cov_p, mu_q, cov_q):
        """
        Calculate the KL divergence between two bivariate Gaussian distributions.

        Parameters:
        mu_p (np.array): Mean vector of the first Gaussian (2,)
        cov_p (np.array): Covariance matrix of the first Gaussian (2, 2)
        mu_q (np.array): Mean vector of the second Gaussian (2,)
        cov_q (np.array): Covariance matrix of the second Gaussian (2, 2)

        Returns:
        float: KL divergence D_KL(P || Q)
        """
        dim = mu_p.shape[0]
        # Calculate the inverse of the covariance matrix of Q
        inv_cov_q = np.linalg.inv(cov_q)
        # Calculate the difference in means
        mean_diff = mu_q - mu_p
        # Calculate the trace term
        trace_term = np.trace(inv_cov_q @ cov_p)
        # Calculate the quadratic form term
        quadratic_term = mean_diff.T @ inv_cov_q @ mean_diff
        # Calculate the log determinant term
        log_det_term = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
        # Calculate the KL divergence
        kl_div = 0.5 * (trace_term + quadratic_term - dim + log_det_term)

        return kl_div

    def merging_check(self, idx1 = None):
        if idx1 != None:
            loop1 = [idx1]
        else:
            loop1 = range(len(self.label))
        min_crit = 1e6
        idx1 = -1
        idx2 = -1
        for i in loop1:
            for j in range(len(self.label)):
                if self.label[i] == self.label[j] and i != j and self.v[i] > self.ceasefire and self.v[j] > self.ceasefire:
                    if self.pruning['criterion'] == 'KLD':
                        crit = self.kl_divergence(self.mu[i], self.Sigma[i], self.mu[i], self.Sigma[i])
                        crit = np.linalg.norm(self.mu[i] - self.mu[j])
                    elif self.pruning['criterion'] == 'euclidean distance':
                        x1, y1 = self.mu[i]
                        x2, y2 = self.mu[j]
                        crit = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    elif self.pruning['criterion'] == 'mahalanobis distance':
                        d1 = np.matmul(np.reshape(self.mu[i]-self.mu[j], (1, 2)), np.linalg.inv(self.Sigma[j]))
                        d1 = np.matmul(d1, np.reshape(self.mu[i]-self.mu[j], (2, 1)))
                        d2 = np.matmul(np.reshape(self.mu[j]-self.mu[i], (1, 2)), np.linalg.inv(self.Sigma[i]))
                        d2 = np.matmul(d2, np.reshape(self.mu[j]-self.mu[i], (2, 1)))
                        crit = d1+d2
                    else:
                        print('Unrecognised criterion')
                    if crit < min_crit:
                        min_crit = copy.deepcopy(crit)
                        idx1 = i
                        idx2 = j
        return idx1, idx2    
    
    def remove_elements_by_indices(self, input_list, indices):
        # Ensure indices are in the correct order
        if len(indices) > 1:
            indices = sorted(indices, reverse=True)
        for index in indices:
            if 0 <= index < len(input_list):
                input_list.pop(index)
        return input_list

    def cluster_merging(self, idx1, idx2):
        if self.label[idx1] != self.label[idx2]:
            print('Cluster labels are different, something is wrong')
        label = self.label[idx1]
        alpha = self.alpha[idx1] + self.alpha[idx2]
        v = max(self.v[idx1], self.v[idx2])
        sp = 0

        # print('Merging!')
        # print('Cluster1:')
        # print('Mean:', self.mu[idx1])
        # print('Covariance:', self.Sigma[idx1])
        # print("")
        # print('Cluster2:')
        # print('Mean:', self.mu[idx2])
        # print('Covariance:', self.Sigma[idx2])
        # print("")
        # print("")

        samples1 = np.random.multivariate_normal(self.mu[idx1], self.Sigma[idx1], int(self.alpha[idx1]*1e3))
        samples2 = np.random.multivariate_normal(self.mu[idx2], self.Sigma[idx2], int(self.alpha[idx2]*1e3))

        # Combine the samples
        combined_samples = np.vstack((samples1, samples2))

        # Fit a single bivariate Gaussian to the combined samples
        mean_combined = np.mean(combined_samples, axis=0)
        cov_combined = np.cov(combined_samples, rowvar=False)

        self.mu = self.remove_elements_by_indices(self.mu, [idx1, idx2])
        self.Sigma = self.remove_elements_by_indices(self.Sigma, [idx1, idx2])
        self.label = self.remove_elements_by_indices(self.label, [idx1, idx2])
        self.v = self.remove_elements_by_indices(self.v, [idx1, idx2])
        self.sp = self.remove_elements_by_indices(self.sp, [idx1, idx2])
        self.alpha = self.remove_elements_by_indices(self.alpha, [idx1, idx2])

        self.mu.append(mean_combined)
        self.Sigma.append(cov_combined)
        self.label.append(label)
        self.v.append(v)
        self.sp.append(sp)
        self.alpha.append(alpha)
       
    def erase_cluster(self):
        if self.pruning['criterion'] == 'smallest':
            smallest = 1e12
            idx = None
            for i in range(len(self.alpha)):
                if self.alpha[i] < smallest and self.v[i] > self.ceasefire:
                    idx = copy.deepcopy(i)
                    smallest = copy.deepcopy(self.alpha[i])
                elif self.alpha[i] == smallest and self.v[i] > self.ceasefire and idx != None:
                    if self.v[i] > self.v[idx]:
                        idx = copy.deepcopy(i)
                        smallest = copy.deepcopy(self.alpha[i])
                        
        elif self.pruning['criterion'] == 'oldest':
            idx = np.argmax(self.sp)

        idx = int(idx)

        self.mu = self.remove_elements_by_indices(self.mu, [idx])
        self.Sigma = self.remove_elements_by_indices(self.Sigma, [idx])
        self.label = self.remove_elements_by_indices(self.label, [idx])
        self.v = self.remove_elements_by_indices(self.v, [idx])
        self.sp = self.remove_elements_by_indices(self.sp, [idx])
        self.alpha = self.remove_elements_by_indices(self.alpha, [idx])
    
    def is_positive_semi_definite(self, matrix):
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues >= 0)
    
    def learn(self, stimulus, label):
        d_wm = np.argmax(self.wm_p[-1])
        d_cl = np.argmax(self.cl_p[-1])
        if d_wm == label:
            r_wm = 1
        else:
            r_wm = -1
        if d_cl == label:
            r_cl = 1
        else:
            r_cl = -1
        if self.wm_p[-1][0] > -0.5 and self.switch:
            Q_wm = self.wm_p[-1][label]*r_wm#*(1-self.phi)
            Q_cl = self.cl_p[-1][label]*r_cl#*self.phi
            Q = np.array([self.phi, 1-self.phi]) + self.lr*np.array([Q_cl, Q_wm])
            for i in range(len(Q)):
                if Q[i] < 0:
                    Q[i] = 0
            self.phi = Q[0]/np.sum(Q)
            self.phis.append(self.phi)
            self.max_clusters = max(self.min_clusters, np.round(self.tot_memory*self.phi))
        r = self.r[-1]*(label == self.stim_labels[-1])
        if len(self.alpha) > 0:
            for i in range(len(self.alpha)): 
                self.v[i] += 1
                self.sp[i] += 1

        if len(self.r) == 1 or np.sum(r) == 0:
            #print('Adding cluster')
            self.add_cluster(stimulus, label)
        else:
            r = r/np.sum(r)
            for n, resp in enumerate(r):
                if resp < 0.001:
                    r[n] = 0
            mc = np.argmax(r)
            dist = np.matmul(np.reshape(stimulus-self.mu[mc], (1, 2)), np.linalg.inv(self.Sigma[mc]))
            dist = np.matmul(dist, np.reshape(stimulus-self.mu[mc], (2, 1)))
            #print(dist, chi2.cdf(dist, 2))
            if chi2.cdf(dist, 2) > self.threshold:
                # print('Adding cluster')
                # print('Stimulus:', stimulus)
                # print('Chi2:', chi2.cdf(dist, 2))
                # print("")
        
                self.add_cluster(stimulus, label)
                
            else:
                # print('Update existing clusters')
                # print('Stimulus:', stimulus)
                # print('Responsibilities:', r)
                # print('Winning cluster:', mc, r[mc])
                # print("")

                for i in range(len(r)): 
                    #updates
                    self.alpha[i] += r[i]
                    e1 = stimulus - self.mu[i]
                    w = r[i]/self.alpha[i]
                    dm = w*e1
                    self.mu[i] = self.mu[i] + dm
                    e = stimulus - self.mu[i]

                    #self.Sigma[i] = (1-w)*self.Sigma[i] + w * np.matmul(e.T, e) - np.matmul(dm.T, dm) + np.eye(len(e)) * 1e-6
                    new_cov = (1 - w) * self.Sigma[i] + w * np.outer(e, e) + np.eye(len(e)) * 1e-10 - np.outer(dm, dm)

                    if self.is_positive_semi_definite(new_cov):
                        self.Sigma[i] = new_cov
                    else:
                        self.Sigma[i] = (1 - w) * self.Sigma[i] + w * np.outer(e, e) + np.eye(len(e)) * 1e-10
                        print('INVALID COVARIANCE MATRIX')
                        print('cluster', i)
                        print('resp', r[i])
                        print('mean', self.mu[i])
                        print('stimulus', stimulus)
                        print('sigma:')
                        print(self.Sigma[i])
                        print('w:', w)
                        print('e1:', e1)
                        print('e2:', e)
                        print('diff:')
                        print(w*np.outer(e, e) - np.outer(dm, dm))
                
                self.sp[int(np.argmax(r))] = 0

                
    def trial(self, stimulus, true_label, feedback):
        action_prob = self.decision(stimulus)
        self.working_memory.append(
            {
                'position' : stimulus,
                'label' : true_label
            }
        )
        
        self.generate_feedback(true_label, action_prob, feedback)

        if feedback:

            self.learn(stimulus, true_label)

            alpha_norm = self.alpha/np.sum(self.alpha)
            to_erase = []
            for i, p in enumerate(alpha_norm):
                if p < 0.05 and self.v[i] > self.ceasefire:
                    to_erase.append(i)
                    #print('cluster too small, merging it')
            for i in to_erase:
                idx1 = copy.deepcopy(i)
                if self.pruning['name'] == 'merging':
                    idx1, idx2 = self.merging_check(idx1)
                    self.cluster_merging(idx1, idx2)
                elif self.pruning['name'] == 'erasing':
                    self.erase_cluster()

            if len(self.label) > self.max_clusters:
                if self.pruning['name'] == 'merging':
                    idx1, idx2 = self.merging_check()
                    self.cluster_merging(idx1, idx2)
                elif self.pruning['name'] == 'erasing':
                    self.erase_cluster()

        wm_size = self.tot_memory - len(self.label)
        self.working_memory = self.working_memory[-wm_size:]
        return action_prob

    def __call__(self, stimuli, labels, feedback = True):
        if feedback == False:
            self.phi = 1
        action_probs = []
        for stimulus, true_label in zip(stimuli, labels):
            ap = self.trial(np.array(stimulus), true_label, feedback)
        action_probs.append(ap)
        return action_probs
        
      